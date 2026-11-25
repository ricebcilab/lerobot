"""The pi0.5 bundle: the checkpoint, its action statistics, and the steering policy's one control path.

The station's action is 8 numbers in the checkpoint's action units; pi0.5-DROID was trained on 32-wide
state and action vectors whose first 8 dims are the arm's joints and the gripper, everything beyond
padding. The published checkpoint (``lerobot/pi05_droid``) does not carry the normalizer's statistics:
its cached pre/post-processor JSONs declare the features empty and the ``normalization_mapping`` says
``QUANTILES`` for STATE and ACTION, so the q01/q99 statistics have to come from openpi's
``assets/droid/norm_stats.json`` and be installed here before the pipelines mean anything
(``DESIGN.md``, "What the checkpoint's own code settles").

This module is the only place torch meets the checkpoint in the example. ``load_bundle`` builds the
policy and its two pipelines on the configured device (weights and the gated PaliGemma tokenizer are
needed for a real one: the probe on the station is that [V]); ``install_stats`` puts the statistics on
the steps whose ``QUANTILES`` mode reads them; ``make_steering`` wraps the loaded policy in the mode's
``SteeringPolicy``; ``Bundle.action`` is the one call a control tick makes, with or without the
operator's 7-dim command; and ``teleop_action`` is the no-policy path's 8-wide action. ``fake_bundle``
is the numpy-only stand-in ``--dry-run`` uses, so importing this module and driving it must not need
torch, the checkpoint or the tokenizer: every lerobot import is inside the function that needs it.

Where the 8 dims are sliced: **after** the post-processor, never before. The model works in normalized
space over the 32-wide padded action, and the unnormalizer's QUANTILES map is a per-dim affine map
(``q01``/``q99``); the station's action units are that map's output, so the first 8 dims are taken from
its result. Slicing the normalized action before it would hand the station numbers in [-1, 1] with no
physical meaning, and the steering cores' statistics (``FlowControlSteering.action_mean_std``) are the
affine form of the same quantile map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from config import Config, SteeringConfig

    from lerobot.policies.steering import SteeringPolicy

ARM_DIMS = 7
"""The Franka's joints: the steering command's width, and dims 0-6 of the action."""

STATION_DIMS = 8
"""The station's action width: the seven joints and the gripper (dim 7)."""

CHECKPOINT_WIDTH = 32
"""pi0.5-DROID's padded state/action width; the statistics are installed at this width."""

DEFAULT_DEADBAND = 0.02
"""The operator's idle threshold in action units (DESIGN's ``[command] deadband`` default)."""

REFERENCE_STEP = 1.0
"""FRS's action units one full deflection commands per control step (DESIGN's ``[steering] reference_step``).

``configs/franka.toml``'s schema has ``[steering] reference_step = 1.0``; the merged ``config.py``
dataclass has no field for it, so this module carries the study value (``make_steering`` takes a
config's own ``reference_step`` when a config provides one)."""

STEERING_MODES = ("none", "flow_control", "frs")
"""The modes ``make_steering`` builds, matching ``config.STEERING_MODES`` and the steering config's."""


@dataclass(frozen=True)
class Bundle:
    """A loaded pi0.5, its pipelines and its action statistics: everything one control tick reads.

    ``policy``, ``preprocessor`` and ``postprocessor`` are the real objects on the configured device
    (``None`` on the fake bundle). ``mean``/``std`` are the action statistics in the steering cores'
    affine form, 8 wide and in raw action units (``None`` when no statistics file was installed).
    ``deadband`` is the operator's idle threshold (``config.command.deadband``): it lives outside the
    ``[steering]``/``[modes.*]`` tables in the config, and ``make_steering(bundle, steering)`` has no
    other place to read it. ``_cache`` holds the one ``SteeringPolicy`` ``make_steering`` makes (the
    loop calls ``action`` about 15 times a second, so it is made once and reset per episode) and the
    fake bundle's fixed action.
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    mean: NDArray | None = None
    std: NDArray | None = None
    deadband: float = DEFAULT_DEADBAND
    _cache: dict = field(default_factory=dict, repr=False, compare=False, hash=False)

    def action(self, raw_batch: dict, command7: NDArray | None = None) -> NDArray:
        """The station's 8 numbers for one observation, in the checkpoint's raw action units.

        ``raw_batch`` is the station's raw dict (images, 32-wide state, the task string): this runs the
        pre-processor, then the base policy (``command7`` None: ``autonomous``) or the mode's
        ``SteeringPolicy`` (``command7``: the operator's 7 joint-velocity command in action units, the
        steered modes), then the post-processor, and returns its first 8 dims as float64.

        Without action statistics there is no action unit to return: the checkpoint's QUANTILES
        normalization is what maps the model's normalized output into action units, so this raises
        instead of handing the station unnormalized numbers. Teleop never reaches this method.
        """
        fixed = self._cache.get("fixed_action")
        if fixed is not None:  # --dry-run's bundle: the stand-in action, whatever the command
            return fixed.copy()
        if self.mean is None or self.std is None:
            raise ValueError(
                "the bundle has no action statistics: pi0.5-DROID's QUANTILES normalization needs openpi's "
                "assets/droid/norm_stats.json (set [policy] stats); only teleop runs without it"
            )
        batch = steering_batch(self, raw_batch)
        if command7 is None:
            action = self.policy.select_action(batch)
        else:
            action = self._steering().select_action(_command_batch(batch, command7, _device(self)))
        return _action_units(self.postprocessor(action))

    def reset(self) -> None:
        """Start a new episode: the base policy's chunk queues and the steering policy's rollout state."""
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        steering = self._cache.get("steering")
        if steering is not None:
            steering.reset()

    def _steering(self) -> SteeringPolicy:
        """The ``SteeringPolicy`` ``make_steering`` made for this bundle and cached (made once, not per tick)."""
        steering = self._cache.get("steering")
        if steering is None:
            raise ValueError(
                "Bundle.action with a command needs the mode's SteeringPolicy: call "
                "make_steering(bundle, steering_cfg) once before the control loop; it caches it here"
            )
        return steering


# ---------------------------------------------------------------- loading


def load_bundle(cfg: Config, device: str | None = None) -> Bundle:
    """pi0.5-DROID on ``device`` (default: ``cfg.policy.device``) with its pipelines and statistics.

    The config's ``pretrained_path``, ``device``, ``dtype`` and ``n_action_steps`` are put on the
    checkpoint's own config; the policy is built by its class (``make_policy`` insists on a dataset or
    env for features, and the checkpoint's config already declares them) and evaluated; the pre/post
    processors are loaded from the checkpoint's JSONs exactly as the LIBERO session loads them,
    overriding the device and the (empty) rename map; then ``install_stats`` is applied when the config
    names a statistics file that is on disk. Without one, ``Bundle.mean``/``std`` are ``None`` (the
    openpi asset is gitignored): the shared modes then refuse to build in ``make_steering`` and
    ``Bundle.action`` refuses to act, and teleop never comes here at all.

    This needs the checkpoint's weights and the gated PaliGemma tokenizer; it cannot run in the
    offline test environment (it runs on the station, ``probe`` first).
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class

    path = str(cfg.policy.path)
    device = str(device if device is not None else cfg.policy.device)

    policy_cfg = PreTrainedConfig.from_pretrained(path)
    policy_cfg.pretrained_path = path
    policy_cfg.device = device
    policy_cfg.dtype = cfg.policy.dtype
    policy_cfg.n_action_steps = cfg.policy.n_action_steps
    policy = get_policy_class(policy_cfg.type).from_pretrained(path, config=policy_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    mean = std = None
    stats = getattr(cfg.policy, "stats", None)
    if stats is not None and Path(stats).is_file():
        mean, std = install_stats(preprocessor, postprocessor, Path(stats))

    command = getattr(cfg, "command", None)
    return Bundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        mean=mean,
        std=std,
        deadband=float(getattr(command, "deadband", DEFAULT_DEADBAND)),
    )


def install_stats(
    preprocessor: Any, postprocessor: Any, stats_path: Path, dims: int = STATION_DIMS
) -> tuple[NDArray, NDArray]:
    """Install openpi's DROID statistics on the steps that read them; return the 8-wide action (mean, std).

    ``stats_path`` is ``assets/droid/norm_stats.json``: ``{"norm_stats": {"state": {...}, "actions":
    {...}}}``, each group ``mean``/``std``/``q01``/``q99`` 32 wide, meaningful (not q01 == q99 == 0) in
    the first ``dims`` dims and zero after. The checkpoint's ``normalization_mapping`` is ``QUANTILES``
    for STATE and ACTION, so a ``NormalizerProcessorStep``/``UnnormalizerProcessorStep`` reads the
    group's **q01/q99** (never mean/std) for the keys it sees: the pre-processor's normalizer gets both
    ``observation.state`` and ``action``, the post-processor's unnormalizer gets ``action``. A step only
    touches the keys in its ``features`` dict (and the ACTION key through ``stats`` alone), and the
    checkpoint's JSONs ship ``features: {}``, so the feature entries are added too and the step's tensor
    statistics are rebuilt.

    The ``(mean, std)`` returned for the steering cores is the affine form of that same quantile map,
    ``(q01 + q99) / 2`` and ``(q99 - q01) / 2`` (the cores' own conversion,
    ``FlowControlSteering.action_mean_std``, is what computes it; this does not re-derive the map).
    """
    path = Path(stats_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error
    groups = data.get("norm_stats") if isinstance(data, dict) else None
    if not isinstance(groups, dict):
        raise ValueError(f"{path}: expected a 'norm_stats' table, got {type(groups).__name__}")

    state = _quantile_pair(groups, "state", path, dims)
    actions = _quantile_pair(groups, "actions", path, dims)

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    stats = {
        "observation.state": {"q01": state["q01"], "q99": state["q99"]},
        "action": {"q01": actions["q01"], "q99": actions["q99"]},
    }
    # The state is normalized through the normalizer's features dict; the action is reached by its
    # stats key, but declaring both keeps each step's feature set the checkpoint's 32-wide one.
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(CHECKPOINT_WIDTH,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(CHECKPOINT_WIDTH,)),
    }
    normalizers = [step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
    unnormalizers = [step for step in postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)]
    if not normalizers:
        raise ValueError(f"{path}: the pre-processor has no NormalizerProcessorStep to install statistics on")
    if not unnormalizers:
        raise ValueError(
            f"{path}: the post-processor has no UnnormalizerProcessorStep to install statistics on"
        )
    for step in (*normalizers, *unnormalizers):
        _install_on(step, stats, features)

    from lerobot.policies.steering import FlowControlSteering

    mean, std = FlowControlSteering.action_mean_std(postprocessor)
    return mean[:dims], std[:dims]


def _quantile_pair(groups: dict, name: str, path: Path, dims: int) -> dict[str, NDArray]:
    """One group of ``norm_stats`` (``state`` or ``actions``): its 32-wide q01/q99, width-checked."""
    group = groups.get(name)
    if not isinstance(group, dict):
        raise ValueError(f"{path}: expected a 'norm_stats.{name}' table, got {type(group).__name__}")
    keys = ("mean", "std", "q01", "q99")
    missing = [key for key in keys if key not in group]
    if missing:
        raise ValueError(f"{path}: norm_stats.{name} is missing {', '.join(missing)}")
    arrays = {key: np.asarray(group[key], dtype=np.float64).reshape(-1) for key in keys}
    widths = sorted({array.shape[0] for array in arrays.values()})
    if widths != [CHECKPOINT_WIDTH]:
        raise ValueError(
            f"{path}: norm_stats.{name} must be {CHECKPOINT_WIDTH} wide, got widths {widths} "
            "(pi0.5-DROID's padded state and action vectors)"
        )
    q01, q99 = arrays["q01"], arrays["q99"]
    # DESIGN.md: dims 0-7 meaningful, zero after. "Meaningful" is the first index where both quantiles
    # are exactly zero (the pad dims), and nothing beyond it may be non-zero.
    zero = (q01 == 0) & (q99 == 0)
    meaningful = int(np.argmax(zero)) if zero.any() else CHECKPOINT_WIDTH
    if meaningful != dims or bool((~zero[dims:]).any()):
        raise ValueError(
            f"{path}: norm_stats.{name} is meaningful in {meaningful} dims (zeros from index {meaningful}); "
            f"expected {dims} meaningful dims and zeros beyond"
        )
    return {"q01": q01, "q99": q99}


def _install_on(step: Any, stats: dict, features: dict) -> None:
    """Merge the two keys into a normalization step and rebuild its tensor statistics from them."""
    step.stats = {**(step.stats or {}), **stats}
    step.features = {**(step.features or {}), **features}
    step.to()  # _NormalizationMixin.to: re-derives _tensor_stats from .stats


# ---------------------------------------------------------------- the steering modes


def make_steering(bundle: Bundle, steering: SteeringConfig) -> SteeringPolicy:
    """The mode's ``SteeringPolicy`` over the bundle's loaded pi0.5 and post-processor, cached on the bundle.

    ``steering`` is one resolved entry of ``config.py`` (a ``[modes.*]`` table over ``[steering]``): its
    mode, the method's knobs and its adapter fields. The Franka's operator space is the 7 joints
    (``n_dims = 7``, ``operator_dims = None``: the whole command is commanded, nothing named) and the
    study's ``reference_step`` is ``REFERENCE_STEP``; the adapter's fixed 7x7 matrix arrives already
    resolved (``adapter_matrix`` rows) because turning ``"corruption"``/``"corruption_inverse"`` into the
    matrix needs the station pose, which is the composition root's job; the learned user and task
    modules are paths. ``SteeringPolicy`` builds the ``AdapterModules`` holder itself from those fields.

    The core is built here (``steering.core``), so the knobs are validated before the first control
    tick, as the LIBERO session does. A mode that steers (flow control and FRS) needs the checkpoint's
    action statistics: without them the cores cannot build, so a bundle without ``mean``/``std`` gets
    a ``ValueError`` naming the missing file, and ``mode = "none"`` (the base policy's own chunk) does
    not need them.
    """
    from lerobot.policies.steering import SteeringConfig as PolicySteeringConfig, SteeringPolicy

    mode = getattr(steering, "mode", "none")
    if mode not in STEERING_MODES:
        raise ValueError(f"unknown steering mode {mode!r} (expected one of {STEERING_MODES})")
    if mode != "none" and (bundle.mean is None or bundle.std is None):
        raise ValueError(
            f"steering mode {mode!r} needs the checkpoint's action statistics, and the bundle has none: "
            "set [policy] stats = assets/droid/norm_stats.json (openpi's assets) and reload the bundle"
        )

    base_config = getattr(bundle.policy, "config", None)
    reference_step = getattr(steering, "reference_step", None)
    fields: dict[str, Any] = {
        "mode": mode,
        "n_dims": ARM_DIMS,
        "operator_dims": None,
        "deadband": float(bundle.deadband),
        "reference_step": REFERENCE_STEP if reference_step is None else float(reference_step),
        "base_policy_type": str(getattr(base_config, "type", "pi05")),
        "device": str(_device(bundle)),
        "adapter_user": None,
        "adapter_task": None,
    }
    for key in ("tau", "lam", "ic_gain", "n_reversal_steps", "pinned_steps"):
        value = getattr(steering, key, None)
        if value is not None:
            fields[key] = value
    adapter_matrix = getattr(steering, "adapter_matrix", None)
    if adapter_matrix is not None:
        fields["adapter_matrix"] = _adapter_matrix(adapter_matrix)
    for key in ("adapter_user", "adapter_task"):
        value = getattr(steering, key, None)
        if value is not None:
            fields[key] = str(value)

    policy = SteeringPolicy(
        PolicySteeringConfig(**fields), base_policy=bundle.policy, postprocessor=bundle.postprocessor
    )
    if mode != "none":
        _ = policy.core  # built now: the mode's knobs are checked before the loop starts
    bundle._cache["steering"] = policy
    return policy


def _adapter_matrix(rows: Any) -> list[list[float]]:
    """A resolved 7x7 adapter matrix as the steering config's rows-of-lists form."""
    if isinstance(rows, str):
        raise ValueError(
            f"adapter_matrix = {rows!r} is not resolved: turn 'corruption'/'corruption_inverse' into its "
            f"{ARM_DIMS}x{ARM_DIMS} matrix with bridge.corruption_lift/lift_inverse before make_steering"
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (ARM_DIMS, ARM_DIMS):
        raise ValueError(f"adapter_matrix must be {ARM_DIMS}x{ARM_DIMS} rows, got shape {matrix.shape}")
    return matrix.tolist()


def steering_batch(bundle: Bundle, raw_batch: dict) -> dict:
    """The station's raw observation through the bundle's pre-processor: batched, normalized, tokenized, on device.

    The operator's command is *not* added here: a control loop reads it once per chunk and puts it in
    ``batch["command"]`` afterwards (the LIBERO session's order), because only the shared modes have one.
    """
    return bundle.preprocessor(raw_batch)


def _command_batch(batch: dict, command7: NDArray, device: Any) -> dict:
    """``batch`` plus the operator's command as the steering policy takes it: ``(1, 7)`` float32, on ``device``."""
    command = np.asarray(command7, dtype=np.float64).reshape(-1)
    if command.shape != (ARM_DIMS,):
        raise ValueError(f"command7 must be ({ARM_DIMS},) action units, got shape {command.shape}")
    import torch  # the real bundle's path only; fake_bundle never takes it

    return {**batch, "command": torch.as_tensor(command, dtype=torch.float32)[None].to(device)}


def _device(bundle: Bundle) -> Any:
    """The bundle's device: the loaded policy config's, else CPU."""
    return getattr(getattr(bundle.policy, "config", None), "device", None) or "cpu"


# ---------------------------------------------------------------- teleop and the fake bundle


def teleop_action(served7: NDArray, gripper_closed: bool) -> NDArray:
    """The operator's 8-wide action: the served 7 joint commands and the binarized gripper button.

    Teleop never touches a policy: the station's response is ``[served7, 1.0 if closed else 0.0]`` in
    action units (``station.action_dict`` clips the first seven and binarizes the gripper at 0.5).
    """
    served = np.asarray(served7, dtype=np.float64).reshape(-1)
    if served.shape != (ARM_DIMS,):
        raise ValueError(f"served7 must be ({ARM_DIMS},) action units, got shape {served.shape}")
    action = np.empty(STATION_DIMS, dtype=np.float64)
    action[:ARM_DIMS] = served
    action[ARM_DIMS] = 1.0 if gripper_closed else 0.0
    return action


class _FixedActionPolicy:
    """The fake bundle's stand-in: ``select_action`` is the fixed 8-wide action, in any batch."""

    def __init__(self, action: NDArray):
        self._action = action

    def select_action(self, batch: dict) -> NDArray:
        return self._action[None, :]

    def reset(self) -> None:
        pass


def _identity(value):
    return value


def fake_bundle(action8: tuple[float, ...] = (0.1,) * ARM_DIMS + (1.0,)) -> Bundle:
    """A numpy-only bundle for ``--dry-run``: ``action`` returns ``action8`` and nothing is loaded.

    No checkpoint, no tokenizer, no torch: the "policy" is the fixed action, the pipelines are the
    identity, and the action is returned whatever the command (a dry run walks every mode's
    composition without steering anything).
    """
    action = np.asarray(action8, dtype=np.float64).reshape(-1)
    if action.shape != (STATION_DIMS,):
        raise ValueError(f"a fake action must be ({STATION_DIMS},), got shape {action.shape}")
    bundle = Bundle(policy=_FixedActionPolicy(action), preprocessor=_identity, postprocessor=_identity)
    bundle._cache["fixed_action"] = action
    return bundle


# ---------------------------------------------------------------- the station's action


def _action_units(action: Any) -> NDArray:
    """The post-processed action's first 8 dims as float64, in raw action units.

    The slice happens after the post-processor because the unnormalizer's QUANTILES map is what turns
    the model's normalized 32-wide action into action units; before it the numbers are in [-1, 1] model
    space. The gripper is dim 7 of the result, by the station's (and the checkpoint's) layout.
    """
    if hasattr(action, "detach"):  # a torch tensor, without importing torch: bf16 needs .float() first
        action = action.detach().to("cpu").float().numpy()
    values = np.asarray(action, dtype=np.float64)
    if values.ndim == 2:
        values = values[0]
    if values.ndim != 1 or values.shape[0] < STATION_DIMS:
        raise ValueError(
            f"a post-processed action must be (batch, >= {STATION_DIMS}) or ({STATION_DIMS},), "
            f"got shape {values.shape}"
        )
    return values[:STATION_DIMS].copy()
