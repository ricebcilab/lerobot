"""The pi0.5 bundle: the checkpoint, its action statistics, and the steering policy's one control path.

The station's action is 8 numbers in the checkpoint's action units; pi0.5-DROID was trained on 32-wide
state and action vectors whose first 8 dims are the arm's joints and the gripper, everything beyond
padding. The published checkpoint (``lerobot/pi05_droid``) does not carry the normalizer's statistics:
its cached pre/post-processor JSONs declare the features empty and the ``normalization_mapping`` says
``QUANTILES`` for STATE and ACTION, so the q01/q99 statistics have to come from openpi's
``assets/droid/norm_stats.json`` and be installed here before the pipelines mean anything
(``DESIGN.md``, "What the checkpoint's own code settles").

This module is the policy half of that: ``load_bundle`` builds the checkpoint and its two pipelines on
the configured device (weights and the gated PaliGemma tokenizer are needed for a real one: the probe on
the station is that [V]), and, when the caller names a MARS task checkpoint, that checkpoint's scene
step (``scene.Scene``) so ``prepare`` turns the scene camera's pictures into the tokens the task module
reads; ``make_steering`` wraps the loaded policy in the mode's ``SteeringPolicy``;
``Bundle.prepare``/``Bundle.act`` are the two halves of a control tick — the batch once per chunk, the
action every tick, because the policies only read a batch when their action queue is empty — and
``Bundle.action`` is the convenient single call for callers without that boundary; and
``teleop_action`` is the no-policy path's 8-wide action. The statistics half lives in ``stats.py``
(``install_stats``, with the ``normalize_processor.py`` lines it follows in that module's docstring).
``fake_bundle`` is the numpy-only stand-in ``--dry-run`` uses, so importing this module and driving it
must not need torch, the checkpoint or the tokenizer: every lerobot import is inside the function that
needs it, and ``stats`` is numpy-only at import.

Where the 8 dims are sliced: **after** the post-processor, never before. The model works in normalized
space over the 32-wide padded action, and the unnormalizer's QUANTILES map is a per-dim affine map
(``q01``/``q99``); the station's action units are that map's output, so the first 8 dims are taken from
its result. Slicing the normalized action before it would hand the station numbers in [-1, 1] with no
physical meaning, and the statistics ``install_stats`` returns for the steering cores are the affine
form of the same quantile map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from station import SCENE_IMAGES
from stats import STATION_DIMS, install_stats

if TYPE_CHECKING:
    from config import Config, SteeringConfig
    from scene import Scene

    from lerobot.policies.steering import SteeringPolicy

ARM_DIMS = 7
"""The Franka's joints: the steering command's width, and dims 0-6 of the action."""

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
    other place to read it. ``scene`` is a task checkpoint's scene (``scene.Scene``: the token source
    and render size its module was built with), or ``None`` when no task checkpoint was named: its one
    step is what ``prepare`` adds to the batch, while the module itself stays the steering policy's to
    load from the mode's ``adapter_task`` path. ``_cache`` holds the one ``SteeringPolicy``
    ``make_steering`` makes (the loop calls ``act`` about 15 times a second, so it is made once and
    reset per episode), the scene step's pipeline (built once: its encoder and token cache are state)
    and the fake bundle's fixed action.

    A control tick has two halves, because the policies only read a batch when their action queue is
    empty (once per chunk of ``n_action_steps``): ``prepare`` runs the pre-processor (the expensive part:
    two images, the tokenizer, the device transfer) once per chunk, ``act`` turns a prepared batch into
    one action every tick, and ``action`` is the convenient single call for callers without a chunk
    boundary.
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    mean: NDArray | None = None
    std: NDArray | None = None
    deadband: float = DEFAULT_DEADBAND
    scene: Scene | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False, hash=False)

    def prepare(self, raw_batch: dict) -> dict:
        """The station's raw observation through the pre-processor: batched, normalized, tokenized, on device.

        The expensive half of a tick, so the loop builds the batch once per chunk of ``n_action_steps``
        and calls ``act`` on the ticks in between (the policies only read a batch when their action queue
        is empty). The operator's command is not added here.

        With a ``scene`` (a task checkpoint's), the scene's one-step pipeline runs first: it encodes
        ``observation.scene_images`` into ``observation.scene_tokens``, the tokens the steering policy
        hands the task module, and the base pre-processor then carries them to the device with the rest
        of the batch. A raw batch is one row, so its scene picture is given the leading batch dimension
        the step reads ([B, 3, S, S]; a trainer's batch of B rows is already there). Without a scene the
        batch is exactly the pre-processor's.
        """
        if self.scene is not None:
            raw_batch = self._scene_pipeline()(_scene_row(raw_batch))
        return steering_batch(self, raw_batch)

    def act(self, batch: dict, command7: NDArray | None = None) -> NDArray:
        """One station action, in raw action units, from a batch ``prepare`` made: no preprocessing.

        ``command7`` None: the base policy's action (``autonomous``); the operator's 7 joint-velocity
        command in action units otherwise (the steered modes, through the mode's ``SteeringPolicy``).
        The action is post-processed and its first 8 dims returned as float64.

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
        if command7 is None:
            action = self.policy.select_action(batch)
        else:
            action = self._steering().select_action(_command_batch(batch, command7, _device(self)))
        return _action_units(self.postprocessor(action))

    def action(self, raw_batch: dict, command7: NDArray | None = None) -> NDArray:
        """``act(prepare(raw_batch), command7)``: the convenient call, preprocessing every time.

        The control loop uses ``prepare``/``act`` instead, so a mid-chunk tick does not re-run the
        pre-processor; this exists for a caller that has no chunk boundary to hoist it across.
        """
        return self.act(self.prepare(raw_batch), command7)

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
                "Bundle.act with a command needs the mode's SteeringPolicy: call "
                "make_steering(bundle, steering_cfg) once before the control loop; it caches it here"
            )
        return steering

    def _scene_pipeline(self) -> Any:
        """The scene's one-step pipeline, built once and kept: its encoder and its token cache are state."""
        pipeline = self._cache.get("scene_pipeline")
        if pipeline is None:
            pipeline = self.scene.step(device=str(_device(self)))
            self._cache["scene_pipeline"] = pipeline
        return pipeline


# ---------------------------------------------------------------- loading


def load_bundle(cfg: Config, device: str | None = None, task: Path | None = None) -> Bundle:
    """pi0.5-DROID on ``device`` (default: ``cfg.policy.device``) with its pipelines and statistics.

    The config's ``pretrained_path``, ``device``, ``dtype`` and ``n_action_steps`` are put on the
    checkpoint's own config; the policy is built by its class (``make_policy`` insists on a dataset or
    env for features, and the checkpoint's config already declares them) and evaluated; the pre/post
    processors are loaded from the checkpoint's JSONs exactly as the LIBERO session loads them,
    overriding the device and the (empty) rename map; then ``install_stats`` is applied when the config
    names a statistics file that is on disk. Without one, ``Bundle.mean``/``std`` are ``None`` (the
    openpi asset is gitignored): the shared modes then refuse to build in ``make_steering`` and
    ``Bundle.action`` refuses to act, and teleop never comes here at all.

    ``task`` is a MARS task module's checkpoint (``AdapterModules.save_module``'s format). Its recorded
    scene (``scene.from_checkpoint``: the token source and render size no module or weights are
    needed to read) goes on the bundle, so ``prepare`` encodes the scene camera's pictures before the
    base pre-processor; the module itself is still loaded by the ``SteeringPolicy`` from the mode's
    ``adapter_task`` path, which is the composition root's choice: the scene step is the only thing the
    bundle adds. Without ``task`` there is no scene and ``prepare`` is exactly the policy's. A task
    checkpoint whose token source is out of this phase's scope raises here, before the checkpoint or the
    VLM it would need is loaded.

    This needs the checkpoint's weights and the gated PaliGemma tokenizer; it cannot run in the
    offline test environment (it runs on the station, ``probe`` first).
    """
    scene = None
    if task is not None:
        from scene import from_checkpoint

        scene = from_checkpoint(Path(task))

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
        scene=scene,
    )


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
    not need them. A named ``adapter_user``/``adapter_task`` checkpoint that is not on disk is refused
    here too, before anything is loaded: ``train-user`` writes the user module, and the message says so.
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
    for kind in ("user", "task"):
        path = getattr(steering, f"adapter_{kind}", None)
        if path is not None and not Path(path).is_file():
            raise ValueError(
                f"adapter_{kind}: {path} is not a file. Train the {kind} module first (Phase B's "
                f"`mars-franka train-user` for the user module) or point the mode table at a checkpoint"
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


def _scene_row(raw_batch: dict) -> dict:
    """The raw batch with its scene picture given the leading batch dimension the scene step reads.

    A raw observation is one row: its ``observation.scene_images`` is the station's channel-first
    picture, ``(3, S, S)`` (``scene.scene_images``'s), while ``SteeringSceneTokensProcessorStep`` reads a
    batch, ``[B, 3, S, S]``. A batch that already has the leading dimension (a trainer's, B rows) is left
    as it is; a batch without the key is left to the step, which refuses it by name.
    """
    pictures = raw_batch.get(SCENE_IMAGES)
    if pictures is None:
        return raw_batch
    array = np.asarray(pictures)
    return raw_batch if array.ndim != 3 else {**raw_batch, SCENE_IMAGES: array[None]}


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
    composition without steering anything). ``std`` is a stand-in (0.3 in every action dim) so the
    loop's readout noise, which is defined in action-standard-deviation units, has a scale to use
    without the checkpoint's statistics; ``mean`` stays None because no statistics were installed.
    """
    action = np.asarray(action8, dtype=np.float64).reshape(-1)
    if action.shape != (STATION_DIMS,):
        raise ValueError(f"a fake action must be ({STATION_DIMS},), got shape {action.shape}")
    bundle = Bundle(
        policy=_FixedActionPolicy(action),
        preprocessor=_identity,
        postprocessor=_identity,
        std=np.full(STATION_DIMS, 0.3),
    )
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
