"""The user module's offline validation: FRS on held-out states, against the cue's intent.

DESIGN.md's Phase B "Validation (offline, no arm, no env)", in the order ``validate`` runs it:

1. the states file is split with the trainer's own rule (``train_user.split_states``, the same seed
   ``[training] seed`` and ``[training] val_frac``), and only its held-out half is evaluated -- the
   module never saw those states;
2. for each held-out state, one cue is drawn from the held-out seed ``[training].seed, 2``
   (``cues.cue_twists``/``cue_commands``, speeds from the file's deflections through
   ``cues.magnitudes``): ``intent`` is the faithful command, and the session's readout serves it as
   ``served`` (``train_user.served_cues``: the config's corruption at the file's ``q0`` and episode, the
   auto gain, one noise draw at the middle of ``[user] sigma`` -- the same sigma for every draw), with
   ``clean`` the same intent and the same noise draw through a faithful readout. Both commands reach
   the steering core as references through the core's own ``reference_chunk`` rule (the study's
   ``[modes.frs]``, ``reference_step = 1.0``), never a re-implementation;
3. for each FRS depth in ``[training] val_depths``, one steered chunk per arm, all from the same seeded
   flow noise: ``plan`` the policy's own chunk (no command), ``none`` FRS on ``served``, ``clean`` FRS
   on ``clean``, ``ideal`` FRS on ``intent`` (no readout noise), ``f`` FRS on ``served`` with the fixed
   adapter matrix (the readout's lift ``F``, the ``mars-f`` convention) and ``user`` FRS on ``served``
   with the trained module (``AdaptedField(None, user)``);
4. the metric: each arm's executed chunk (``chunk[:, :n_action_steps, :7]``, the core's normalized
   space) is unnormalized with the bundle's 8-wide action statistics into action units, multiplied by
   ``bridge.ACTION_UNIT`` (one action unit = 3 rad/s) into joint velocities in rad/s, mapped through
   ``bridge.geometric_jacobian(q)`` at the state's joints to a Cartesian twist (m/s, rad/s; linear rows
   first) and averaged over the executed steps; the reported number is the cosine between that twist's
   linear part and the cue's intent twist's linear part. The mean over the held-out states is reported
   per (depth, arm), with its count.

The module repairs the command only if ``user`` moves from ``none`` toward ``clean``. The go/no-go is
the reader's: ``validate`` returns the numbers and prints the table, and never turns them into an exit
code.

Two details follow from the loop's own semantics. A cue inside ``[command] deadband`` is what the loop
would count as idle, so that arm executes the policy's own plan (the cue speeds come from the pilot's
measured pushes, so this is not expected). And the ``clean`` command is the *faithful readout*'s serve
-- the config with ``[readout] corruption = "none"``, hence the ``auto`` gain 1.0 -- rather than the
corrupted readout's gain applied to the intent: the gain exists to keep the corrupted command inside
the clip.

One cue per state, not ``[user] cues`` of them: the calibration draws ``[user] cues`` per state to fit
the module, and the validation only needs one fresh draw per held-out state to compare the arms. The
torch flow noise is seeded per batch from ``[training].seed, 3``, so every arm of every depth gets the
same Gaussian sample and the difference between arms is the command and the field alone. ``[user]
checkpoint`` is the module to evaluate; a checkpoint trained under another prompt is refused, because
the module corrects one prompt-conditioned field and validating another measures something else. The
states' recorded readout must be this config's (``train_user._check_readout``), for the same reason the
trainer refuses it: the replay would serve another condition's commands.

Everything torch-free (the states, the cues, the metric's arithmetic) stays in the sibling modules; this
module loads the checkpoint, so it needs torch and a GPU when ``[training] device`` is ``cuda``. Its
imports are lazy in ``cli.py``, so no other command's path pulls torch. The real velocity field, the
real statistics and a real session are verification on the station (chopper) only; the offline tests
use a fake bundle and a fake module, as ``test_train_user.py`` does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import bridge
import numpy as np
import statefiles
import torch
import train_user
from bridge import ACTION_UNIT, ARM_DIMS
from config import Config, load_config
from cues import magnitudes
from numpy.typing import NDArray
from policy import load_bundle
from stats import STATION_DIMS

from lerobot.policies.steering import AdaptedField, AdapterModules

ARMS = ("plan", "none", "clean", "ideal", "f", "user")
"""The evaluated arms, in the order the table prints them.

``plan`` is the policy's own chunk, ``none`` FRS on the corrupted command, ``clean`` FRS on a faithful
readout's command with the same noise, ``ideal`` FRS on the intent itself, ``f`` FRS with the fixed
adapter matrix (the ``mars-f`` convention) and ``user`` FRS with the trained module.
"""

UNITS = (
    "executed action units * bridge.ACTION_UNIT (3 rad/s per unit) are joint velocities in rad/s, mapped "
    "through bridge.geometric_jacobian(q) to a (m/s, rad/s) twist and averaged over the executed steps; "
    "the cosine is between that twist's linear part (m/s) and the cue twist's linear part (m/s)"
)


# ---------------------------------------------------------------- the metric's pure parts


def executed_twist(chunk: NDArray, q: NDArray, mean: NDArray, std: NDArray, n_action_steps: int) -> NDArray:
    """(6,) the mean Cartesian twist of an executed chunk, at the state's joints ``q``.

    ``chunk`` is one normalized chunk (H, >= 8) as the steering core returns it: dims 0-6 unnormalized
    with the bundle's 8-wide action statistics ``(x - mean) / std``, then multiplied by
    ``bridge.ACTION_UNIT`` into joint velocities in rad/s and mapped through
    ``bridge.geometric_jacobian``; the mean over the first ``n_action_steps``. The result is in the
    bridge's units: three linear (m/s) then three angular (rad/s).
    """
    values = np.asarray(chunk, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < STATION_DIMS:
        raise ValueError(f"chunk: expected a normalized (H, >= {STATION_DIMS}) chunk, got {values.shape}")
    executed = values[: int(n_action_steps), :STATION_DIMS]
    action_units = (
        executed[:, :ARM_DIMS] * np.asarray(std, dtype=np.float64)[:ARM_DIMS]
        + np.asarray(mean, dtype=np.float64)[:ARM_DIMS]
    )
    joint_velocity = action_units * ACTION_UNIT
    return bridge.geometric_jacobian(np.asarray(q, dtype=np.float64)) @ joint_velocity.mean(axis=0)


def translation_cosine(twist: NDArray, intent: NDArray) -> float:
    """Cosine between two twists' linear parts (both (6,): m/s, then rad/s).

    ``0.0`` when either linear part is numerically zero: a chunk that does not move has no heading to
    agree with the cue's, and a mean over cosine-of-nothing would be undefined.
    """
    executed = np.asarray(twist, dtype=np.float64)[:3]
    want = np.asarray(intent, dtype=np.float64)[:3]
    norm = float(np.linalg.norm(executed) * np.linalg.norm(want))
    if norm < 1e-12:
        return 0.0
    return float(np.dot(executed, want) / norm)


# ---------------------------------------------------------------- the arms


def _arm_chunks(
    core, batch: dict, cued: Sequence[train_user.Cues], *, user, action_std: NDArray, depths, seed: int
) -> dict[int, dict[str, torch.Tensor]]:
    """The chunk each arm executes for one batch: ``{depth: {arm: chunk}}``, every arm from the same noise.

    ``plan`` has no command, so it does not depend on the reversal depth: it is drawn once and shared by
    every depth's column. The five steered arms call the core's ``steered_chunk`` with their command and
    their adapter field -- ``f`` the readout's lift normalized by the action std, as the loop's
    ``mars-f`` holder does, and ``user`` the trained module; ``_seeded`` restores the same Gaussian sample
    for each of them.
    """
    served = np.stack([cue.served[0] for cue in cued])
    commands = {
        "none": served,
        "clean": np.stack([cue.clean[0] for cue in cued]),
        "ideal": np.stack([cue.intent[0] for cue in cued]),
        "f": served,
        "user": served,
    }
    plan = _seeded(core, seed, batch, None, None)
    fixed = _fixed_field(cued, action_std, like=plan)
    user_field = AdaptedField(None, user)
    chunks = {}
    for depth in depths:
        core.n_reversal_steps = depth
        arms = {"plan": plan}
        for arm in ("none", "clean", "ideal", "f", "user"):
            field = user_field if arm == "user" else fixed if arm == "f" else None
            arms[arm] = _seeded(core, seed, batch, commands[arm], field)
        chunks[depth] = arms
    return chunks


def _seeded(core, seed: int, batch: dict, commands, field) -> torch.Tensor:
    """One ``steered_chunk`` from the seed's Gaussian sample: every arm sees the same flow noise."""
    torch.manual_seed(seed)
    with torch.no_grad():
        chunk, _ = core.steered_chunk(batch, commands, field=field)
    return chunk


def _fixed_field(cued: Sequence[train_user.Cues], action_std: NDArray, *, like: torch.Tensor):
    """``mars-f``'s ``AdaptedField``: the readout's lift F, normalized by the action std (None: corruption off).

    The loop's ``mars-f`` puts F (the operator's units) on the steering policy's holder and the core
    normalizes it by the postprocessor's action std; that is ``AdapterModules.field(std=..., like=...)``,
    with one matrix per state when the readout's ``{random: ...}`` family drew one per episode.
    """
    if cued[0].matrix is None:
        return None
    std7 = np.asarray(action_std, dtype=np.float64)[:ARM_DIMS]
    matrices = np.stack([cue.matrix for cue in cued])
    return AdapterModules(matrix=matrices, n_dims=None).field(std=std7, like=like)


# ---------------------------------------------------------------- the run's fixed pieces


def _depths(config: Config, num_inference_steps: int) -> tuple[int, ...]:
    """``[training] val_depths``, each a reversal depth of this checkpoint's own schedule."""
    for depth in config.training.val_depths:
        if not 1 <= depth <= num_inference_steps:
            raise ValueError(
                f"[training] val_depths: {depth} is not a reversal depth of this checkpoint's "
                f"{num_inference_steps} Euler steps (choose from 1 to {num_inference_steps})"
            )
    return tuple(config.training.val_depths)


def _device(config: Config) -> str:
    """``[training] device``, named before the checkpoint is loaded when CUDA is not there."""
    device = str(config.training.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            f"[training] device = {device!r} but CUDA is not available on this machine: the validator "
            'needs the policy\'s velocity field on the GPU (set [training] device = "cpu" to run it slowly)'
        )
    return device


def _user_checkpoint(checkpoint: str | Path) -> Path:
    """The ``[user] checkpoint`` file, named with how it is made when it is not there."""
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError(
            f"[user] checkpoint: {path} is not a file. Train the user module first: "
            "`mars-franka train-user --config <config> --prompt <text>` writes <out>/user.pt"
        )
    return path


def _user_module(checkpoint: Path, device: str) -> tuple[Any, dict]:
    """The trained module from ``[user] checkpoint``: its ``LearnedUserModule`` and the checkpoint's meta."""
    module, kind, meta = AdapterModules.load_module(checkpoint, device)
    if kind != "user":
        raise ValueError(f"[user] checkpoint: {checkpoint} holds a {kind} module, not a user module")
    return module, meta


def _check_prompt(checkpoint: Path, meta: dict, prompt: str) -> None:
    """Refuse a module trained on another prompt: it corrects that field, not this one."""
    recorded = meta.get("prompt")
    if recorded is not None and str(recorded) != str(prompt):
        raise ValueError(
            f"[user] checkpoint: {checkpoint} was trained on prompt {recorded!r}, not {prompt!r}: the "
            "module corrects one prompt-conditioned field, so validate-user takes the trainer's --prompt"
        )


def _batches(indices: Sequence[int], size: int) -> Iterator[list[int]]:
    """The held-out row indices in batches of ``[training] batch``: one velocity-field encoding per batch."""
    for start in range(0, len(indices), size):
        yield list(indices[start : start + size])


# ---------------------------------------------------------------- the validation


def validate(
    config: Config,
    *,
    prompt: str,
    states_path: Path,
    checkpoint: Path,
    out: Path | None = None,
) -> dict:
    """Validate the user module at ``checkpoint`` on the held-out states of ``states_path``.

    ``prompt`` is the study's velocity-field prompt, the one the module was trained under; ``out`` is the
    directory ``validation.json`` lands in (``None``: ``[user] out``). Returns the JSON object written,
    every depth and arm's mean cosine and count included. ``device`` is ``[training] device``: the
    checkpoint and the fake bundle of the offline tests are loaded there too.
    """
    path = train_user._states_file(states_path)
    arrays, session = statefiles.load(path)
    rows = train_user._rows(arrays)
    checked = train_user._check_readout(config, path, session, arrays["q0"])
    if not checked:
        print(
            f"warning: {path} records no readout provenance; its readout is the config's [readout], unchecked",
            flush=True,
        )
    checkpoint = _user_checkpoint(checkpoint)
    device = _device(config)
    user, meta = _user_module(checkpoint, device)
    _check_prompt(checkpoint, meta, prompt)
    split = np.random.default_rng([config.training.seed, 0])
    _, val_indices = train_user.split_states(rows, config.training.val_frac, split)
    bundle = load_bundle(config, device=device)
    facts = train_user._checkpoint(bundle)
    core = train_user._steering(bundle, config).core
    depths = _depths(config, facts["num_inference_steps"])
    mean, std = bundle.mean, bundle.std
    if mean is None or std is None:  # make_steering refuses this before the core exists; named for a reader
        raise ValueError("the bundle has no action statistics: [policy] stats is what unnormalizes the chunk")

    sigma = (float(config.user.sigma[0]) + float(config.user.sigma[1])) / 2
    cue_rng = np.random.default_rng([config.training.seed, 2])
    pool = magnitudes(arrays["deflection"], config.command.translation_gain, rng=cue_rng)
    noise_rng = np.random.default_rng([config.training.seed, 3])
    cosines = {depth: {arm: [] for arm in ARMS} for depth in depths}
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        for group in _batches(val_indices, config.training.batch):
            group_rows = [rows[index] for index in group]
            cued = [
                train_user.served_cues(
                    config,
                    q=row["joint_position"],
                    episode=row["episode"],
                    pool=pool,
                    q0=arrays["q0"],
                    action_std7=std,
                    rng=cue_rng,
                    sigma=sigma,
                    count=1,
                )
                for row in group_rows
            ]
            batch = train_user._stack(
                [bundle.prepare(train_user._raw_batch(row, prompt)) for row in group_rows]
            )
            chunks = _arm_chunks(
                core,
                batch,
                cued,
                user=user,
                action_std=std,
                depths=depths,
                seed=int(noise_rng.integers(2**31)),
            )
            for depth in depths:
                for arm in ARMS:
                    executed = chunks[depth][arm].detach().float().cpu().numpy()
                    for position, (row, cue) in enumerate(zip(group_rows, cued, strict=True)):
                        twist = executed_twist(
                            executed[position],
                            row["joint_position"],
                            mean,
                            std,
                            facts["n_action_steps"],
                        )
                        cosines[depth][arm].append(translation_cosine(twist, cue.twists[0]))
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)

    out = Path(config.user.out if out is None else out)
    json_path = out / "validation.json"
    result = _result(
        config,
        prompt=prompt,
        path=path,
        rows=rows,
        val_indices=val_indices,
        checkpoint=checkpoint,
        user=user,
        meta=meta,
        checked=checked,
        sigma=sigma,
        depths=depths,
        cosines=cosines,
        n_action_steps=facts["n_action_steps"],
        json_path=json_path,
    )
    out.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    _print_table(result)
    return result


def _result(
    config: Config,
    *,
    prompt: str,
    path: Path,
    rows: Sequence[dict],
    val_indices: Sequence[int],
    checkpoint: Path,
    user: Any,
    meta: dict,
    checked: bool,
    sigma: float,
    depths: Sequence[int],
    cosines: dict[int, dict[str, list[float]]],
    n_action_steps: int,
    json_path: Path,
) -> dict:
    """The JSON object ``validate`` writes: the provenance, the per-(depth, arm) means and the counts."""
    results = {}
    for depth in depths:
        results[str(depth)] = {
            arm: {
                "cosine": float(np.mean(cosines[depth][arm])) if cosines[depth][arm] else None,
                "count": len(cosines[depth][arm]),
            }
            for arm in ARMS
        }
    module = getattr(user, "config", {})
    return {
        "kind": "mars-franka-user-validation",
        "prompt": str(prompt),
        "states": str(path),
        "states_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "states_rows": len(rows),
        "held_out": {
            "indices": [int(index) for index in val_indices],
            "rows": len(val_indices),
            "episodes": sorted({int(rows[index]["episode"]) for index in val_indices}),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_prompt": meta.get("prompt"),
        "module": {
            "horizon": getattr(user, "horizon", None),
            "n_dims": getattr(user, "n_dims", None),
            "d": module.get("d"),
            "tdim": module.get("tdim"),
            "time_embedding": module.get("time_embedding"),
            "step_rank": module.get("step_rank"),
            "cross_step_rank": module.get("cross_step_rank"),
            "additive": module.get("additive"),
        },
        "readout": {
            "corruption": config.readout.corruption,
            "noise": config.readout.noise,
            "seed": config.readout.seed,
            "checked": checked,
        },
        "sigma": sigma,
        "depths": [int(depth) for depth in depths],
        "arms": list(ARMS),
        "n_action_steps": int(n_action_steps),
        "count": len(val_indices),
        "units": UNITS,
        "results": results,
        "out": str(json_path),
    }


# ---------------------------------------------------------------- the table


def _print_table(result: dict) -> None:
    """The small table: one row per arm, one column per depth, and the gate's readings under it."""
    depths = result["depths"]
    results = result["results"]
    module = result["module"]
    print(
        f"validate-user: {result['held_out']['rows']} held-out of {result['states_rows']} states from "
        f"{result['states']}, one cue each; prompt {result['prompt']!r}",
        flush=True,
    )
    print(
        f"checkpoint: {result['checkpoint']} (user module horizon {module['horizon']}, "
        f"n_dims {module['n_dims']}); sigma {result['sigma']:g} (the middle of [user] sigma)",
        flush=True,
    )
    print("arm   " + "".join(f"{'d' + str(depth):>10}" for depth in depths), flush=True)
    for arm in result["arms"]:
        print(
            f"{arm:<6}" + "".join(f"{results[str(depth)][arm]['cosine']:>+10.3f}" for depth in depths),
            flush=True,
        )
    print(
        f"mean cosine between the executed translation and the cue's intent twist (n = {result['count']})",
        flush=True,
    )
    readings = []
    for depth in depths:
        row = results[str(depth)]
        if any(row[arm]["cosine"] is None for arm in ("none", "clean", "user")):
            readings.append(f"d{depth}: not evaluated")
            continue
        user_gap = abs(row["user"]["cosine"] - row["clean"]["cosine"])
        none_gap = abs(row["none"]["cosine"] - row["clean"]["cosine"])
        readings.append(
            f"d{depth}: |user - clean| {user_gap:.3f} {'<' if user_gap < none_gap else '>='} "
            f"|none - clean| {none_gap:.3f}"
        )
    print(
        "gate (the reader's, never an exit status): the module is worth the study only where it moves "
        "the heading from none toward clean, i.e. |user - clean| < |none - clean|;",
        flush=True,
    )
    for reading in readings:
        print(f"  {reading}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """The script entry point: ``--config`` and ``--prompt``, with ``[user]`` naming the input and output.

    ``cli.py``'s ``validate-user`` command is the shipped surface; this is the form for a run without it
    (``python validate.py --config ... --prompt ...``). ``--states``, ``--checkpoint`` and ``--out``
    override ``[user]``.
    """
    parser = argparse.ArgumentParser(
        prog="validate-user",
        description="Validate a trained MARS user module offline on a Franka session's held-out states.",
    )
    parser.add_argument("--config", type=Path, required=True, metavar="PATH", help="the config file")
    parser.add_argument("--prompt", required=True, metavar="TEXT", help="the velocity field's prompt")
    parser.add_argument("--states", type=Path, metavar="PATH", help="override [user] states")
    parser.add_argument("--checkpoint", type=Path, metavar="PATH", help="override [user] checkpoint")
    parser.add_argument("--out", type=Path, metavar="PATH", help="override [user] out")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = validate(
        config,
        prompt=args.prompt,
        states_path=config.user.states if args.states is None else args.states,
        checkpoint=config.user.checkpoint if args.checkpoint is None else args.checkpoint,
        out=config.user.out if args.out is None else args.out,
    )
    print(f"validation: {result['out']}", flush=True)
    return 0
