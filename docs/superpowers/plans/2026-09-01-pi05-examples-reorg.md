# pi0.5 Examples Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the LIBERO shared-autonomy branch into `main` and reorganize both pi0.5 pipelines under `examples/pi05/` with responsibility-cut modules, intent-stating configs, the steering algorithms in `src/`, and CPU unit tests.

**Architecture:** The steering wrappers and reverse-flow integrator move to `src/lerobot/policies/pi05/steering.py`. The example splits into `teleop.py` (operator input), `live_view.py` (browser window), `session.py` (policy + scene + teleop + view + mode, with `rollout()`), `config.py` (YAML to settings, matrix specs), and two thin entry points that drive a `Session`. Experiment conditions are `extends: base.yaml` overlays with inline matrix specs.

**Tech Stack:** Python 3.12, PyTorch, numpy, PyYAML, pandas, scipy, LIBERO (`hf-libero`), uv, pytest, ruff, pre-commit.

**Spec:** `docs/superpowers/specs/2026-09-01-pi05-examples-reorg-design.md`

## Global Constraints

- Python 3.12, ruff line length 110, ruff rules `E W F I B C4 T20 N UP SIM` (print is allowed).
- No new runtime dependencies. pandas is a base dependency; scipy and matplotlib come with the `pi`/`libero` extras.
- Every command runs through `uv run --no-sync` from the repo root after Task 0's `uv sync`.
- Pipeline behavior does not change: same modes, same REPL commands, same run-directory files and field names (`trials.jsonl` fields, `.npz` arrays, `config.yaml` keys `corruption_matrix` and `flow_adapter_matrix`).
- Use `git mv` for every move so blame follows.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Source line numbers below refer to the files as they exist right after Task 0's merge (identical to `origin/pi05-libero-shared-autonomy-experiments`).

## File map

| Path | Responsibility |
| --- | --- |
| `src/lerobot/policies/pi05/steering.py` | Steering algorithms, matrix builders, `ReversalAdapter`, `TeleopSource` protocol, action-dim constants |
| `examples/pi05/README.md` | Index of the two pipelines |
| `examples/pi05/feeding_finetune/*` | Moved feeding pipeline, renamed scripts |
| `examples/pi05/libero_shared_autonomy/env.sh` | Shared environment block for launchers |
| `.../teleop.py` | SpaceMouse, keyboard, merge, corruption, noise, recording, `TeleopChain` |
| `.../live_view.py` | HTML page, `FrameStream`, `LiveView` HTTP server |
| `.../session.py` | `Session`, `RolloutResult`, mode wiring, rollout loop |
| `.../config.py` | Settings dataclasses, YAML read/merge/flatten/validate, matrix spec resolution |
| `.../interactive.py` | REPL over a `Session` |
| `.../experiment.py` | Scheduled trials over a `Session`, `TrialRecorder` |
| `.../notebooks/analyze.py`, `analyze_experiments.ipynb` | Analysis library and stripped notebook |
| `.../configs/interactive.yaml`, `configs/experiment/*.yaml` | Settings files |
| `tests/policies/pi0_pi05/test_pi05_steering.py` | CPU tests for steering |
| `tests/policies/pi0_pi05/test_pi05.py` | Sampler hook tests (GPU + HF token) |
| `tests/examples/pi05_libero_shared_autonomy/*` | CPU tests for config, teleop, experiment helpers, analyze |

---

### Task 0: Merge the branch and set up the environment

**Files:**
- Modify: git history only

**Interfaces:**
- Produces: a `reorg-pi05-examples` branch containing the branch's files at `examples/pi05_libero/`, and a `.venv` where `uv run --no-sync pytest` works.

- [ ] **Step 1: Merge the branch into main**

`main` already carries the spec commit on top of the branch point, so this is a regular merge, not a fast-forward. The branch touches nothing under `docs/`, so there are no conflicts.

```bash
cd /home/user/Projects/lerobot
git status --short            # must be empty
git merge --no-ff origin/pi05-libero-shared-autonomy-experiments \
  -m "Merge pi05-libero-shared-autonomy-experiments into main

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git log --oneline -5
```

Expected: a merge commit; `ls examples/pi05_libero` lists `experiment.py`, `teleop_input.py`, five `config_experiment_*.yaml` files.

- [ ] **Step 2: Create the work branch**

The working tree is clean and this checkout has no venv yet, so work on a branch in place. (A worktree would need its own `uv sync`; skip it.)

```bash
git switch -c reorg-pi05-examples
```

- [ ] **Step 3: Create the environment**

```bash
uv sync --locked --extra test --extra pi --extra libero
uv run --no-sync python -c "import lerobot, torch, scipy, pandas, yaml; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.11.0+cu128 True`. If CUDA is `False`, stop and report; the driver is 570 (CUDA 12.8) and the lock pins cu128, so this should not happen.

- [ ] **Step 4: Confirm the baseline test run**

```bash
uv run --no-sync pytest tests/policies/pi0_pi05/test_pi05.py -q 2>&1 | tail -3
uv run --no-sync pre-commit run --all-files 2>&1 | tail -5
```

Expected: tests pass or skip; pre-commit passes (the merged branch was committed through these hooks).

---

### Task 1: Move the feeding pipeline

**Files:**
- Move: `examples/feeding_pi05/` to `examples/pi05/feeding_finetune/`
- Rename: `build_dataset_parallel.py` to `build_dataset.py`, `train_feeding.sh` to `train.sh`
- Modify: `examples/pi05/feeding_finetune/README.md`, `build_dataset.py`, `train.sh`

- [ ] **Step 1: Move and rename**

```bash
mkdir -p examples/pi05
git mv examples/feeding_pi05 examples/pi05/feeding_finetune
git mv examples/pi05/feeding_finetune/build_dataset_parallel.py examples/pi05/feeding_finetune/build_dataset.py
git mv examples/pi05/feeding_finetune/train_feeding.sh examples/pi05/feeding_finetune/train.sh
rm -rf examples/pi05/feeding_finetune/__pycache__
```

- [ ] **Step 2: Update references inside the moved files**

```bash
cd examples/pi05/feeding_finetune
sed -i 's#build_dataset_parallel\.py#build_dataset.py#g; s#train_feeding\.sh#train.sh#g; s#examples/feeding_pi05/#examples/pi05/feeding_finetune/#g' README.md build_dataset.py train.sh convert_nwb_to_lerobot.py
cd -
grep -rn 'feeding_pi05/\|build_dataset_parallel\|train_feeding' examples/ docs/ AGENT_GUIDE.md CLAUDE.md
```

Expected: the grep prints nothing except lines inside `docs/superpowers/` (the spec mentions the old names on purpose). Dataset repo ids such as `rice/feeding_pi05_v1` are Hub names, not paths, and stay unchanged; the sed patterns above only match the path form with a trailing slash.

- [ ] **Step 3: Verify the scripts still parse**

```bash
bash -n examples/pi05/feeding_finetune/train.sh
uv run --no-sync python -m py_compile examples/pi05/feeding_finetune/build_dataset.py examples/pi05/feeding_finetune/convert_nwb_to_lerobot.py
uv run --no-sync python examples/pi05/feeding_finetune/build_dataset.py --help | head -3
```

Expected: no errors; the help text starts with the parallel-driver docstring.

- [ ] **Step 4: Commit**

```bash
git add -A examples/pi05/feeding_finetune examples/feeding_pi05
git commit -m "Move the feeding finetune pipeline to examples/pi05/feeding_finetune

Rename build_dataset_parallel.py to build_dataset.py and train_feeding.sh
to train.sh; update the README paths. No behavior change.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Move the LIBERO folder and share the launcher environment

**Files:**
- Move: `examples/pi05_libero/` to `examples/pi05/libero_shared_autonomy/`
- Create: `examples/pi05/libero_shared_autonomy/env.sh`
- Modify: `setup.sh`, `eval.sh`, `interactive.sh`, `experiment.sh` in the new folder

**Interfaces:**
- Produces: `env.sh` exporting `REPO_ROOT`, `SCRIPT_DIR`, `LIBERO_CONFIG_PATH`, `MUJOCO_GL`, `HF_HUB_CACHE`, `MPLCONFIGDIR`; the launchers `source` it.

- [ ] **Step 1: Move the folder**

```bash
git mv examples/pi05_libero examples/pi05/libero_shared_autonomy
```

- [ ] **Step 2: Write `env.sh`**

`examples/pi05/libero_shared_autonomy/env.sh`:

```bash
#!/usr/bin/env bash
# Environment shared by every launcher in this folder. Source it; do not run it.
#
# Keeps LIBERO's config, the Hugging Face hub cache and Matplotlib's cache under
# the repo's ignored .cache/ directory, and selects EGL for headless MuJoCo.
# HF_HOME is left alone so an existing `hf auth login` credential still applies.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.cache/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/.cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"

UV_RUN=(uv run --locked --extra pi --extra libero)
```

Note the extra `..`: the folder is now three levels below the repo root.

- [ ] **Step 3: Rewrite the four launchers**

`setup.sh`:

```bash
#!/usr/bin/env bash
# One-time setup: install the pi + libero extras and configure LIBERO's assets.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

cd "${REPO_ROOT}"
uv sync --locked --extra pi --extra libero --no-dev
"${UV_RUN[@]}" python "${SCRIPT_DIR}/configure_libero.py" --config-dir "${LIBERO_CONFIG_PATH}"

echo
echo "Pi0.5 + LIBERO setup complete."
echo "Run the smoke evaluation with: ${SCRIPT_DIR}/eval.sh"
```

`eval.sh` keeps its variables and `lerobot-eval` invocation exactly as today; replace everything from `set -euo pipefail` through the `mkdir -p` line with:

```bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
```

and replace the final `exec uv run --locked --extra pi --extra libero lerobot-eval \` with `exec "${UV_RUN[@]}" lerobot-eval \`.

`interactive.sh`:

```bash
#!/usr/bin/env bash
# Interactive LIBERO runner: a REPL where you type an instruction, the policy
# runs one rollout, and you watch it live in a browser. Extra args go to
# interactive.py (e.g. ./interactive.sh --suite libero_object --task-id 2).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "${REPO_ROOT}"
exec "${UV_RUN[@]}" python "${SCRIPT_DIR}/interactive.py" "$@"
```

`experiment.sh`:

```bash
#!/usr/bin/env bash
# Shared-autonomy user study on LIBERO: runs the trials described by a config
# file and records each one to disk. Extra args go to experiment.py
# (e.g. ./experiment.sh --config configs/experiment/reverse_flow_full.yaml).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "${REPO_ROOT}"
exec "${UV_RUN[@]}" python "${SCRIPT_DIR}/experiment.py" "$@"
```

Keep the Apache header comment block at the top of each file where it exists today.

- [ ] **Step 4: Verify**

```bash
for f in examples/pi05/libero_shared_autonomy/*.sh; do bash -n "$f" && echo "ok $f"; done
bash -c 'source examples/pi05/libero_shared_autonomy/env.sh; echo "$REPO_ROOT"; echo "$LIBERO_CONFIG_PATH"'
```

Expected: five `ok` lines; `/home/user/Projects/lerobot` and `/home/user/Projects/lerobot/.cache/libero`.

- [ ] **Step 5: Commit**

```bash
git add -A examples/pi05_libero examples/pi05/libero_shared_autonomy
git commit -m "Move the LIBERO example to examples/pi05/libero_shared_autonomy

Factor the launchers' environment block into env.sh.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 3: Extract the steering algorithms into `src/lerobot/policies/pi05/steering.py`

**Files:**
- Create: `src/lerobot/policies/pi05/steering.py`
- Create: `tests/policies/pi0_pi05/test_pi05_steering.py`
- Modify: `examples/pi05/libero_shared_autonomy/interactive.py` (delete the moved definitions, import them), `experiment.py` (import `FlowAdapter` as `ReversalAdapter`), `teleop_input.py` (import `validate_matrix` from steering)

**Interfaces:**
- Produces (all in `lerobot.policies.pi05.steering`):
  - `ACTION_DIMS: tuple[str, ...]`, `N_ACTION_DIMS = 7`, `GRIPPER_DIM = 6`, `DEADBAND = 0.05`, `GRIPPER_OPEN = -1.0`, `GRIPPER_CLOSE = 1.0`
  - `class TeleopSource(Protocol)` with properties `translation -> np.ndarray`, `gripper -> float`
  - `validate_matrix(matrix, size: int, where: str) -> np.ndarray`
  - `rotation_about_z(degrees: float) -> np.ndarray` (3x3)
  - `translation_matrix(spec, where: str = "matrix", literal_key: str = "M") -> np.ndarray` (3x3)
  - `build_reversal_adapter(spec, corruption: np.ndarray | None = None, where: str = "reversal_adapter") -> np.ndarray | None` (7x7)
  - `class ReversalAdapter(matrix=None, label=None)` with `matrix`, `label`, `describe() -> str`
  - `get_action_mean_std(postprocessor) -> tuple[np.ndarray, np.ndarray]`
  - `class FlowControlPolicy(policy, source: TeleopSource, tau: int, postprocessor)`
  - `reverse_flow(x, velocity, num_steps, adapter=None, n_reverse_steps=None) -> Tensor`
  - `class ReverseFlowSteeringPolicy(policy, source: TeleopSource, postprocessor, adapter: ReversalAdapter | None = None, n_reverse_steps: int | None = None)`

- [ ] **Step 1: Write the failing tests**

`tests/policies/pi0_pi05/test_pi05_steering.py`:

```python
"""CPU tests for the shared-autonomy steering wrappers (no checkpoint needed)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.pi05.steering import (
    DEADBAND,
    GRIPPER_DIM,
    N_ACTION_DIMS,
    FlowControlPolicy,
    ReversalAdapter,
    ReverseFlowSteeringPolicy,
    build_reversal_adapter,
    get_action_mean_std,
    reverse_flow,
    rotation_about_z,
    translation_matrix,
    validate_matrix,
)

NUM_STEPS = 10
CHUNK = 4
MAX_DIM = 32


class FakePolicy:
    """Records the kwargs of predict_action_chunk and runs the hooks like the real sampler."""

    def __init__(self):
        self.config = SimpleNamespace(
            num_inference_steps=NUM_STEPS, chunk_size=CHUNK, max_action_dim=MAX_DIM, n_action_steps=2
        )
        self.calls: list[dict] = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(kwargs)
        x = torch.zeros(1, CHUNK, MAX_DIM)
        noise_fn = kwargs.get("noise_fn")
        if noise_fn is not None:
            x = noise_fn(lambda x_t, t: torch.ones_like(x_t), x)
        hook = kwargs.get("x_t_hook")
        if hook is not None:
            for step in range(NUM_STEPS):
                x = hook(step, 1.0 - step / NUM_STEPS, x)
        return x


class Source:
    def __init__(self, translation=(0.0, 0.0, 0.0), gripper=-1.0):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.gripper = gripper


def postprocessor(mean=None, std=None):
    mean = np.zeros(N_ACTION_DIMS) if mean is None else np.asarray(mean)
    std = np.ones(N_ACTION_DIMS) if std is None else np.asarray(std)
    return SimpleNamespace(steps=[SimpleNamespace(stats={"action": {"mean": mean, "std": std}})])


# ---------------------------------------------------------------- matrices


def test_validate_matrix_rejects_shape_and_nan():
    with pytest.raises(ValueError, match="3x3"):
        validate_matrix([[1, 0], [0, 1]], 3, "m")
    with pytest.raises(ValueError, match="finite"):
        validate_matrix([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], 3, "m")


def test_rotation_about_z_is_exact():
    r = rotation_about_z(90)
    np.testing.assert_allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation_about_z(0), np.eye(3))


def test_translation_matrix_forms():
    np.testing.assert_allclose(translation_matrix({"rotation_z_deg": 20}), rotation_about_z(20))
    np.testing.assert_allclose(translation_matrix({"scale": [1, 2, 3]}), np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(translation_matrix({"M": np.eye(3).tolist()}), np.eye(3))
    np.testing.assert_allclose(translation_matrix(np.eye(3).tolist()), np.eye(3))
    with pytest.raises(ValueError, match="rotation_z_deg"):
        translation_matrix({"rotate": 20})
    with pytest.raises(ValueError, match="scale"):
        translation_matrix({"scale": [1, 2]})


def test_build_reversal_adapter_blocks():
    m = rotation_about_z(20)
    f = build_reversal_adapter({"translation": "corruption", "orientation": "zero", "gripper": "zero"}, m)
    assert f.shape == (7, 7)
    np.testing.assert_allclose(f[:3, :3], m)
    np.testing.assert_allclose(f[3:, :], 0)
    np.testing.assert_allclose(f[:3, 3:], 0)

    f = build_reversal_adapter({"translation": {"scale": [2, 2, 2]}})
    np.testing.assert_allclose(f[:3, :3], 2 * np.eye(3))
    np.testing.assert_allclose(f[3:, 3:], np.eye(4))

    assert build_reversal_adapter(None) is None
    np.testing.assert_allclose(build_reversal_adapter({"F": np.eye(7).tolist()}), np.eye(7))


def test_build_reversal_adapter_errors():
    with pytest.raises(ValueError, match="corruption"):
        build_reversal_adapter({"translation": "corruption"}, None)
    with pytest.raises(ValueError, match="orientation"):
        build_reversal_adapter({"orientation": "half"})
    with pytest.raises(ValueError, match="unknown"):
        build_reversal_adapter({"rotation": "zero"})
    with pytest.raises(ValueError, match="F"):
        build_reversal_adapter({"F": np.eye(7).tolist(), "gripper": "zero"})


def test_reversal_adapter_holder():
    holder = ReversalAdapter()
    assert holder.matrix is None
    assert "off" in holder.describe()
    holder.matrix = np.eye(7)
    holder.label = "identity"
    assert "identity" in holder.describe()
    with pytest.raises(ValueError):
        holder.matrix = np.eye(3)


# ---------------------------------------------------------------- reverse_flow


def test_reverse_flow_constant_field_closed_form():
    x = torch.zeros(1, CHUNK, MAX_DIM)
    out = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS)
    torch.testing.assert_close(out, torch.ones_like(x))  # 10 steps of h=0.1, v=1
    out = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS, n_reverse_steps=3)
    torch.testing.assert_close(out, torch.full_like(x, 0.3))


def test_reverse_flow_visits_times_zero_to_one():
    seen = []

    def velocity(x_t, t):
        seen.append(round(t, 6))
        return torch.zeros_like(x_t)

    reverse_flow(torch.zeros(1, CHUNK, MAX_DIM), velocity, NUM_STEPS)
    assert seen == [round(k / NUM_STEPS, 6) for k in range(NUM_STEPS)]


def test_reverse_flow_adapter_touches_only_action_dims():
    x = torch.zeros(1, CHUNK, MAX_DIM)
    adapter = torch.zeros(7, 7)
    adapter[0, 0] = 2.0  # double dx, kill the other six action dims
    out = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS, adapter=adapter)
    torch.testing.assert_close(out[..., 0], torch.full((1, CHUNK), 2.0))
    torch.testing.assert_close(out[..., 1:7], torch.zeros(1, CHUNK, 6))
    torch.testing.assert_close(out[..., 7:], torch.ones(1, CHUNK, MAX_DIM - 7))  # padding untouched
    identity = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS, adapter=torch.eye(7))
    torch.testing.assert_close(identity, torch.ones_like(x))


# ---------------------------------------------------------------- FlowControlPolicy


def test_get_action_mean_std_reads_first_action_stats():
    mean, std = get_action_mean_std(postprocessor(mean=np.arange(7), std=np.zeros(7)))
    np.testing.assert_allclose(mean, np.arange(7))
    assert np.all(std >= 1e-6)
    with pytest.raises(RuntimeError):
        get_action_mean_std(SimpleNamespace(steps=[]))


def test_flow_control_writes_normalized_translation_for_tau_steps():
    policy = FakePolicy()
    source = Source(translation=(0.5, 0.0, -0.5))
    wrapper = FlowControlPolicy(policy, source, tau=3, postprocessor(mean=[0.1] * 7, std=[0.5] * 7))
    wrapper.reset()
    action = wrapper.select_action({})
    assert action.shape == (1, MAX_DIM)
    assert wrapper.hook_calls == 3
    expected = (np.array([0.5, 0.0, -0.5]) - 0.1) / 0.5
    np.testing.assert_allclose(action[0, :3].numpy(), expected, atol=1e-6)
    assert policy.resets == 1


def test_flow_control_skips_idle_input_and_queues_actions():
    policy = FakePolicy()
    wrapper = FlowControlPolicy(policy, Source(translation=(DEADBAND / 2, 0, 0)), 5, postprocessor())
    first = wrapper.select_action({})
    second = wrapper.select_action({})
    assert wrapper.hook_calls == 0
    torch.testing.assert_close(first, torch.zeros(1, MAX_DIM))
    torch.testing.assert_close(second, torch.zeros(1, MAX_DIM))
    assert len(policy.calls) == 1  # n_action_steps=2 actions per chunk


# ---------------------------------------------------------------- ReverseFlowSteeringPolicy


def test_reverse_flow_steering_passes_schedule_only_while_steering():
    policy = FakePolicy()
    source = Source(translation=(0.0, 0.0, 0.0))
    wrapper = ReverseFlowSteeringPolicy(policy, source, postprocessor(), n_reverse_steps=4)
    wrapper.select_action({})
    assert "flow_start_time" not in policy.calls[-1]
    assert wrapper.steered_chunks == 0

    source.translation = np.array([1.0, 0.0, 0.0])
    wrapper.select_action({})
    wrapper.select_action({})  # served from the queue, no new call
    assert len(policy.calls) == 2
    assert policy.calls[-1]["flow_start_time"] == pytest.approx(0.4)
    assert policy.calls[-1]["num_forward_steps"] == 6
    assert wrapper.steered_chunks == 1
    assert len(wrapper.reconstruction_errors) == 1


def test_reverse_flow_steering_full_reversal_has_no_schedule_kwargs():
    policy = FakePolicy()
    wrapper = ReverseFlowSteeringPolicy(policy, Source(translation=(1.0, 0, 0)), postprocessor())
    wrapper.select_action({})
    assert "flow_start_time" not in policy.calls[-1]
    assert wrapper.n_reverse_steps is None


def test_reverse_flow_steering_validates_n_reverse_steps():
    wrapper = ReverseFlowSteeringPolicy(FakePolicy(), Source(), postprocessor())
    for bad in (0, 11, 2.5, "3"):
        with pytest.raises(ValueError):
            wrapper.n_reverse_steps = bad
    wrapper.n_reverse_steps = NUM_STEPS
    assert wrapper.n_reverse_steps is None  # N means full reversal


def test_reference_chunk_uses_last_gripper_and_normalization():
    policy = FakePolicy()
    post = postprocessor(mean=[0.0] * 6 + [0.5], std=[2.0] * 7)
    wrapper = ReverseFlowSteeringPolicy(policy, Source(translation=(1.0, 0, 0)), post)
    ref = wrapper.reference_chunk(np.array([1.0, 0.0, 0.0]))
    assert ref.shape == (1, CHUNK, MAX_DIM)
    assert ref[0, 0, 0].item() == pytest.approx(0.5)  # (1 - 0) / 2
    assert ref[0, 0, GRIPPER_DIM].item() == pytest.approx((-1.0 - 0.5) / 2.0)  # open at start
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --no-sync pytest tests/policies/pi0_pi05/test_pi05_steering.py -q 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'lerobot.policies.pi05.steering'`.

- [ ] **Step 3: Write `steering.py`**

Create `src/lerobot/policies/pi05/steering.py` with the Apache header used by the other files in that folder, then:

```python
"""Shared-autonomy steering of pi0.5's flow-matching sampler.

Two ways for an operator's translation command to steer the policy without
touching the executed action, which is always the model's own output:

- ``FlowControlPolicy``: the command is written into dims 0-2 of ``x_t`` for the
  first ``tau`` denoising steps of each chunk (the rest denoise freely).
- ``ReverseFlowSteeringPolicy``: a uniform-velocity reference chunk built from
  the command is integrated *backward* through the policy's velocity field to
  the latent noise that maps to it (Flow Reversal Steering, Tang et al. 2026,
  arXiv:2606.13675); the forward flow then starts from that noise.

Both read an operator ``TeleopSource`` and the checkpoint's action statistics
from the postprocessor. ``reverse_flow`` is the integrator; ``ReversalAdapter``
holds an optional 7x7 matrix F that adapts the reversal's velocity field only.
"""

from collections import deque
from typing import Protocol

import numpy as np
import torch

# LIBERO's OSC_POSE action: end-effector position deltas, axis-angle deltas, gripper.
ACTION_DIMS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
N_ACTION_DIMS = len(ACTION_DIMS)
GRIPPER_DIM = ACTION_DIMS.index("gripper")
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0
DEADBAND = 0.05  # below this (env units, full deflection = 1) the operator counts as idle


class TeleopSource(Protocol):
    """What the steering wrappers read: the operator's current command."""

    @property
    def translation(self) -> np.ndarray:  # shape (3,), env units in [-1, 1]
        ...

    @property
    def gripper(self) -> float:  # GRIPPER_OPEN or GRIPPER_CLOSE
        ...


# ---------------------------------------------------------------- matrices


def validate_matrix(matrix, size: int, where: str) -> np.ndarray:
    """Coerce `matrix` to a finite (size, size) float array or raise ValueError."""
    try:
        m = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: entries must be numbers ({e})") from e
    if m.shape != (size, size):
        raise ValueError(f"{where}: expected a {size}x{size} matrix, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{where}: entries must be finite")
    return m


def rotation_about_z(degrees: float) -> np.ndarray:
    """3x3 rotation of a translation command about the z axis."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def translation_matrix(spec, where: str = "matrix", literal_key: str = "M") -> np.ndarray:
    """Build a 3x3 from a spec: {rotation_z_deg: d}, {scale: [sx, sy, sz]}, {M: rows} or bare rows."""
    if isinstance(spec, dict):
        keys = set(spec)
        if keys == {"rotation_z_deg"}:
            return rotation_about_z(float(spec["rotation_z_deg"]))
        if keys == {"scale"}:
            scale = np.asarray(spec["scale"], dtype=np.float64)
            if scale.shape != (3,) or not np.all(np.isfinite(scale)):
                raise ValueError(f"{where}: scale must be three finite numbers, got {spec['scale']!r}")
            return np.diag(scale)
        if keys == {literal_key}:
            return validate_matrix(spec[literal_key], 3, f"{where}.{literal_key}")
        raise ValueError(
            f"{where}: expected exactly one of rotation_z_deg, scale or {literal_key}, got {sorted(keys)}"
        )
    return validate_matrix(spec, 3, where)


_BLOCK_CHOICES = ("identity", "zero")
_ORIENTATION = slice(3, 6)
_GRIPPER = slice(GRIPPER_DIM, GRIPPER_DIM + 1)


def build_reversal_adapter(
    spec, corruption: np.ndarray | None = None, where: str = "reversal_adapter"
) -> np.ndarray | None:
    """Build the 7x7 reversal adapter F from a spec, or None when the spec is None.

    Forms: ``{F: rows}`` (literal), or blocks ``translation`` (``corruption`` to
    copy the 3x3 command corruption, ``identity``, ``zero``, or any
    ``translation_matrix`` spec), ``orientation`` and ``gripper`` (``identity``
    or ``zero``). Missing blocks are identity.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(spec).__name__}")
    if "F" in spec:
        if set(spec) != {"F"}:
            raise ValueError(f"{where}: a literal F cannot be combined with block keys")
        return validate_matrix(spec["F"], N_ACTION_DIMS, f"{where}.F")
    unknown = set(spec) - {"translation", "orientation", "gripper"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (expected translation, orientation, gripper)")

    f = np.eye(N_ACTION_DIMS)
    translation = spec.get("translation", "identity")
    if translation == "corruption":
        if corruption is None:
            raise ValueError(f"{where}.translation: 'corruption' requested but control.corruption is null")
        f[:3, :3] = corruption
    elif translation == "zero":
        f[:3, :3] = 0.0
    elif translation != "identity":
        f[:3, :3] = translation_matrix(translation, f"{where}.translation")
    for key, dims in (("orientation", _ORIENTATION), ("gripper", _GRIPPER)):
        choice = spec.get(key, "identity")
        if choice not in _BLOCK_CHOICES:
            raise ValueError(f"{where}.{key}: expected identity or zero, got {choice!r}")
        if choice == "zero":
            f[dims, dims] = 0.0
    return f


class ReversalAdapter:
    """Mutable holder for the reversal adapter F (None = off).

    ``ReverseFlowSteeringPolicy`` reads the holder on every chunk, so replacing
    ``matrix`` takes effect immediately. ``label`` is display-only.
    """

    def __init__(self, matrix=None, label: str | None = None):
        self.matrix = matrix
        self.label = label

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @matrix.setter
    def matrix(self, value) -> None:
        self._matrix = None if value is None else validate_matrix(value, N_ACTION_DIMS, "reversal adapter")

    def describe(self) -> str:
        if self._matrix is None:
            return "Reversal adapter: off."
        identity = " (identity, a no-op)" if np.allclose(self._matrix, np.eye(N_ACTION_DIMS)) else ""
        rows = "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in self._matrix)
        return f"Reversal adapter: reverse integration uses x_t += h * F @ v, F = {self.label}{identity} = [{rows}]."
```

Then append, moved verbatim from `examples/pi05/libero_shared_autonomy/interactive.py`, in this order: `get_action_mean_std`, `FlowControlPolicy`, `reverse_flow`, `ReverseFlowSteeringPolicy`. Apply these edits while moving:

- `FlowControlPolicy.__init__(self, policy, reader: NoisyReader, tau, postprocessor)` becomes `(self, policy, source: TeleopSource, tau: int, postprocessor)`; rename `self._reader` to `self._source` throughout both classes. Delete the class attribute `DEADBAND = 0.05` from `FlowControlPolicy` and `DEADBAND = FlowControlPolicy.DEADBAND` from `ReverseFlowSteeringPolicy`; use the module `DEADBAND`.
- In `ReverseFlowSteeringPolicy`: delete `GRIPPER_DIM = ACTION_LABELS.index("gripper")`; replace every `len(ACTION_LABELS)` with `N_ACTION_DIMS`; replace `self.GRIPPER_DIM` with `GRIPPER_DIM`; the `adapter` parameter type becomes `ReversalAdapter | None`; the class docstring's "An optional `FlowAdapter`" becomes "An optional `ReversalAdapter`".
- Keep both `@torch.compiler.disable` decorators and their comments.

- [ ] **Step 4: Run the tests**

```bash
uv run --no-sync pytest tests/policies/pi0_pi05/test_pi05_steering.py -q 2>&1 | tail -3
```

Expected: all pass. If `test_reference_chunk_uses_last_gripper_and_normalization` fails on the gripper value, check that `_gripper_ref` is initialized with `self._normalize_gripper(GRIPPER_OPEN)` as in the original.

- [ ] **Step 5: Point the example at the new module**

In `examples/pi05/libero_shared_autonomy/interactive.py`:

1. Delete the definitions of `FlowAdapter`, `load_flow_adapter`, `load_adapter`, `describe_adapter`, `get_action_mean_std`, `FlowControlPolicy`, `reverse_flow`, `ReverseFlowSteeringPolicy`.
2. Add to the `lerobot` import block:

```python
from lerobot.policies.pi05.steering import (
    N_ACTION_DIMS,
    FlowControlPolicy,
    ReversalAdapter as FlowAdapter,
    ReverseFlowSteeringPolicy,
    validate_matrix,
)
```

3. Re-add the three thin helpers that the REPL still uses, right after `MODES`:

```python
def load_flow_adapter(path: str | Path):
    """Read the 7x7 flow-reversal adapter F from a YAML file."""
    return load_matrix(path, size=N_ACTION_DIMS, keys=("F", "matrix"))


def load_adapter(adapter: FlowAdapter, path: Path) -> None:
    adapter.matrix = load_flow_adapter(path)
    adapter.label = path.name


def describe_adapter(adapter: FlowAdapter) -> str:
    return adapter.describe()
```

4. Remove `validate_matrix` from the `teleop_input` import list (it now comes from steering). In `teleop_input.py`, delete the `validate_matrix` definition and add `from lerobot.policies.pi05.steering import validate_matrix` after the numpy import; add `# noqa: E402`-free ordering by placing it in the third-party block (ruff isort treats `lerobot` as first-party, so it goes in its own block after `yaml`).

`experiment.py` imports `FlowAdapter` from `interactive`; that name now resolves to the aliased `ReversalAdapter`, so no change there.

- [ ] **Step 6: Verify the example still imports and lint passes**

```bash
uv run --no-sync python -c "import sys; sys.path.insert(0, 'examples/pi05/libero_shared_autonomy'); import interactive, experiment; print('ok')"
uv run --no-sync ruff check src/lerobot/policies/pi05/steering.py tests/policies/pi0_pi05/test_pi05_steering.py examples/pi05/libero_shared_autonomy
uv run --no-sync ruff format src/lerobot/policies/pi05/steering.py tests/policies/pi0_pi05/test_pi05_steering.py examples/pi05/libero_shared_autonomy
```

Expected: `ok`, no ruff errors.

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/policies/pi05/steering.py tests/policies/pi0_pi05/test_pi05_steering.py examples/pi05/libero_shared_autonomy
git commit -m "Move pi0.5 shared-autonomy steering into lerobot.policies.pi05.steering

FlowControlPolicy, ReverseFlowSteeringPolicy, reverse_flow and the reversal
adapter holder leave the LIBERO example. Adds intent-level matrix builders
(rotation_z_deg, scale, block-form reversal adapter) and CPU tests.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 4: Tests for the sampler hooks in `modeling_pi05.py`

**Files:**
- Modify: `tests/policies/pi0_pi05/test_pi05.py` (append)

**Interfaces:**
- Consumes: `PI05Pytorch.sample_actions(..., noise_fn=, flow_start_time=, num_forward_steps=)` as merged from the branch.

These need the real model (the sampler encodes a prefix with PaliGemma), so they carry the file's existing `@require_cuda` and `@require_hf_token` decorators, like `test_policy_instantiation`.

- [ ] **Step 1: Append the tests**

Add at the end of `tests/policies/pi0_pi05/test_pi05.py`:

```python
def _small_policy():
    """A pi0.5 with 7-dim actions built the same way as test_policy_instantiation."""
    set_seed(0)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32", num_inference_steps=4)
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = PI05Policy(config).to("cuda").eval()
    batch = {
        "observation.state": torch.zeros(1, 14, device="cuda"),
        "observation.images.base_0_rgb": torch.zeros(1, 3, 224, 224, device="cuda"),
        "task": ["do something"],
    }
    return policy, batch


@require_cuda
@require_hf_token
def test_sample_actions_noise_fn_replaces_starting_noise():
    policy, batch = _small_policy()
    seen = {}

    def noise_fn(velocity, noise):
        seen["noise_shape"] = tuple(noise.shape)
        v = velocity(noise, 1.0)  # the velocity closure must evaluate the field
        seen["velocity_shape"] = tuple(v.shape)
        return torch.zeros_like(noise)

    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch, noise_fn=noise_fn)
    assert seen["noise_shape"] == (1, policy.config.chunk_size, policy.config.max_action_dim)
    assert seen["velocity_shape"] == seen["noise_shape"]
    assert chunk.shape[0] == 1


@require_cuda
@require_hf_token
def test_sample_actions_partial_schedule_lands_on_zero():
    policy, batch = _small_policy()
    times = []

    def x_t_hook(step, time, x_t):
        times.append(round(time, 6))
        return x_t

    with torch.inference_mode():
        policy.predict_action_chunk(batch, x_t_hook=x_t_hook, flow_start_time=0.5, num_forward_steps=2)
    assert times == [0.5, 0.25]  # two Euler steps of -0.25 from t=0.5 reach t=0

    with pytest.raises(ValueError, match="num_forward_steps"):
        with torch.inference_mode():
            policy.predict_action_chunk(batch, flow_start_time=0.5, num_forward_steps=0)
```

Check the top of the file: `PI05Policy`, `PI05Config`, `torch`, `pytest`, and `set_seed` must already be imported (they are used by the existing tests); add `import pytest` if missing.

- [ ] **Step 2: Run the tests**

```bash
uv run --no-sync pytest tests/policies/pi0_pi05/test_pi05.py -q -k "noise_fn or partial_schedule" 2>&1 | tail -5
```

Expected: 2 passed (or 2 skipped with "requires HF token" if no token is logged in; in that case run `uv run --no-sync hf auth login` and rerun). Note the model weights are random here; only the schedule and hook plumbing are under test, so the run takes seconds after the tokenizer download.

- [ ] **Step 3: Commit**

```bash
git add tests/policies/pi0_pi05/test_pi05.py
git commit -m "Test the pi0.5 sampler's noise_fn and partial-schedule hooks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `teleop.py`: one module for operator input

**Files:**
- Move: `examples/pi05/libero_shared_autonomy/teleop_input.py` to `teleop.py`
- Delete: `examples/pi05/libero_shared_autonomy/spacemouse.py` (content merged into `teleop.py`)
- Modify: `interactive.py`, `experiment.py` (imports)
- Create: `tests/examples/__init__.py`, `tests/examples/pi05_libero_shared_autonomy/__init__.py`, `conftest.py`, `test_teleop.py`

**Interfaces:**
- Produces (module `teleop`, importable once the example dir is on `sys.path`):
  - SpaceMouse: `SPACEMOUSE_HID_ID`, `AXIS_SCALE`, `AXIS_SOURCE`, `AXIS_SIGN`, `find_spacemouse_device`, `parse_report`, `normalize`, `SpaceMouseReader`
  - Keyboard: `KEY_PUSH`, `DEFAULT_SPEED`, `FAST_KEY`, `STALE_AFTER`, `KeyboardReader(speed, stale_after, clock)` with `update(held, toggles)`, `translation`, `gripper`, `held`
  - `CombinedReader(sources)`
  - `CommandCorruption(source, matrix=None, label=None)` with `matrix`, `label`, `describe()`, `translation`, `gripper`
  - `build_corruption(spec, where="corruption") -> np.ndarray | None`
  - `read_matrix_spec(path) -> dict | list` (YAML file for the REPL's live loading)
  - `NoisyReader(source, std=0.0, rng=None)`, `RecordingReader(source)`
  - `TeleopChain(keyboard, input_noise=0.0)` with `reader`, `raw`, `served`, `corruption`, `noisy`, `combined`, `spacemouse`, `attach_spacemouse()`
  - Re-exported from steering: `DEADBAND`, `GRIPPER_OPEN`, `GRIPPER_CLOSE`

- [ ] **Step 1: Write the test scaffolding and failing tests**

`tests/examples/__init__.py` and `tests/examples/pi05_libero_shared_autonomy/__init__.py`: empty files.

`tests/examples/pi05_libero_shared_autonomy/conftest.py`:

```python
"""Make the example's modules importable (they are scripts, not a package)."""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "libero_shared_autonomy"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
```

`tests/examples/pi05_libero_shared_autonomy/test_teleop.py`:

```python
import numpy as np
import pytest
import teleop
from teleop import (
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    CombinedReader,
    CommandCorruption,
    KeyboardReader,
    NoisyReader,
    RecordingReader,
    TeleopChain,
    build_corruption,
    normalize,
    parse_report,
    read_matrix_spec,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Source:
    def __init__(self, translation=(0.0, 0.0, 0.0), gripper=GRIPPER_OPEN):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.gripper = gripper


def test_parse_report_and_normalize():
    kind, xyz = parse_report(bytes([1]) + (350).to_bytes(2, "little", signed=True) * 3)
    assert kind == "translation" and xyz == (350, 350, 350)
    assert parse_report(bytes([3, 1, 0, 0, 0])) == ("buttons", 1)
    assert parse_report(b"\x09")[0] == "other"
    assert np.all(np.abs(normalize((700, -700, 0))) <= 1.0)


def test_keyboard_reader_speed_stale_and_toggle_parity():
    clock = Clock()
    kb = KeyboardReader(speed=0.5, stale_after=1.0, clock=clock)
    kb.update({"ArrowUp"}, toggles=0)
    forward = kb.translation
    assert np.linalg.norm(forward) == pytest.approx(0.5)
    kb.update({"ArrowUp", "Shift"}, toggles=0)
    assert np.linalg.norm(kb.translation) == pytest.approx(1.0)
    kb.update({"ArrowUp"}, toggles=1)
    assert kb.gripper == GRIPPER_CLOSE
    kb.update({"ArrowUp"}, toggles=2)
    assert kb.gripper == GRIPPER_OPEN
    kb.update({"ArrowUp"}, toggles=0)  # page reload restarts the counter: no toggle
    assert kb.gripper == GRIPPER_OPEN
    clock.now = 2.0
    np.testing.assert_array_equal(kb.translation, 0)  # stale
    assert kb.held == set()
    assert kb.gripper == GRIPPER_OPEN  # gripper keeps its state


def test_combined_reader_priority_and_gripper_parity():
    a, b = Source((0.0, 0.0, 0.0)), Source((0.3, 0.0, 0.0), GRIPPER_CLOSE)
    combined = CombinedReader([a, b])
    np.testing.assert_array_equal(combined.translation, [0.3, 0.0, 0.0])
    assert combined.gripper == GRIPPER_CLOSE
    a.translation = np.array([-0.5, 0.0, 0.0])
    np.testing.assert_array_equal(combined.translation, [-0.5, 0.0, 0.0])  # first deflected wins
    a.gripper = GRIPPER_CLOSE
    assert combined.gripper == GRIPPER_OPEN  # two closed = even parity


def test_command_corruption_applies_only_when_commanding():
    source = Source((1.0, 0.0, 0.0))
    corrupted = CommandCorruption(source, matrix=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], label="rot90")
    np.testing.assert_allclose(corrupted.translation, [0.0, 1.0, 0.0])
    source.translation = np.array([DEADBAND / 2, 0.0, 0.0])
    np.testing.assert_allclose(corrupted.translation, [DEADBAND / 2, 0.0, 0.0])
    corrupted.matrix = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    source.translation = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(corrupted.translation, [1.0, 0.0, 0.0])  # clipped
    assert "rot90" in corrupted.describe()
    corrupted.matrix = None
    assert "off" in corrupted.describe()
    with pytest.raises(ValueError):
        corrupted.matrix = np.eye(2)


def test_build_corruption_and_read_matrix_spec(tmp_path):
    assert build_corruption(None) is None
    np.testing.assert_allclose(build_corruption({"rotation_z_deg": 0}), np.eye(3))
    path = tmp_path / "m.yaml"
    path.write_text("rotation_z_deg: 90\n")
    m = build_corruption(read_matrix_spec(path))
    np.testing.assert_allclose(m @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)
    path.write_text("M:\n  - [1, 0, 0]\n  - [0, 1, 0]\n  - [0, 0, 1]\n")
    np.testing.assert_allclose(build_corruption(read_matrix_spec(path)), np.eye(3))
    path.write_text("- 1\n")
    with pytest.raises(ValueError):
        build_corruption(read_matrix_spec(path))


def test_noisy_reader_is_seeded_and_idle_stays_idle():
    source = Source((0.5, 0.5, 0.5))
    a = NoisyReader(source, std=0.1, rng=np.random.default_rng(1)).translation
    b = NoisyReader(source, std=0.1, rng=np.random.default_rng(1)).translation
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, source.translation)
    source.translation = np.zeros(3)
    np.testing.assert_array_equal(NoisyReader(source, std=0.1).translation, 0)
    with pytest.raises(ValueError):
        NoisyReader(source, std=-1)


def test_recording_reader_counts_reads():
    rec = RecordingReader(Source((0.2, 0.0, 0.0), GRIPPER_CLOSE))
    assert rec.reads == 0
    rec.translation
    rec.translation
    assert rec.reads == 2
    np.testing.assert_array_equal(rec.last_translation, [0.2, 0.0, 0.0])
    assert rec.gripper == GRIPPER_CLOSE and rec.last_gripper == GRIPPER_CLOSE


def test_teleop_chain_order_and_records():
    kb = KeyboardReader(clock=lambda: 0.0)
    kb.update({"ArrowRight"}, toggles=0)  # full push right at half speed
    chain = TeleopChain(kb, input_noise=0.0)
    chain.corruption.matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    served = chain.reader.translation
    raw = chain.raw.last_translation
    assert chain.served.reads == 1 and chain.raw.reads == 1
    np.testing.assert_allclose(served, chain.corruption.matrix @ raw)
    assert chain.spacemouse is None
    assert chain.combined.sources == [kb]


def test_module_reexports_shared_constants():
    from lerobot.policies.pi05 import steering

    assert teleop.DEADBAND is steering.DEADBAND
    assert teleop.GRIPPER_OPEN == steering.GRIPPER_OPEN
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --no-sync pytest tests/examples/pi05_libero_shared_autonomy/test_teleop.py -q 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'teleop'`.

- [ ] **Step 3: Build `teleop.py`**

```bash
cd examples/pi05/libero_shared_autonomy
git mv teleop_input.py teleop.py
```

Edit `teleop.py`:

1. Replace the module docstring with:

```python
"""Operator input for the LIBERO shared-autonomy runners.

Where the operator's command comes from and what happens to it on the way to
the policy:

    SpaceMouse + keyboard -> CombinedReader -> RecordingReader (raw)
        -> CommandCorruption (M @ x) -> NoisyReader (+ noise)
        -> RecordingReader (served)

`TeleopChain` builds that pipeline; the modes consume `chain.reader`. Every
reader exposes `translation` (shape 3, env units in [-1, 1]) and `gripper`
(-1 open, +1 close), the `TeleopSource` protocol from
`lerobot.policies.pi05.steering`.
"""
```

2. Replace the import block with:

```python
import struct
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import yaml

from lerobot.policies.pi05.steering import (
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    translation_matrix,
    validate_matrix,
)

__all__ = ["DEADBAND", "GRIPPER_CLOSE", "GRIPPER_OPEN"]  # re-exported for readers of this module
```

3. Paste the body of `spacemouse.py` (everything after its imports: the `SPACEMOUSE_HID_ID` through `SpaceMouseReader` definitions) right after the imports, deleting its `GRIPPER_OPEN`/`GRIPPER_CLOSE` lines (they now come from steering). Then `git rm spacemouse.py`.

4. Delete the local `DEADBAND = 0.05` constant and the `validate_matrix`, `load_matrix`, and `load_corruption_matrix` definitions.

5. Rename `MatrixReader` to `CommandCorruption`, keep its behavior, and add `describe()`:

```python
    def describe(self) -> str:
        if self._matrix is None:
            return "Command corruption: off."
        rows = "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in self._matrix)
        return f"Command corruption: x/y/z -> M @ x/y/z while you command, M = {self.label} = [{rows}]."
```

Its `matrix` setter calls `validate_matrix(value, 3, "corruption matrix")` as before.

6. Add after `CommandCorruption`:

```python
def build_corruption(spec, where: str = "corruption") -> np.ndarray | None:
    """3x3 M from a spec ({rotation_z_deg: d}, {scale: [...]}, {M: rows}, bare rows) or None."""
    return None if spec is None else translation_matrix(spec, where)


def read_matrix_spec(path: str | Path):
    """Read a matrix spec from a YAML file (for the REPL's live `corruption` / `adapter` commands)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"{path}: empty file")
    return data
```

7. Move `TeleopChain` from `interactive.py` to the end of `teleop.py`, replacing `MatrixReader(self.raw)` with `CommandCorruption(self.raw)`. Its docstring stays.

- [ ] **Step 4: Update the entry points' imports**

In `interactive.py`: replace the `from spacemouse import ...` and `from teleop_input import ...` lines with

```python
from teleop import (
    CombinedReader,
    CommandCorruption,
    KeyboardReader,
    NoisyReader,
    RecordingReader,
    TeleopChain,
    build_corruption,
    read_matrix_spec,
)
```

then delete the `TeleopChain` class, the `DEFAULT_CORRUPTION_FILE` line, and `load_corruption`; replace `describe_corruption(x)` calls with `x.describe()` and delete `describe_corruption`; replace `load_corruption(corruption, path)` calls with

```python
corruption.matrix = build_corruption(read_matrix_spec(path))
corruption.label = path.name
```

(there are two: one for the `--deterministic-corruption` flag, one in the REPL). Do the same for the adapter: replace `load_adapter(flow_adapter, path)` with

```python
flow_adapter.matrix = build_reversal_adapter(read_matrix_spec(path))
flow_adapter.label = path.name
```

importing `build_reversal_adapter` from steering, and delete `load_flow_adapter`, `load_adapter`, `load_matrix` uses. Remove `GRIPPER_OPEN` from the imports if it is no longer referenced. Type hints `MatrixReader` become `CommandCorruption`.

In `experiment.py`: the `from interactive import (...)` list drops `describe_adapter`, `describe_corruption`, `load_adapter`, `load_corruption`, `KeyboardReader`, `TeleopChain`; add `from teleop import KeyboardReader, TeleopChain, build_corruption, read_matrix_spec` and `from lerobot.policies.pi05.steering import build_reversal_adapter`; replace the two `load_*` calls and two `describe_*` calls as above.

- [ ] **Step 5: Run the tests and import check**

```bash
uv run --no-sync pytest tests/examples/pi05_libero_shared_autonomy/test_teleop.py tests/policies/pi0_pi05/test_pi05_steering.py -q 2>&1 | tail -3
uv run --no-sync python -c "import sys; sys.path.insert(0, 'examples/pi05/libero_shared_autonomy'); import interactive, experiment; print('ok')"
uv run --no-sync ruff check examples/pi05/libero_shared_autonomy tests/examples && uv run --no-sync ruff format examples/pi05/libero_shared_autonomy tests/examples
```

Expected: all pass, `ok`, ruff clean. `ruff` will flag `import teleop` ordering in the test as third-party: that is correct for a script module, leave it where ruff puts it.

- [ ] **Step 6: Commit**

```bash
git add -A examples/pi05/libero_shared_autonomy tests/examples
git commit -m "Merge spacemouse and teleop_input into teleop.py

One module for operator input: devices, merge, command corruption, noise,
recording and the TeleopChain. Corruption matrices are built from intent
specs (rotation_z_deg, scale, literal). Adds CPU tests.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 6: `config.py` and the new config files

**Files:**
- Create: `examples/pi05/libero_shared_autonomy/config.py`
- Create: `configs/interactive.yaml`, `configs/experiment/base.yaml` and the five condition files
- Create: `tests/examples/pi05_libero_shared_autonomy/test_config.py`

The old `config_*.yaml`, `deterministic_corruption.yaml` and `flow_reversal_adapter.yaml` stay until Task 9 rewrites the entry points that still read them.

**Interfaces:**
- Produces (module `config`):
  - `MODES`, `TASK_ORDERS`, `PROMPT_FROM_TASK`, `CONFIG_DIR`, `DEFAULT_INTERACTIVE_CONFIG`
  - `@dataclass ControlSettings(mode, tau, n_reverse_steps, input_noise, max_steps, corruption, reversal_adapter, corruption_matrix, reversal_adapter_matrix)`
  - `@dataclass SessionSettings(policy_path, n_action_steps, compile, suite, task_id, port, output_dir, control)`
  - `@dataclass ExperimentSettings(name, session, n_trials, seed, task_order, output_dir, task_ids, prompt)`
  - `read_yaml(path) -> dict`, `deep_merge(base, overlay) -> dict`, `load_yaml_with_extends(path) -> dict`, `flatten(data, schema, where) -> dict`
  - `spec_label(spec) -> str | None`
  - `build_control(flat: dict, where: str) -> ControlSettings`
  - `load_interactive_settings(path: Path | None, overrides: dict) -> SessionSettings`
  - `load_experiment_settings(path: Path, overrides: dict) -> ExperimentSettings`

- [ ] **Step 1: Write the config files**

`configs/interactive.yaml`:

```yaml
# Settings for interactive.py (`interactive.sh --config [PATH]`).
#
# Every key mirrors a command-line flag. Delete a key or set it to null to keep
# the built-in default; a flag passed on the command line still wins. Unknown
# keys are rejected so a typo cannot silently do nothing.

policy:
  path: lerobot/pi05_libero_finetuned
  n_action_steps: 10
  compile: false

scene:
  suite: libero_spatial       # libero_spatial | libero_object | libero_goal | libero_10 | libero_90 | libero_100
  task_id: 0

control:
  mode: policy                # policy | teleop | shared_override | shared_flow_control | shared_reverse_flow_steering
  tau: 5                      # shared_flow_control: leading denoising steps your input steers
  n_reverse_steps: null       # shared_reverse_flow_steering: steps to reverse the reference (null = all the way to noise)
  input_noise: 0.0            # std of the Gaussian noise added to your x/y/z command (0 = off)
  corruption: null            # what the robot gets = M @ what you commanded, e.g. {rotation_z_deg: 20}
  reversal_adapter: null      # reverse integration follows F @ v, e.g. {translation: corruption, orientation: zero, gripper: zero}

server:
  port: 8765

output_dir: outputs/pi05_libero_interactive
```

`configs/experiment/base.yaml`:

```yaml
# Shared settings for every experiment condition in this folder. A condition
# file says `extends: base.yaml` and overrides only what it changes. Unknown
# keys are rejected; a key set to null keeps the built-in default.

experiment:
  n_trials: 10                # trials per run
  seed: 0                     # seeds the task schedule (same seed = same tasks in every condition)
  task_order: random          # random (uniform, with replacement) | shuffled (permutations) | sequential
  output_dir: outputs/pi05_libero_experiments

scene:
  suite: libero_goal
  task_ids: null              # null = every task in the suite, or e.g. [0, 3, 7]

# Instruction handed to the VLA on every trial. The human is always shown the
# real task; a generic prompt dissociates the two. The literal string `task`
# gives the VLA the scene's own instruction instead.
prompt: "do something"

policy:
  path: lerobot/pi05_libero_finetuned
  n_action_steps: 10
  compile: false

control:
  mode: shared_reverse_flow_steering
  tau: 5                      # shared_flow_control only
  n_reverse_steps: null       # shared_reverse_flow_steering only; null = all the way to noise
  input_noise: 0.0
  max_steps: null             # null = the suite's own episode length
  corruption: null            # what the robot gets = M @ what the operator commanded (env units)
  reversal_adapter: null      # reverse integration follows F @ v (normalized action space)

server:
  port: 8765
```

Condition files (each starts with `extends: base.yaml`):

`flow_control_tau8.yaml`:

```yaml
# Shared flow control, tau = 8, clean operator command.
extends: base.yaml
control:
  mode: shared_flow_control
  tau: 8
```

`flow_control_tau8_rotz20.yaml`:

```yaml
# Shared flow control, tau = 8, operator command rotated 20 degrees about z.
extends: base.yaml
control:
  mode: shared_flow_control
  tau: 8
  corruption: {rotation_z_deg: 20}
```

`reverse_flow_full.yaml`:

```yaml
# Reverse flow steering, full reversal to noise, clean operator command.
extends: base.yaml
control:
  mode: shared_reverse_flow_steering
  n_reverse_steps: null
```

`reverse_flow_full_rotz20.yaml`:

```yaml
# Reverse flow steering, full reversal, command rotated 20 degrees about z.
# The reversal adapter applies the same rotation to the velocity's translation
# block and freezes orientation and gripper during the reversal.
extends: base.yaml
control:
  mode: shared_reverse_flow_steering
  n_reverse_steps: null
  corruption: {rotation_z_deg: 20}
  reversal_adapter: {translation: corruption, orientation: zero, gripper: zero}
```

`reverse_flow_5steps_rotz20.yaml`:

```yaml
# Reverse flow steering stopped after 5 of 10 steps (t = 0.5), command rotated
# 20 degrees about z, same adapter as reverse_flow_full_rotz20.
extends: base.yaml
control:
  mode: shared_reverse_flow_steering
  n_reverse_steps: 5
  corruption: {rotation_z_deg: 20}
  reversal_adapter: {translation: corruption, orientation: zero, gripper: zero}
```

- [ ] **Step 2: Write the failing tests**

`tests/examples/pi05_libero_shared_autonomy/test_config.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import yaml
from config import (
    CONFIG_DIR,
    DEFAULT_INTERACTIVE_CONFIG,
    MODES,
    ExperimentSettings,
    SessionSettings,
    deep_merge,
    flatten,
    load_experiment_settings,
    load_interactive_settings,
    load_yaml_with_extends,
    spec_label,
)

from lerobot.policies.pi05.steering import rotation_about_z

CONDITIONS = sorted(p.name for p in (CONFIG_DIR / "experiment").glob("*.yaml") if p.name != "base.yaml")


def test_deep_merge_overlay_wins_and_null_resets():
    base = {"control": {"mode": "policy", "tau": 5}, "prompt": "x"}
    merged = deep_merge(base, {"control": {"tau": None, "n_reverse_steps": 3}})
    assert merged == {"control": {"mode": "policy", "tau": None, "n_reverse_steps": 3}, "prompt": "x"}
    assert base["control"]["tau"] == 5  # not mutated


def test_flatten_rejects_unknown_keys_and_skips_null():
    schema = {"control": {"mode": "mode"}, "prompt": "prompt"}
    assert flatten({"control": {"mode": "teleop"}, "prompt": None}, schema, "f") == {"mode": "teleop"}
    with pytest.raises(ValueError, match="unknown key 'controls'"):
        flatten({"controls": {}}, schema, "f")
    with pytest.raises(ValueError, match="control.tau"):
        flatten({"control": {"tau": 1}}, schema, "f")
    with pytest.raises(ValueError, match="must be a mapping"):
        flatten({"control": 3}, schema, "f")


def test_extends_resolves_relative_to_the_file(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: {c: 2, d: 3}\n")
    (tmp_path / "cond.yaml").write_text("extends: base.yaml\nb: {d: 4}\n")
    assert load_yaml_with_extends(tmp_path / "cond.yaml") == {"a": 1, "b": {"c": 2, "d": 4}}
    (tmp_path / "loop.yaml").write_text("extends: loop.yaml\n")
    with pytest.raises(ValueError, match="extends"):
        load_yaml_with_extends(tmp_path / "loop.yaml")


def test_spec_label():
    assert spec_label(None) is None
    assert spec_label({"rotation_z_deg": 20}) == "rotation_z_deg=20"
    assert spec_label({"translation": "corruption", "orientation": "zero", "gripper": "zero"}) == (
        "translation=corruption, orientation=zero, gripper=zero"
    )
    assert spec_label({"M": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}) == "M=literal"


def test_shipped_interactive_config_loads():
    s = load_interactive_settings(DEFAULT_INTERACTIVE_CONFIG, {})
    assert isinstance(s, SessionSettings)
    assert s.control.mode == "policy" and s.control.corruption_matrix is None
    assert s.output_dir == Path("outputs/pi05_libero_interactive")


def test_interactive_cli_overrides_win_over_file():
    s = load_interactive_settings(DEFAULT_INTERACTIVE_CONFIG, {"task_id": 3, "mode": "teleop"})
    assert s.task_id == 3 and s.control.mode == "teleop"
    s = load_interactive_settings(None, {})
    assert s.suite == "libero_spatial"


@pytest.mark.parametrize("name", CONDITIONS)
def test_every_shipped_condition_loads(name):
    s = load_experiment_settings(CONFIG_DIR / "experiment" / name, {})
    assert isinstance(s, ExperimentSettings)
    assert s.name == Path(name).stem
    assert s.session.control.mode in MODES
    assert s.session.suite == "libero_goal" and s.seed == 0 and s.prompt == "do something"


def test_condition_matrices_resolve():
    s = load_experiment_settings(CONFIG_DIR / "experiment" / "reverse_flow_5steps_rotz20.yaml", {})
    c = s.session.control
    assert c.n_reverse_steps == 5
    np.testing.assert_allclose(c.corruption_matrix, rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[:3, :3], rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[3:, :], 0)
    assert spec_label(c.corruption) == "rotation_z_deg=20"
    clean = load_experiment_settings(CONFIG_DIR / "experiment" / "flow_control_tau8.yaml", {})
    assert clean.session.control.tau == 8 and clean.session.control.corruption_matrix is None


def test_experiment_validation_errors(tmp_path):
    def write(text):
        p = tmp_path / "c.yaml"
        p.write_text(text)
        return p

    with pytest.raises(ValueError, match="n_trials"):
        load_experiment_settings(write("experiment: {n_trials: 0}"), {})
    with pytest.raises(ValueError, match="control.mode"):
        load_experiment_settings(write("control: {mode: nope}"), {})
    with pytest.raises(ValueError, match="task_order"):
        load_experiment_settings(write("experiment: {task_order: backwards}"), {})
    with pytest.raises(ValueError, match="prompt"):
        load_experiment_settings(write("prompt: ''"), {})
    with pytest.raises(ValueError, match="corruption"):
        load_experiment_settings(write("control: {reversal_adapter: {translation: corruption}}"), {})
    with pytest.raises(ValueError, match="n_reverse_steps"):
        load_experiment_settings(write("control: {n_reverse_steps: 0}"), {})


def test_experiment_cli_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("prompt: task\n")
    s = load_experiment_settings(p, {"n_trials": 2, "mode": "teleop", "port": 9000, "output_dir": "o"})
    assert s.n_trials == 2 and s.session.control.mode == "teleop" and s.session.port == 9000
    assert s.output_dir == Path("o") and s.prompt == "task"


def test_base_yaml_matches_dataclass_defaults():
    data = yaml.safe_load((CONFIG_DIR / "experiment" / "base.yaml").read_text())
    assert set(data) == {"experiment", "scene", "prompt", "policy", "control", "server"}
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run --no-sync pytest tests/examples/pi05_libero_shared_autonomy/test_config.py -q 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 4: Write `config.py`**

```python
"""YAML settings for the LIBERO shared-autonomy runners.

Both entry points read the same nested YAML shape (policy / scene / control /
server) into dataclasses. A file may say `extends: other.yaml` (relative to
itself) to overlay on another; a key set to null keeps the built-in default,
and unknown keys are rejected. Matrix specs under `control` are resolved here
into arrays (see `lerobot.policies.pi05.steering` for the accepted forms).
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from teleop import build_corruption

from lerobot.policies.pi05.steering import build_reversal_adapter

MODES = ("policy", "shared_override", "shared_flow_control", "shared_reverse_flow_steering", "teleop")
TASK_ORDERS = ("random", "shuffled", "sequential")
PROMPT_FROM_TASK = "task"  # `prompt: task` gives the VLA the scene's own instruction

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_INTERACTIVE_CONFIG = CONFIG_DIR / "interactive.yaml"


@dataclass
class ControlSettings:
    mode: str = "policy"
    tau: int = 5
    n_reverse_steps: int | None = None
    input_noise: float = 0.0
    max_steps: int | None = None
    corruption: dict | list | None = None  # spec as written
    reversal_adapter: dict | None = None  # spec as written
    corruption_matrix: np.ndarray | None = field(default=None, repr=False)
    reversal_adapter_matrix: np.ndarray | None = field(default=None, repr=False)


@dataclass
class SessionSettings:
    policy_path: str = "lerobot/pi05_libero_finetuned"
    n_action_steps: int = 10
    compile: bool = False
    suite: str = "libero_spatial"
    task_id: int = 0
    port: int = 8765
    output_dir: Path = Path("outputs/pi05_libero_interactive")
    control: ControlSettings = field(default_factory=ControlSettings)


@dataclass
class ExperimentSettings:
    name: str
    session: SessionSettings
    n_trials: int = 10
    seed: int = 0
    task_order: str = "random"
    output_dir: Path = Path("outputs/pi05_libero_experiments")
    task_ids: list[int] | None = None  # None = every task in the suite
    prompt: str = "do something"


_CONTROL_SCHEMA = {
    "mode": "mode",
    "tau": "tau",
    "n_reverse_steps": "n_reverse_steps",
    "input_noise": "input_noise",
    "max_steps": "max_steps",
    "corruption": "corruption",
    "reversal_adapter": "reversal_adapter",
}
_POLICY_SCHEMA = {"path": "policy_path", "n_action_steps": "n_action_steps", "compile": "compile"}

INTERACTIVE_SCHEMA = {
    "policy": _POLICY_SCHEMA,
    "scene": {"suite": "suite", "task_id": "task_id"},
    "control": _CONTROL_SCHEMA,
    "server": {"port": "port"},
    "output_dir": "output_dir",
}

EXPERIMENT_SCHEMA = {
    "experiment": {"n_trials": "n_trials", "seed": "seed", "task_order": "task_order", "output_dir": "output_dir"},
    "scene": {"suite": "suite", "task_ids": "task_ids"},
    "prompt": "prompt",
    "policy": _POLICY_SCHEMA,
    "control": _CONTROL_SCHEMA,
    "server": {"port": "port"},
}


# ---------------------------------------------------------------- YAML mechanics


def read_yaml(path: str | Path) -> dict:
    """Read a YAML mapping (empty file -> {})."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def deep_merge(base: dict, overlay: dict) -> dict:
    """Return base updated by overlay, recursing into nested mappings. Neither input is mutated."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_with_extends(path: str | Path, _seen: tuple = ()) -> dict:
    """Read a YAML file, overlaying it on the file named by its `extends` key (relative to it)."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"{path}: circular extends chain")
    data = read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ValueError(f"{path}: extends must be a file name, got {parent!r}")
    base = load_yaml_with_extends(path.parent / parent, (*_seen, path))
    return deep_merge(base, data)


def flatten(data: dict, schema: dict, where: str) -> dict:
    """Map a nested config dict onto flat setting names, rejecting unknown keys and skipping nulls."""
    flat = {}
    for key, value in data.items():
        if key not in schema:
            raise ValueError(f"{where}: unknown key '{key}' (expected one of {', '.join(schema)})")
        spec = schema[key]
        if isinstance(spec, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{where}: '{key}' must be a mapping of {', '.join(spec)}")
            for sub, sub_value in value.items():
                if sub not in spec:
                    raise ValueError(f"{where}: unknown key '{key}.{sub}' (expected one of {', '.join(spec)})")
                if sub_value is not None:
                    flat[spec[sub]] = sub_value
        elif value is not None:
            flat[spec] = value
    return flat


def spec_label(spec) -> str | None:
    """Short display form of a matrix spec, e.g. 'rotation_z_deg=20' (None for no spec)."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        parts = []
        for key, value in spec.items():
            parts.append(f"{key}=literal" if isinstance(value, list) else f"{key}={value}")
        return ", ".join(parts)
    return "literal"


# ---------------------------------------------------------------- validation


def _fail(where: str, message: str):
    raise ValueError(f"{where}: {message}")


def build_control(flat: dict, where: str) -> ControlSettings:
    """ControlSettings from flat settings, with ranges checked and matrix specs resolved."""
    control = ControlSettings(**{k: flat[k] for k in ControlSettings.__dataclass_fields__ if k in flat})
    if control.mode not in MODES:
        _fail(where, f"control.mode must be one of {', '.join(MODES)}, got {control.mode!r}")
    if not isinstance(control.tau, int) or control.tau < 0:
        _fail(where, "control.tau must be a non-negative integer")
    if control.n_reverse_steps is not None and (
        not isinstance(control.n_reverse_steps, int) or control.n_reverse_steps < 1
    ):
        _fail(where, "control.n_reverse_steps must be a positive integer or null (null = all the way to noise)")
    if not isinstance(control.input_noise, int | float) or control.input_noise < 0:
        _fail(where, "control.input_noise must be >= 0")
    if control.max_steps is not None and (not isinstance(control.max_steps, int) or control.max_steps < 1):
        _fail(where, "control.max_steps must be a positive integer or null")
    control.corruption_matrix = build_corruption(control.corruption, f"{where}: control.corruption")
    control.reversal_adapter_matrix = build_reversal_adapter(
        control.reversal_adapter, control.corruption_matrix, f"{where}: control.reversal_adapter"
    )
    return control


def _session_from_flat(flat: dict, where: str, defaults: SessionSettings) -> SessionSettings:
    session = SessionSettings(
        policy_path=flat.get("policy_path", defaults.policy_path),
        n_action_steps=flat.get("n_action_steps", defaults.n_action_steps),
        compile=flat.get("compile", defaults.compile),
        suite=flat.get("suite", defaults.suite),
        task_id=flat.get("task_id", defaults.task_id),
        port=flat.get("port", defaults.port),
        output_dir=Path(flat.get("output_dir", defaults.output_dir)),
        control=build_control(flat, where),
    )
    if not isinstance(session.n_action_steps, int) or session.n_action_steps < 1:
        _fail(where, "policy.n_action_steps must be a positive integer")
    return session


def load_interactive_settings(path: str | Path | None, overrides: dict) -> SessionSettings:
    """Settings for interactive.py: file (optional) then CLI overrides (flat names, None = unset)."""
    where = str(path) if path is not None else "command line"
    flat = flatten(load_yaml_with_extends(path), INTERACTIVE_SCHEMA, where) if path is not None else {}
    flat.update({k: v for k, v in overrides.items() if v is not None})
    return _session_from_flat(flat, where, SessionSettings())


def load_experiment_settings(path: str | Path, overrides: dict) -> ExperimentSettings:
    """Settings for experiment.py: the condition file (with extends) then CLI overrides."""
    path = Path(path)
    where = str(path)
    flat = flatten(load_yaml_with_extends(path), EXPERIMENT_SCHEMA, where)
    flat.update({k: v for k, v in overrides.items() if v is not None})
    defaults = ExperimentSettings(name=path.stem, session=SessionSettings())
    session_defaults = SessionSettings(output_dir=defaults.output_dir)
    session_flat = {k: v for k, v in flat.items() if k not in ("n_trials", "seed", "task_order", "task_ids", "prompt")}
    settings = ExperimentSettings(
        name=path.stem,
        session=_session_from_flat(session_flat, where, session_defaults),
        n_trials=flat.get("n_trials", defaults.n_trials),
        seed=flat.get("seed", defaults.seed),
        task_order=flat.get("task_order", defaults.task_order),
        output_dir=Path(flat.get("output_dir", defaults.output_dir)),
        task_ids=flat.get("task_ids", defaults.task_ids),
        prompt=flat.get("prompt", defaults.prompt),
    )
    if not isinstance(settings.n_trials, int) or settings.n_trials < 1:
        _fail(where, f"experiment.n_trials must be a positive integer, got {settings.n_trials!r}")
    if settings.task_order not in TASK_ORDERS:
        _fail(where, f"experiment.task_order must be one of {', '.join(TASK_ORDERS)}, got {settings.task_order!r}")
    if not isinstance(settings.prompt, str) or not settings.prompt.strip():
        _fail(where, "prompt must be a non-empty string (or `task` to use the scene's own instruction)")
    if settings.task_ids is not None and (
        not isinstance(settings.task_ids, list)
        or not settings.task_ids
        or not all(isinstance(t, int) for t in settings.task_ids)
    ):
        _fail(where, "scene.task_ids must be a non-empty list of task ids (or null for all of them)")
    return settings
```

Note: `output_dir` appears in both schemas under different YAML keys (`output_dir` at top level for interactive, `experiment.output_dir` for experiments) and maps to one flat name; for experiments it lands on `ExperimentSettings.output_dir` and the session's own `output_dir` is unused. The task-id range check against the suite needs LIBERO and stays in `experiment.py` (Task 9).

- [ ] **Step 5: Run the tests**

```bash
uv run --no-sync pytest tests/examples/pi05_libero_shared_autonomy -q 2>&1 | tail -3
uv run --no-sync ruff check examples/pi05/libero_shared_autonomy/config.py tests/examples && uv run --no-sync ruff format examples/pi05/libero_shared_autonomy/config.py tests/examples
```

Expected: all pass, ruff clean. The `input_noise` check uses `int | float` in an `isinstance` call, which is valid on 3.12.

- [ ] **Step 6: Commit**

```bash
git add examples/pi05/libero_shared_autonomy/config.py examples/pi05/libero_shared_autonomy/configs tests/examples
git commit -m "Add config.py and base+overlay experiment configs with inline matrix specs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 7: `live_view.py`: the browser window

**Files:**
- Create: `examples/pi05/libero_shared_autonomy/live_view.py`
- Modify: `interactive.py`, `experiment.py` (use `LiveView`)

**Interfaces:**
- Produces (module `live_view`):
  - `ACTION_LABELS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]`
  - `FrameStream` (unchanged: `publish(rgb)`, `wait_frame(last_seq, timeout)`, `set_status(**kw)`, `get_status()`)
  - `class LiveView(port: int, keyboard: KeyboardReader, status_extra: Callable[[], dict])` with `stream: FrameStream`, `url: str`, `start()`, `close()`

- [ ] **Step 1: Create `live_view.py`**

Move from `interactive.py`, verbatim: `ACTION_LABELS`, `PAGE`, `FrameStream`. Then replace `make_handler` with:

```python
class LiveView:
    """Serves the page, the MJPEG stream, a status JSON, and takes keyboard events.

    `status_extra()` is called on every /status request and merged into the
    stream's status dict, so the view needs no knowledge of the teleop chain
    beyond the keyboard reader it feeds.
    """

    def __init__(self, port: int, keyboard: KeyboardReader, status_extra: Callable[[], dict]):
        self.port = port
        self.stream = FrameStream()
        self._keyboard = keyboard
        self._status_extra = status_extra
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def _handler(self):
        stream, keyboard, status_extra = self.stream, self._keyboard, self._status_extra

        class Handler(BaseHTTPRequestHandler):
            ...  # the body of the old make_handler's Handler, with the /status branch below

        return Handler
```

In the `/status` branch of `do_GET`, replace the five `status[...] = ...` lines with:

```python
                status = stream.get_status()
                status["keys"] = sorted(keyboard.held)
                status["keyboard_gripper"] = keyboard.gripper
                status.update(status_extra())
```

Module docstring: `"""The operator's browser window: live frames, status, and keyboard capture."""`. Imports: `io`, `json`, `threading`, `time` (as used by `FrameStream`), `Callable` from `collections.abc`, `BaseHTTPRequestHandler`, `ThreadingHTTPServer`, `numpy`, `PIL.Image`, and `from teleop import KeyboardReader`.

- [ ] **Step 2: Use it from the entry points**

In `interactive.py`, replace `stream = FrameStream()` ... `threading.Thread(...).start()` and the print of the URL with:

```python
    keyboard = KeyboardReader()
    chain = TeleopChain(keyboard, input_noise=args.input_noise)
    corruption, noisy, reader = chain.corruption, chain.noisy, chain.reader
    flow_adapter = FlowAdapter()
    view = LiveView(
        args.port,
        keyboard,
        lambda: {
            "input_noise": noisy.std,
            "corruption": corruption.label if corruption.matrix is not None else None,
            "flow_adapter": flow_adapter.label if flow_adapter.matrix is not None else None,
        },
    )
    view.start()
    stream = view.stream
    print(f"\nLive view: {view.url}  (VSCode should auto-forward the port)\n")
```

keeping the two corruption/adapter loading blocks between the chain and the view where they were. Replace `server.shutdown()` in the `finally` with `view.close()`. Delete the moved definitions and the now-unused imports (`io`, `BaseHTTPRequestHandler`, `ThreadingHTTPServer`, `Image`, `json` if unused). Import `from live_view import ACTION_LABELS, LiveView`.

Do the same in `experiment.py` (it imports `FrameStream`, `make_handler` from `interactive`; switch both to `LiveView`).

- [ ] **Step 3: Verify**

```bash
uv run --no-sync python -c "import sys; sys.path.insert(0, 'examples/pi05/libero_shared_autonomy'); import interactive, experiment, live_view; print('ok')"
uv run --no-sync ruff check examples/pi05/libero_shared_autonomy && uv run --no-sync ruff format examples/pi05/libero_shared_autonomy
uv run --no-sync python - <<'EOF'
import sys, urllib.request, json
sys.path.insert(0, "examples/pi05/libero_shared_autonomy")
from live_view import LiveView
from teleop import KeyboardReader
kb = KeyboardReader()
view = LiveView(8799, kb, lambda: {"extra": 1})
view.start()
view.stream.set_status(prompt="p", step=0)
req = urllib.request.Request(view.url + "/keys", data=json.dumps({"held": ["ArrowUp"], "toggles": 1}).encode(), headers={"Content-Type": "application/json"})
assert urllib.request.urlopen(req).status == 204
status = json.load(urllib.request.urlopen(view.url + "/status"))
assert status["extra"] == 1 and status["keys"] == ["ArrowUp"] and status["keyboard_gripper"] == 1.0, status
assert b"LIBERO interactive" in urllib.request.urlopen(view.url).read()
view.close()
print("live view ok")
EOF
```

Expected: `ok`, ruff clean, `live view ok`.

- [ ] **Step 4: Commit**

```bash
git add -A examples/pi05/libero_shared_autonomy
git commit -m "Extract the browser live view into live_view.py

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `session.py` and the rewritten `interactive.py`

**Files:**
- Create: `examples/pi05/libero_shared_autonomy/session.py`
- Rewrite: `examples/pi05/libero_shared_autonomy/interactive.py`
- Delete: `configs`' predecessor `config_interactive.yaml`, `deterministic_corruption.yaml`, `flow_reversal_adapter.yaml`

After this task `experiment.py` is broken (it still imports from the old `interactive`); Task 9 rewrites it. Nothing under `tests/` imports `experiment` yet, so the test suite stays green.

**Interfaces:**
- Consumes: `config.SessionSettings`, `teleop.TeleopChain`, `live_view.LiveView`, `steering.*`
- Produces (module `session`):
  - `VIDEO_FPS = 30`, `RATE_HZ = 20`, `TELEOP_MAX_STEPS = 900`
  - `@dataclass RolloutResult(success: bool, steps: int, frames: list[np.ndarray], metrics: dict)`
  - `class Session(settings: SessionSettings)` with attributes `settings`, `policy`, `policy_cfg`, `chain`, `adapter: ReversalAdapter`, `view: LiveView`, `suite`, `task_id`, `mode`, `tau`, `n_reverse_steps`, `vec_env`; methods `list_tasks(suite) -> list[str]` (static), `task_description` (property), `set_scene(suite, task_id)`, `show_scene()`, `set_mode(mode, tau=None, n_reverse_steps=None)`, `set_corruption(matrix, label)`, `set_reversal_adapter(matrix, label)`, `announce_mode() -> str`, `mode_label() -> str`, `rollout(prompt, recorder=None, max_steps=None) -> RolloutResult`, `metrics() -> dict`, `stats_line() -> str`, `resolved_matrices() -> dict`, `close()`

- [ ] **Step 1: Write `session.py`**

```python
"""A loaded policy, a LIBERO scene, the operator's input and the live view, in one object.

Both entry points drive a `Session`: the REPL changes its scene and mode
between rollouts, the experiment driver walks a schedule of scenes. `rollout`
is the step loop of `lerobot_eval.rollout()` with the user's prompt in place
of the env's task description, an optional action hook (teleop modes) and an
optional per-step recorder (experiments).
"""

import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
from config import MODES, SessionSettings, spec_label
from live_view import ACTION_LABELS, LiveView
from teleop import KeyboardReader, TeleopChain

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.steering import (
    DEADBAND,
    FlowControlPolicy,
    ReversalAdapter,
    ReverseFlowSteeringPolicy,
)
from lerobot.utils.constants import ACTION

VIDEO_FPS = 30
RATE_HZ = 20  # LIBERO control rate; caps the loop so teleop feels real-time
# Suite defaults (~14 s) are too short for manual driving. Must stay below
# robosuite's horizon of 1000 (which includes the 10 settle steps at reset):
# LIBERO's step() replaces the horizon `done` with task success, so exceeding
# the horizon raises "executing action in terminated episode" instead of ending.
TELEOP_MAX_STEPS = 900


@dataclass
class RolloutResult:
    success: bool
    steps: int
    frames: list[np.ndarray] = field(repr=False)
    metrics: dict


class _ZeroPolicy:
    """Stand-in policy for teleop mode: outputs zeros, the teleop hook fills them in."""

    def reset(self) -> None:
        pass

    def select_action(self, obs) -> torch.Tensor:
        return torch.zeros(1, len(ACTION_LABELS))


def _identity(x):
    return x


def _teleop_hook(reader, paste_gripper: bool) -> Callable[[np.ndarray], np.ndarray]:
    """Paste the teleop translation (and, for teleop mode, gripper) into the env action."""

    def hook(action: np.ndarray) -> np.ndarray:
        cmd = reader.translation
        if paste_gripper or np.max(np.abs(cmd)) >= DEADBAND:
            action[0, :3] = cmd
        if paste_gripper:
            action[0, 6] = reader.gripper
        return action

    return hook


class Session:
    def __init__(self, settings: SessionSettings):
        self.settings = settings
        control = settings.control

        self.keyboard = KeyboardReader()
        self.chain = TeleopChain(self.keyboard, input_noise=control.input_noise)
        self.adapter = ReversalAdapter()
        self.set_corruption(control.corruption_matrix, spec_label(control.corruption))
        self.set_reversal_adapter(control.reversal_adapter_matrix, spec_label(control.reversal_adapter))
        self.view = LiveView(settings.port, self.keyboard, self._status_extra)
        self.view.start()

        logging.info(f"Loading policy {settings.policy_path} ...")
        self.policy_cfg = PreTrainedConfig.from_pretrained(settings.policy_path)
        self.policy_cfg.pretrained_path = settings.policy_path
        self.policy_cfg.n_action_steps = settings.n_action_steps
        if hasattr(self.policy_cfg, "compile_model"):
            self.policy_cfg.compile_model = settings.compile

        self.suite = self.task_id = None
        self.envs_dict = self.vec_env = None
        self.env_cfg, self.envs_dict, self.vec_env = self._build_env(settings.suite, settings.task_id)
        self.suite, self.task_id = settings.suite, settings.task_id

        self.policy = make_policy(cfg=self.policy_cfg, env_cfg=self.env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=str(self.policy_cfg.pretrained_path),
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=self.policy_cfg
        )
        self.autocast_ctx = (
            torch.autocast(device_type=torch.device(self.policy.config.device).type)
            if self.policy_cfg.use_amp
            else nullcontext()
        )

        self._zero_policy = _ZeroPolicy()
        self._flow_policy: FlowControlPolicy | None = None
        self._frs_policy: ReverseFlowSteeringPolicy | None = None
        self.mode = "policy"
        self.tau = control.tau
        self.n_reverse_steps = control.n_reverse_steps
        self.set_mode(control.mode, control.tau, control.n_reverse_steps)

    @classmethod
    def from_settings(cls, settings: SessionSettings) -> "Session":
        return cls(settings)

    # ------------------------------------------------------------ scene

    @staticmethod
    def list_tasks(suite: str) -> list[str]:
        from libero.libero import benchmark

        bench = benchmark.get_benchmark_dict()
        if suite not in bench:
            raise ValueError(f"Unknown suite '{suite}'. Available: {', '.join(sorted(bench))}")
        return [t.language for t in bench[suite]().tasks]

    @staticmethod
    def _build_env(suite: str, task_id: int):
        """Build a 1-env SyncVectorEnv for one LIBERO scene, mirroring eval.sh settings."""
        env_cfg = LiberoEnvConfig(task=suite, task_ids=[task_id])
        envs_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
        vec_env = envs_dict[suite][task_id]
        check_env_attributes_and_types(vec_env)
        return env_cfg, envs_dict, vec_env

    @property
    def task_description(self) -> str:
        return self.vec_env.envs[0].task_description

    def set_scene(self, suite: str, task_id: int) -> None:
        """Switch scene; raises ValueError / FileNotFoundError and keeps the old scene on failure."""
        env_cfg, envs_dict, vec_env = self._build_env(suite, task_id)
        close_envs(self.envs_dict)
        self.env_cfg, self.envs_dict, self.vec_env = env_cfg, envs_dict, vec_env
        self.suite, self.task_id = suite, task_id
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.policy_cfg
        )

    def show_scene(self) -> None:
        """Reset the env and put its first frame on the live view."""
        self.vec_env.reset()
        self.view.stream.publish(self.vec_env.envs[0].render())

    # ------------------------------------------------------------ operator perturbations

    def set_corruption(self, matrix, label: str | None) -> None:
        self.chain.corruption.matrix = matrix
        self.chain.corruption.label = label

    def set_reversal_adapter(self, matrix, label: str | None) -> None:
        self.adapter.matrix = matrix
        self.adapter.label = label

    def _status_extra(self) -> dict:
        corruption, adapter = self.chain.corruption, self.adapter
        return {
            "input_noise": self.chain.noisy.std,
            "corruption": corruption.label if corruption.matrix is not None else None,
            "flow_adapter": adapter.label if adapter.matrix is not None else None,
        }

    def resolved_matrices(self) -> dict:
        """For run provenance: the matrices in force, as lists (None when off)."""
        m, f = self.chain.corruption.matrix, self.adapter.matrix
        return {
            "corruption_matrix": None if m is None else m.tolist(),
            "flow_adapter_matrix": None if f is None else f.tolist(),
        }

    # ------------------------------------------------------------ mode

    def set_mode(self, mode: str, tau: int | None = None, n_reverse_steps: int | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}' (expected one of {', '.join(MODES)})")
        total = self.policy.config.num_inference_steps
        if tau is not None:
            if not isinstance(tau, int) or not 0 <= tau <= total:
                raise ValueError(f"tau must be an integer in [0, {total}] (denoising steps)")
            self.tau = tau
        if n_reverse_steps is not None:
            if not isinstance(n_reverse_steps, int) or not 1 <= n_reverse_steps <= total:
                raise ValueError(f"n_reverse_steps must be an integer in [1, {total}]")
            self.n_reverse_steps = n_reverse_steps
        self.mode = mode
        if mode != "policy":
            self.chain.attach_spacemouse()

    def mode_label(self) -> str:
        """Short label for the terminal and the live view."""
        if self.mode == "shared_flow_control":
            return f"{self.mode} tau={self.tau}"
        if self.mode == "shared_reverse_flow_steering":
            return f"{self.mode} n={self.n_reverse_steps or self.policy.config.num_inference_steps}"
        return self.mode

    def announce_mode(self) -> str:
        """What the current mode does, for the terminal."""
        lines = []
        if self.mode != "policy":
            lines.append(
                "Keyboard (click the live view first): arrows = x/y, PgUp/PgDn or W/S = z, "
                "Space = gripper, hold Shift = full speed."
            )
        total = self.policy.config.num_inference_steps
        if self.mode == "teleop":
            lines.append("Teleop: you drive x/y/z and the gripper; the model is not involved.")
            lines.append("Press Enter to start a rollout.")
        elif self.mode == "shared_override":
            lines.append("Shared override: pi0.5 drives, your x/y/z replaces its translation.")
        elif self.mode == "shared_flow_control":
            lines.append(
                f"Shared flow control: your x/y/z steers the first tau={self.tau} of "
                f"{total} denoising steps (idle input = pure policy)."
            )
        elif self.mode == "shared_reverse_flow_steering":
            n = total if self.n_reverse_steps is None else self.n_reverse_steps
            if n >= total:
                lines.append(
                    "Shared reverse flow steering: your x/y/z defines a reference chunk that is "
                    f"inverted through the flow ({total} reverse steps) to its noise; pi0.5 then "
                    "denoises from that noise (idle input = pure policy)."
                )
            else:
                lines.append(
                    "Shared reverse flow steering: your x/y/z defines a reference chunk that is "
                    f"inverted {n} of {total} steps (partway to noise, t={n / total:.1f}); pi0.5 then "
                    f"denoises from there in {total - n} steps (idle input = pure policy)."
                )
            if self.adapter.matrix is not None:
                lines.append(self.adapter.describe())
        return "\n".join(lines)

    def _rollout_kwargs(self) -> dict:
        reader = self.chain.reader
        if self.mode == "teleop":
            return {
                "policy": self._zero_policy,
                "env_preprocessor": _identity,
                "env_postprocessor": _identity,
                "preprocessor": _identity,
                "postprocessor": _identity,
                "action_hook": _teleop_hook(reader, paste_gripper=True),
                "max_steps": TELEOP_MAX_STEPS,
            }
        kwargs = {
            "policy": self.policy,
            "env_preprocessor": self.env_preprocessor,
            "env_postprocessor": self.env_postprocessor,
            "preprocessor": self.preprocessor,
            "postprocessor": self.postprocessor,
            "action_hook": None,
            "max_steps": None,
        }
        if self.mode == "shared_override":
            kwargs["action_hook"] = _teleop_hook(reader, paste_gripper=False)
        elif self.mode == "shared_flow_control":
            if self._flow_policy is None:
                self._flow_policy = FlowControlPolicy(self.policy, reader, self.tau, self.postprocessor)
            self._flow_policy.tau = self.tau
            kwargs["policy"] = self._flow_policy
        elif self.mode == "shared_reverse_flow_steering":
            if self._frs_policy is None:
                self._frs_policy = ReverseFlowSteeringPolicy(
                    self.policy, reader, self.postprocessor, adapter=self.adapter
                )
            self._frs_policy.n_reverse_steps = self.n_reverse_steps
            kwargs["policy"] = self._frs_policy
        return kwargs

    def metrics(self) -> dict:
        """Steering statistics of the last rollout in the current mode."""
        if self.mode == "shared_flow_control" and self._flow_policy is not None:
            return {"guided_denoising_steps": int(self._flow_policy.hook_calls)}
        if self.mode == "shared_reverse_flow_steering" and self._frs_policy is not None:
            errors = self._frs_policy.reconstruction_errors
            total = self.policy.config.num_inference_steps
            return {
                "n_reverse_steps": self._frs_policy.n_reverse_steps or total,
                "steered_chunks": int(self._frs_policy.steered_chunks),
                "reconstruction_error_mean": float(np.mean(errors)) if errors else None,
            }
        return {}

    def stats_line(self) -> str:
        metrics = self.metrics()
        if self.mode == "shared_flow_control" and metrics:
            return f"flow guidance applied on {metrics['guided_denoising_steps']} denoising steps"
        if self.mode == "shared_reverse_flow_steering" and metrics:
            error = metrics["reconstruction_error_mean"]
            detail = f" (mean |executed - reference| translation: {error:.2f} std)" if error else ""
            return f"reverse flow steering applied on {metrics['steered_chunks']} chunks{detail}"
        return ""

    # ------------------------------------------------------------ rollout

    def rollout(self, prompt: str, recorder=None, max_steps: int | None = None) -> RolloutResult:
        """One rollout in the current scene and mode, driven by `prompt`.

        `recorder(step=, observation=, action=, reward=, terminated=, truncated=, info=)`
        is called once per step with the observation the policy saw and the executed
        action. `max_steps` overrides the suite's episode length (teleop mode has its
        own default).
        """
        k = self._rollout_kwargs()
        policy, action_hook = k["policy"], k["action_hook"]
        preprocessor, postprocessor = k["preprocessor"], k["postprocessor"]
        env_preprocessor, env_postprocessor = k["env_preprocessor"], k["env_postprocessor"]
        stream = self.view.stream
        vec_env = self.vec_env

        policy.reset()
        observation, _ = vec_env.reset()
        max_steps = max_steps or k["max_steps"] or int(vec_env.call("_max_episode_steps")[0])
        frames = [vec_env.envs[0].render()]
        stream.publish(frames[-1])
        stream.set_status(prompt=prompt, step=0, max_steps=max_steps, state="running", action=None, mode=self.mode_label())

        success = False
        step = 0
        while step < max_steps:
            step_start = time.time()
            obs = preprocess_observation(observation)
            obs["task"] = [prompt]
            obs = env_preprocessor(obs)
            obs = preprocessor(obs)
            with torch.inference_mode(), self.autocast_ctx:
                action = policy.select_action(obs)
            action = postprocessor(action)
            transition = env_postprocessor({ACTION: action})
            action_numpy = transition[ACTION].to("cpu").numpy()
            if action_hook is not None:
                action_numpy = action_hook(action_numpy)

            previous_observation = observation
            observation, reward, terminated, truncated, info = vec_env.step(action_numpy)
            step += 1
            if recorder is not None:
                recorder(
                    step=step,
                    observation=previous_observation,
                    action=action_numpy,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            stream.set_status(step=step, action=action_numpy[0].tolist())

            if "final_info" in info:
                success = bool(info["final_info"]["is_success"][0])
            elif "is_success" in info:
                is_success = info["is_success"]
                success = bool(is_success[0] if hasattr(is_success, "__len__") else is_success)

            if terminated[0] or truncated[0]:
                # The sub-env auto-resets on termination, so render() would show a
                # fresh scene; use the final observation's agentview pixels instead
                # (flipped to match render() orientation).
                final = observation["pixels"]["image"][0][::-1, ::-1]
                frames.append(final)
                stream.publish(final)
                break

            frames.append(vec_env.envs[0].render())
            stream.publish(frames[-1])
            print(f"\r  step {step}/{max_steps}", end="", flush=True)

            leftover = 1.0 / RATE_HZ - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)

        print()
        stream.set_status(state="success" if success else "failed")
        return RolloutResult(success=success, steps=step, frames=frames, metrics=self.metrics())

    def close(self) -> None:
        close_envs(self.envs_dict)
        self.view.close()
```

- [ ] **Step 2: Rewrite `interactive.py`**

```python
#!/usr/bin/env python
"""Interactive LIBERO runner: type a text prompt, watch the policy act on it.

Loads the pi0.5 policy once, then loops: read an instruction from the terminal,
run one rollout in the chosen LIBERO scene feeding *your* text to the policy
(instead of the task's built-in instruction), and stream frames live to a
browser tab. An MP4 of each rollout is also saved.

Launch via ./interactive.sh (sets MUJOCO_GL/caches and runs under uv), then
open the printed URL. REPL commands:

    <any text>          run a rollout with that instruction
    <empty line>        run a rollout with the scene's built-in instruction
    tasks               list task ids + instructions for the current suite
    task <suite> <id>   switch scene (e.g. `task libero_spatial 3`)
    mode <name> [n]     policy: pi0.5 only
                        shared_override: your x/y/z replaces pi0.5's translation
                        shared_flow_control [tau]: your x/y/z steers dims 0-2 of x_t
                        for the first tau denoising steps (default 5)
                        shared_reverse_flow_steering [n_reverse_steps]: your x/y/z
                        defines a reference chunk that is integrated backward
                        through the flow to its latent noise; the forward flow
                        starts from there (Flow Reversal Steering, arXiv:2606.13675)
                        teleop: you drive x/y/z and the gripper, no model
    noise [std]         show / set the std of the Gaussian noise added to your
                        x/y/z command at every step (0 = off)
    corruption [file|off]
                        show / load / clear the command corruption M (3x3):
                        a YAML file with {rotation_z_deg: d}, {scale: [...]} or M: rows
    adapter [file|off]  show / load / clear the reversal adapter F (7x7): a YAML
                        file with translation/orientation/gripper blocks or F: rows
    quit / Ctrl-D       exit

Teleop input comes from a 3Dconnexion SpaceMouse if one is plugged in (any
button toggles the gripper) and/or from the keyboard in the browser live view:
arrows move x/y, PageUp/PageDown or W/S move z, Space toggles the gripper, hold
Shift for full speed. The SpaceMouse wins while it is deflected.
"""

import argparse
import datetime as dt
import re
import time
from pathlib import Path

from config import DEFAULT_INTERACTIVE_CONFIG, MODES, load_interactive_settings
from session import VIDEO_FPS, Session
from teleop import build_corruption, read_matrix_spec

from lerobot.policies.pi05.steering import build_reversal_adapter
from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import init_logging


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "_".join(words) or "prompt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        nargs="?",
        const=str(DEFAULT_INTERACTIVE_CONFIG),
        default=None,
        metavar="PATH",
        help="read settings from a YAML file (flags passed here still win); PATH defaults to "
        f"{DEFAULT_INTERACTIVE_CONFIG.relative_to(Path(__file__).resolve().parent)}",
    )
    parser.add_argument("--policy-path", dest="policy_path", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task-id", dest="task_id", type=int, default=None)
    parser.add_argument("--n-action-steps", dest="n_action_steps", type=int, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=MODES, help="control mode to start in")
    parser.add_argument("--tau", type=int, default=None, help="shared_flow_control: steps your input steers")
    parser.add_argument(
        "--n-reverse-steps", dest="n_reverse_steps", type=int, default=None, metavar="N",
        help="shared_reverse_flow_steering: reverse N of the denoising steps (default: all)",
    )
    parser.add_argument(
        "--corruption", default=None, metavar="FILE",
        help="YAML matrix spec applied to your x/y/z command at every step (x -> M @ x)",
    )
    parser.add_argument(
        "--reversal-adapter", dest="reversal_adapter", default=None, metavar="FILE",
        help="YAML adapter spec for shared_reverse_flow_steering's reverse integration (x_t += h * F @ v)",
    )
    parser.add_argument(
        "--input-noise", dest="input_noise", type=float, default=None, metavar="STD",
        help="std of Gaussian noise added to x, y, z of your command while you are commanding",
    )
    parser.add_argument("--compile", action="store_const", const=True, default=None, help="torch.compile the model")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="where rollout MP4s are written")
    args = parser.parse_args()
    overrides = vars(args).copy()
    overrides.pop("config")
    for key in ("corruption", "reversal_adapter"):
        if overrides[key] is not None:
            try:
                overrides[key] = read_matrix_spec(overrides[key])
            except (OSError, ValueError) as e:
                parser.error(f"--{key.replace('_', '-')}: {e}")
    try:
        settings = load_interactive_settings(args.config, overrides)
    except (OSError, ValueError) as e:
        parser.error(str(e))
    return settings


# ---------------------------------------------------------------- REPL commands


def cmd_mode(session: Session, tokens: list[str]) -> None:
    usage = "usage: mode policy|shared_override|shared_flow_control [tau]|shared_reverse_flow_steering [n]|teleop"
    if len(tokens) < 2 or tokens[1] not in MODES:
        print(usage)
        return
    mode, arg = tokens[1], (tokens[2] if len(tokens) > 2 else None)
    kwargs = {}
    if arg is not None:
        if not arg.isdigit():
            print(usage)
            return
        kwargs["tau" if mode == "shared_flow_control" else "n_reverse_steps"] = int(arg)
    try:
        session.set_mode(mode, **kwargs)
    except ValueError as e:
        print(e)
        return
    print(session.announce_mode())


def cmd_noise(session: Session, tokens: list[str]) -> None:
    if len(tokens) == 2:
        try:
            session.chain.noisy.std = float(tokens[1])
        except ValueError:
            print("usage: noise <std>   e.g. noise 0.1   (std >= 0, 0 = off)")
            return
    elif len(tokens) > 2:
        print("usage: noise <std>   e.g. noise 0.1   (std >= 0, 0 = off)")
        return
    std = session.chain.noisy.std
    print(f"Input noise: std {std:.3f} on x/y/z while you command." if std > 0 else "Input noise: off.")


def cmd_corruption(session: Session, tokens: list[str]) -> None:
    if len(tokens) == 2 and tokens[1] == "off":
        session.set_corruption(None, None)
    elif len(tokens) == 2:
        try:
            session.set_corruption(build_corruption(read_matrix_spec(tokens[1])), Path(tokens[1]).name)
        except (OSError, ValueError) as e:
            print(f"could not load corruption: {e}")
            return
    elif len(tokens) > 2:
        print("usage: corruption [<file.yaml>|off]")
        return
    print(session.chain.corruption.describe())


def cmd_adapter(session: Session, tokens: list[str]) -> None:
    if len(tokens) == 2 and tokens[1] == "off":
        session.set_reversal_adapter(None, None)
    elif len(tokens) == 2:
        try:
            spec = read_matrix_spec(tokens[1])
            matrix = build_reversal_adapter(spec, session.chain.corruption.matrix)
            session.set_reversal_adapter(matrix, Path(tokens[1]).name)
        except (OSError, ValueError) as e:
            print(f"could not load reversal adapter: {e}")
            return
    elif len(tokens) > 2:
        print("usage: adapter [<file.yaml>|off]")
        return
    print(session.adapter.describe())


def cmd_tasks(session: Session, tokens: list[str]) -> None:
    for i, lang in enumerate(session.list_tasks(session.suite)):
        print(f"  {i}: {lang}")


def cmd_task(session: Session, tokens: list[str]) -> None:
    if len(tokens) != 3 or not tokens[2].lstrip("-").isdigit():
        print("usage: task <suite> <id>   e.g. task libero_spatial 3")
        return
    try:
        session.set_scene(tokens[1], int(tokens[2]))
    except (ValueError, FileNotFoundError) as e:
        print(f"could not build env: {e}")
        return
    session.show_scene()
    print(f"Scene: {session.suite} task {session.task_id}")
    print(f'Built-in instruction: "{session.task_description}"')


COMMANDS = {
    "mode": cmd_mode,
    "noise": cmd_noise,
    "corruption": cmd_corruption,
    "adapter": cmd_adapter,
    "tasks": cmd_tasks,
    "task": cmd_task,
}


def run_prompt(session: Session, line: str, output_dir: Path) -> None:
    prompt = "manual teleop" if session.mode == "teleop" else (line or session.task_description)
    print(f'Running [{session.mode_label()}]: "{prompt}"')
    start = time.time()
    result = session.rollout(prompt)
    video_path = output_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{slugify(prompt)}.mp4"
    write_video(video_path, result.frames, fps=VIDEO_FPS)
    outcome = "SUCCESS" if result.success else "no success"
    print(f"{outcome} after {result.steps} steps ({time.time() - start:.0f}s), video: {video_path}")
    stats = session.stats_line()
    if stats:
        print(stats)
    print()


def main():
    settings = parse_args()
    init_logging()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    session = Session(settings)
    print(f"\nLive view: {session.view.url}  (VSCode should auto-forward the port)\n")
    session.show_scene()
    session.view.stream.set_status(state="idle: type a prompt in the terminal")
    print(f"Scene: {session.suite} task {session.task_id}")
    print(f'Built-in instruction: "{session.task_description}"')
    print(
        "Type an instruction (empty = built-in), `tasks`, `task <suite> <id>`, `mode <name> [n]`, "
        "`noise [std]`, `corruption [file|off]`, `adapter [file|off]`, or `quit`.\n"
    )
    if session.chain.corruption.matrix is not None:
        print(session.chain.corruption.describe() + "\n")
    if session.adapter.matrix is not None:
        print(session.adapter.describe() + "\n")
    if session.mode != "policy":
        print(session.announce_mode() + "\n")
    if session.chain.noisy.std > 0:
        print(f"Input noise: std {session.chain.noisy.std:.3f} on x/y/z while you command.\n")

    try:
        while True:
            try:
                line = input(f"{session.mode}> ").strip()
            except EOFError:
                break
            if line in ("quit", "exit"):
                break
            tokens = line.split()
            if tokens and tokens[0] in COMMANDS:
                COMMANDS[tokens[0]](session, tokens)
                continue
            run_prompt(session, line, settings.output_dir)
    finally:
        session.close()


if __name__ == "__main__":
    main()
```

Keep the Apache header comment between the shebang and the docstring as in the old file.

- [ ] **Step 3: Delete the superseded files**

```bash
cd examples/pi05/libero_shared_autonomy
git rm config_interactive.yaml deterministic_corruption.yaml flow_reversal_adapter.yaml
cd -
```

- [ ] **Step 4: Verify without a GPU**

```bash
uv run --no-sync python -c "import sys; sys.path.insert(0, 'examples/pi05/libero_shared_autonomy'); import session, interactive; print('ok')"
uv run --no-sync python examples/pi05/libero_shared_autonomy/interactive.py --help | head -5
uv run --no-sync ruff check examples/pi05/libero_shared_autonomy && uv run --no-sync ruff format examples/pi05/libero_shared_autonomy
uv run --no-sync pytest tests/examples tests/policies/pi0_pi05/test_pi05_steering.py -q 2>&1 | tail -3
```

Expected: `ok`, the docstring, ruff clean, tests pass.

- [ ] **Step 5: Smoke test on the GPU (short)**

```bash
source examples/pi05/libero_shared_autonomy/env.sh
printf 'mode shared_flow_control 3\n\nquit\n' | ./examples/pi05/libero_shared_autonomy/interactive.sh --config --suite libero_spatial --task-id 0 2>&1 | tail -15
```

Expected: the policy loads, the mode announcement prints, one rollout runs to its end (a few hundred steps; the terminal shows `step N/M`), an MP4 path is printed under `outputs/pi05_libero_interactive/`, and the process exits cleanly. If LIBERO assets are missing, run `./examples/pi05/libero_shared_autonomy/setup.sh` first.

- [ ] **Step 6: Commit**

```bash
git add -A examples/pi05/libero_shared_autonomy
git commit -m "Add Session and rewrite interactive.py as a REPL over it

Session owns the policy, scene, teleop chain, live view and control mode;
rollout() replaces run_rollout + ModeRunner. The REPL reads settings through
config.py and loads matrices from intent specs.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 9: Rewrite `experiment.py` over `Session`

**Files:**
- Rewrite: `examples/pi05/libero_shared_autonomy/experiment.py`
- Delete: the five old `config_experiment_*.yaml`
- Create: `tests/examples/pi05_libero_shared_autonomy/test_experiment.py`

**Interfaces:**
- Consumes: `config.load_experiment_settings`, `config.PROMPT_FROM_TASK`, `session.Session`, `session.RolloutResult`, `teleop.TeleopChain`
- Produces (module `experiment`): `build_schedule(task_ids, n_trials, order, seed) -> list[int]`, `TrialRecorder(chain)` with `__call__(step=, observation=, action=, reward=, terminated=, truncated=, info=)`, `rows`, `total_reads`, `save(path, **scalars)`, `prompt_for(prompt_setting, task_description) -> str`, `parse_args()`, `main()`

- [ ] **Step 1: Write the failing tests**

`tests/examples/pi05_libero_shared_autonomy/test_experiment.py`:

```python
import numpy as np
import pytest
from experiment import TrialRecorder, build_schedule, prompt_for
from teleop import KeyboardReader, TeleopChain


def test_build_schedule_orders():
    assert build_schedule([1, 2, 3], 7, "sequential", 0) == [1, 2, 3, 1, 2, 3, 1]
    shuffled = build_schedule([1, 2, 3], 7, "shuffled", 0)
    assert sorted(shuffled[:3]) == [1, 2, 3] and sorted(shuffled[3:6]) == [1, 2, 3] and len(shuffled) == 7
    random_a = build_schedule([1, 2, 3], 20, "random", 5)
    assert random_a == build_schedule([1, 2, 3], 20, "random", 5)
    assert random_a != build_schedule([1, 2, 3], 20, "random", 6)
    assert set(random_a) <= {1, 2, 3}


def test_prompt_for():
    assert prompt_for("do something", "pick up the bowl") == "do something"
    assert prompt_for("task", "pick up the bowl") == "pick up the bowl"


def test_trial_recorder_rows_and_per_trial_reads(tmp_path):
    kb = KeyboardReader(clock=lambda: 0.0)
    kb.update({"ArrowUp"}, toggles=0)
    chain = TeleopChain(kb)
    chain.reader.translation  # a read before the trial must not count
    recorder = TrialRecorder(chain)
    obs = {"robot_state": {"eef": {"pos": np.zeros((1, 3)), "quat": np.zeros((1, 4))}}}
    chain.reader.translation
    recorder(step=1, observation=obs, action=np.ones((1, 7)), reward=np.array([0.0]),
             terminated=np.array([False]), truncated=np.array([False]), info={})
    recorder(step=2, observation={}, action=np.zeros((1, 7)), reward=np.array([1.0]),
             terminated=np.array([True]), truncated=np.array([False]), info={})
    assert recorder.total_reads == 1
    assert [r["user_reads"] for r in recorder.rows] == [1, 0]
    assert np.isnan(recorder.rows[1]["eef_pos"]).all() and recorder.rows[1]["joint_pos"].shape == (7,)
    recorder.save(tmp_path / "t.npz", success=True, task_id=3)
    z = np.load(tmp_path / "t.npz")
    assert z["action"].shape == (2, 7) and bool(z["success"]) and int(z["task_id"]) == 3
    assert z["user_translation_raw"].shape == (2, 3) and z["terminated"].tolist() == [False, True]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --no-sync pytest tests/examples/pi05_libero_shared_autonomy/test_experiment.py -q 2>&1 | tail -3
```

Expected: ImportError (the old `experiment` imports names that no longer exist in `interactive`).

- [ ] **Step 3: Rewrite `experiment.py`**

```python
#!/usr/bin/env python
"""Run a shared-autonomy user study on LIBERO with pi0.5.

Reads an experiment description from YAML (see configs/experiment/), then runs
`n_trials` trials. Each trial draws a task from the configured scene, shows
*you* the task to accomplish, and lets you attempt it with the VLA in the
shared-autonomy mode from the config. The VLA itself only ever receives the
config's `prompt` (e.g. "do something"), so the language instruction and the
human intent can be dissociated on purpose.

    ./experiment.sh --config configs/experiment/<condition>.yaml [--dry-run] [flags]

Every run is written to `<output_dir>/<timestamp>_<config stem>/`:

    config.yaml     the resolved configuration, schedule and the matrices in force
    trials.jsonl    one JSON record per completed trial
    trial_XXX.npz   per-step arrays (operator input, executed action, robot state)
    trial_XXX.mp4   the rollout video
"""

import argparse
import datetime as dt
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from config import CONFIG_DIR, MODES, PROMPT_FROM_TASK, ExperimentSettings, load_experiment_settings
from session import VIDEO_FPS, Session
from teleop import TeleopChain

from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import init_logging

EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiment"


def build_schedule(task_ids: list[int], n_trials: int, order: str, seed: int) -> list[int]:
    """The task shown in each trial, decided up front so a run is reproducible."""
    rng = random.Random(seed)
    if order == "sequential":
        return [task_ids[i % len(task_ids)] for i in range(n_trials)]
    if order == "random":
        return [rng.choice(task_ids) for _ in range(n_trials)]
    schedule: list[int] = []  # shuffled: permute, cycling through the pool
    while len(schedule) < n_trials:
        block = list(task_ids)
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:n_trials]


class TrialRecorder:
    """Collects one trial's per-step data and writes it as a .npz."""

    def __init__(self, chain: TeleopChain):
        self._chain = chain
        self._reads = chain.served.reads
        self.reads_at_start = chain.served.reads  # the counter is session-wide, so baseline it
        self.rows: list[dict] = []
        self.start = time.time()

    @property
    def total_reads(self) -> int:
        """How many times the teleop input was sampled during *this* trial."""
        return self._chain.served.reads - self.reads_at_start

    def __call__(self, step, observation, action, reward, terminated, truncated, info):
        state = observation.get("robot_state", {}) if isinstance(observation, dict) else {}
        eef = state.get("eef", {})
        gripper = state.get("gripper", {})
        joints = state.get("joints", {})
        reads = self._chain.served.reads
        self.rows.append(
            {
                "t": time.time() - self.start,
                "action": np.asarray(action, dtype=np.float32)[0],
                "user_translation": np.asarray(self._chain.served.last_translation, dtype=np.float32),
                "user_translation_raw": np.asarray(self._chain.raw.last_translation, dtype=np.float32),
                "user_gripper": float(self._chain.served.last_gripper),
                "user_reads": reads - self._reads,  # 0 = the input was not consulted this step
                "eef_pos": _first(eef.get("pos"), 3),
                "eef_quat": _first(eef.get("quat"), 4),
                "gripper_qpos": _first(gripper.get("qpos"), 2),
                "joint_pos": _first(joints.get("pos"), 7),
                "reward": float(np.asarray(reward).reshape(-1)[0]),
                "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
                "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
            }
        )
        self._reads = reads

    def save(self, path: Path, **scalars) -> None:
        arrays = {}
        if self.rows:
            for key in self.rows[0]:
                arrays[key] = np.stack([np.asarray(row[key]) for row in self.rows])
        np.savez_compressed(path, **arrays, **{k: np.asarray(v) for k, v in scalars.items()})


def _first(value, size: int) -> np.ndarray:
    """First sub-env's vector from a batched observation entry (NaNs if absent)."""
    if value is None:
        return np.full(size, np.nan, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        array = array[0]
    return array.reshape(-1)[:size]


def prompt_for(prompt_setting: str, task_description: str) -> str:
    return task_description if prompt_setting == PROMPT_FROM_TASK else prompt_setting


def parse_args() -> tuple[argparse.Namespace, ExperimentSettings]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, metavar="PATH", help="experiment YAML (a condition file)")
    parser.add_argument("--n-trials", dest="n_trials", type=int, default=None, help="override experiment.n_trials")
    parser.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    parser.add_argument("--mode", default=None, choices=MODES, help="override control.mode")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="override experiment.output_dir")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the config, print the trial schedule and exit"
    )
    args = parser.parse_args()
    if not Path(args.config).exists():
        found = sorted(p.name for p in EXPERIMENT_CONFIG_DIR.glob("*.yaml") if p.name != "base.yaml")
        parser.error(f"--config: {args.config} not found. Available in {EXPERIMENT_CONFIG_DIR}: {', '.join(found)}")
    overrides = {k: getattr(args, k) for k in ("n_trials", "seed", "mode", "output_dir", "port")}
    try:
        settings = load_experiment_settings(args.config, overrides)
    except (OSError, ValueError) as e:
        parser.error(f"--config: {e}")
    return args, settings


def _resolve_task_ids(settings: ExperimentSettings, where: str) -> list[int]:
    n_tasks = len(Session.list_tasks(settings.session.suite))  # also validates the suite name
    if settings.task_ids is None:
        return list(range(n_tasks))
    for task_id in settings.task_ids:
        if not 0 <= task_id < n_tasks:
            raise ValueError(f"{where}: scene.task_ids: {task_id!r} is not a task of {settings.session.suite}")
    return settings.task_ids


def _provenance(settings: ExperimentSettings, schedule: list[int], session: Session | None) -> dict:
    control = settings.session.control
    resolved = {
        "name": settings.name,
        "n_trials": settings.n_trials,
        "seed": settings.seed,
        "task_order": settings.task_order,
        "output_dir": str(settings.output_dir),
        "suite": settings.session.suite,
        "task_ids": settings.task_ids,
        "prompt": settings.prompt,
        "policy_path": settings.session.policy_path,
        "n_action_steps": settings.session.n_action_steps,
        "compile": settings.session.compile,
        "mode": control.mode,
        "tau": control.tau,
        "n_reverse_steps": control.n_reverse_steps,
        "input_noise": control.input_noise,
        "max_steps": control.max_steps,
        "corruption": control.corruption,
        "reversal_adapter": control.reversal_adapter,
        "port": settings.session.port,
        "schedule": schedule,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if session is not None:
        resolved.update(session.resolved_matrices())
    return resolved


def main():
    args, settings = parse_args()
    try:
        task_ids = _resolve_task_ids(settings, args.config)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    settings.task_ids = task_ids
    schedule = build_schedule(task_ids, settings.n_trials, settings.task_order, settings.seed)
    tasks = Session.list_tasks(settings.session.suite)
    control = settings.session.control

    if args.dry_run:
        print(f"Config OK ({args.config}).\nTrial schedule ({settings.task_order}, seed {settings.seed}):")
        for i, task_id in enumerate(schedule):
            print(f"  trial {i:03d}: {settings.session.suite} task {task_id}: {tasks[task_id]}")
        print(f'\nVLA prompt: "{settings.prompt}"   mode: {control.mode}')
        if control.corruption_matrix is not None:
            print(f"corruption M = {np.array2string(control.corruption_matrix, precision=3)}")
        if control.reversal_adapter_matrix is not None:
            print(f"reversal adapter F = {np.array2string(control.reversal_adapter_matrix, precision=3)}")
        return

    init_logging()
    run_dir = settings.output_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{settings.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    settings.session.suite, settings.session.task_id = settings.session.suite, schedule[0]
    session = Session(settings.session)
    print(f"\nLive view: {session.view.url}  (VSCode should auto-forward the port)\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_provenance(settings, schedule, session), sort_keys=False))

    mode = session.mode
    print(f"\nExperiment {settings.name}: {settings.n_trials} trials, mode {mode}, output {run_dir}")
    print(f'VLA prompt: "{settings.prompt}"' + ("  (the scene's own instruction)" if settings.prompt == PROMPT_FROM_TASK else ""))
    if session.chain.corruption.matrix is not None:
        print(session.chain.corruption.describe())
    if session.adapter.matrix is not None:
        print(session.adapter.describe())
    print(session.announce_mode())

    results = []
    trials_path = run_dir / "trials.jsonl"
    try:
        for trial, task_id in enumerate(schedule):
            if task_id != session.task_id:  # a new scene needs its own env + processors
                session.set_scene(settings.session.suite, task_id)
            task_description = session.task_description
            vla_prompt = prompt_for(settings.prompt, task_description)

            print("\n" + "=" * 78)
            print(f"TRIAL {trial + 1}/{settings.n_trials}   {settings.session.suite} task {task_id}")
            print(f"YOUR TASK: {task_description}")
            print(f'VLA prompt: "{vla_prompt}"   mode: {mode}')
            print("=" * 78)
            session.show_scene()
            # `task` stays up for the whole trial; rollout only overwrites `prompt`,
            # which is already the VLA's, so nothing visibly changes when it starts.
            session.view.stream.set_status(
                task=f"TRIAL {trial + 1}/{settings.n_trials}: {task_description}",
                prompt=vla_prompt,
                state="get ready",
                mode=session.mode_label(),
            )
            try:
                answer = input("Press Enter to start (s = skip this trial, q = end the experiment): ")
            except EOFError:
                answer = "q"
            if answer.strip().lower() == "q":
                print("Ending the experiment early.")
                break
            if answer.strip().lower() == "s":
                print("Skipped.")
                continue

            recorder = TrialRecorder(session.chain)
            started = time.time()
            result = session.rollout(vla_prompt, recorder=recorder, max_steps=control.max_steps)
            duration = time.time() - started

            video_name, steps_name = f"trial_{trial:03d}.mp4", f"trial_{trial:03d}.npz"
            write_video(run_dir / video_name, result.frames, fps=VIDEO_FPS)
            recorder.save(
                run_dir / steps_name,
                task_description=task_description,
                vla_prompt=vla_prompt,
                success=result.success,
                mode=mode,
                task_id=task_id,
            )
            record = {
                "trial": trial,
                "suite": settings.session.suite,
                "task_id": task_id,
                "task_description": task_description,
                "vla_prompt": vla_prompt,
                "mode": mode,
                "tau": session.tau if mode == "shared_flow_control" else None,
                "n_reverse_steps": (
                    (session.n_reverse_steps or session.policy.config.num_inference_steps)
                    if mode == "shared_reverse_flow_steering"
                    else None
                ),
                "input_noise": control.input_noise,
                "deterministic_corruption": session.chain.corruption.label,
                "flow_reversal_adapter": session.adapter.label,
                "success": bool(result.success),
                "steps": int(result.steps),
                "duration_s": round(duration, 2),
                "user_reads": recorder.total_reads,
                "video": video_name,
                "steps_file": steps_name,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                **result.metrics,
            }
            results.append(record)
            with open(trials_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            print(f"{'SUCCESS' if result.success else 'no success'} after {result.steps} steps ({duration:.0f}s)")
            stats = session.stats_line()
            if stats:
                print(stats)
            print(f"saved {steps_name} + {video_name}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        session.close()

    if results:
        wins = sum(r["success"] for r in results)
        print(
            f"\n{len(results)} trials, {wins} success ({wins / len(results):.0%}), "
            f"mean {np.mean([r['steps'] for r in results]):.0f} steps"
        )
    print(f"Data: {run_dir}")


if __name__ == "__main__":
    main()
```

Keep the Apache header comment between the shebang and the docstring. `asdict` is imported but only needed if you prefer `asdict(settings)` for provenance; remove the import if unused (ruff will flag it).

- [ ] **Step 4: Delete the old condition files**

```bash
cd examples/pi05/libero_shared_autonomy
git rm config_experiment_flowcontrol_tau8.yaml config_experiment_flowcontrol_tau8_deterministic_corruption.yaml \
  config_experiment_reverseflow.yaml config_experiment_reverseflow_deterministic_corruption_reverse5steps.yaml \
  config_experiment_reverseflow_deterministic_corruption_reversefull.yaml
cd -
```

- [ ] **Step 5: Run the tests, ruff, and every condition's dry run**

```bash
uv run --no-sync pytest tests/examples -q 2>&1 | tail -3
uv run --no-sync ruff check examples/pi05/libero_shared_autonomy && uv run --no-sync ruff format examples/pi05/libero_shared_autonomy
source examples/pi05/libero_shared_autonomy/env.sh
for c in examples/pi05/libero_shared_autonomy/configs/experiment/*.yaml; do
  [ "$(basename "$c")" = base.yaml ] && continue
  ./examples/pi05/libero_shared_autonomy/experiment.sh --config "$c" --dry-run | head -4
done
```

Expected: tests pass; each dry run prints `Config OK`, a 10-line schedule for `libero_goal`, and the matrices for the `rotz20` conditions (M's first row `[0.94 -0.342 0.]`).

- [ ] **Step 6: Commit**

```bash
git add -A examples/pi05/libero_shared_autonomy tests/examples
git commit -m "Rewrite experiment.py over Session and the overlay configs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: `notebooks/analyze.py`, the stripped notebook, and the nbstripout hook

**Files:**
- Create: `examples/pi05/libero_shared_autonomy/notebooks/analyze.py`
- Move: `examples/pi05/libero_shared_autonomy/analyze_experiments.ipynb` to `notebooks/analyze_experiments.ipynb` (rewritten, outputs cleared)
- Modify: `.pre-commit-config.yaml`
- Create: `tests/examples/pi05_libero_shared_autonomy/test_analyze.py`

**Interfaces:**
- Produces (module `analyze`, importable with `notebooks/` on `sys.path`):
  - `DEADBAND` (from steering), `MODE_SHORT`
  - `find_runs(root: Path) -> list[Path]` (run dirs that contain `trials.jsonl`, sorted)
  - `label_for(config: dict) -> str`
  - `load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]`
  - `load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, dict]]` (trials, configs by run name)
  - `wilson(successes: int, n: int, z=1.96) -> tuple[float, float]`
  - `success_table(trials: pd.DataFrame, configs: dict) -> pd.DataFrame`
  - `compare(trials: pd.DataFrame, a: str, b: str) -> dict` (paired McNemar on shared trial indices)
  - `paired_comparisons(trials: pd.DataFrame) -> pd.DataFrame`
  - `input_activity(trials: pd.DataFrame) -> pd.DataFrame`
  - `describe_rotation(matrix: np.ndarray) -> str`

- [ ] **Step 1: Write the failing test**

`tests/examples/pi05_libero_shared_autonomy/test_analyze.py`:

```python
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("scipy")
NOTEBOOKS = Path(__file__).resolve().parents[3] / "examples/pi05/libero_shared_autonomy/notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
from analyze import (  # noqa: E402
    compare,
    describe_rotation,
    find_runs,
    input_activity,
    label_for,
    load_runs,
    paired_comparisons,
    success_table,
    wilson,
)


def make_run(root: Path, name: str, mode: str, outcomes: list[bool], corruption=None):
    run = root / name
    run.mkdir(parents=True)
    config = {"mode": mode, "tau": 8, "corruption_matrix": corruption, "flow_adapter_matrix": None, "schedule": [0, 1, 0]}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    with open(run / "trials.jsonl", "w") as f:
        for i, ok in enumerate(outcomes):
            f.write(json.dumps({"trial": i, "task_id": i % 2, "task_description": f"task {i % 2}", "success": ok,
                                "steps": 100 + i, "duration_s": 5.0, "steps_file": f"trial_{i:03d}.npz"}) + "\n")
            raw = np.zeros((3, 3), dtype=np.float32)
            raw[1] = [0.5, 0.0, 0.0]
            np.savez(run / f"trial_{i:03d}.npz", user_translation_raw=raw, user_translation=raw * 0.9,
                     user_gripper=np.array([-1.0, -1.0, 1.0]), user_reads=np.array([1, 0, 1]))
    return run


def test_end_to_end_on_synthetic_runs(tmp_path):
    rot = [[0.94, -0.34, 0.0], [0.34, 0.94, 0.0], [0.0, 0.0, 1.0]]
    a = make_run(tmp_path, "20260101_000000_flow_control", "shared_flow_control", [True, False, True])
    b = make_run(tmp_path, "20260101_000001_reverse_flow", "shared_reverse_flow_steering", [True, True, True], rot)
    (tmp_path / "20260101_000002_aborted").mkdir()
    assert find_runs(tmp_path) == [a, b]

    trials, configs = load_runs([a, b])
    assert len(trials) == 6 and set(trials["run"]) == {a.name, b.name}
    assert label_for(configs[a.name]) == "FC tau=8" and label_for(configs[b.name]) == "RFS+M"

    table = success_table(trials, configs)
    assert table.loc[a.name, "successes"] == 2 and table.loc[b.name, "success_rate"] == 1.0
    low, high = wilson(2, 3)
    assert 0 <= low < 2 / 3 < high <= 1
    assert wilson(0, 0) == (pytest.approx(np.nan, nan_ok=True), pytest.approx(np.nan, nan_ok=True))

    result = compare(trials, a.name, b.name)
    assert result["paired_trials"] == 3 and result["B_only"] == 1 and result["A_only"] == 0
    assert 0 < result["mcnemar_p"] <= 1
    assert len(paired_comparisons(trials)) == 1

    activity = input_activity(trials)
    assert activity["commanding_frac"].iloc[0] == pytest.approx(1 / 3)
    assert activity["reads_this_trial"].iloc[0] == 2

    assert "rotation about z" in describe_rotation(np.array(rot))
    assert describe_rotation(np.eye(3)) == "identity (no-op)"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run --no-sync pytest tests/examples/pi05_libero_shared_autonomy/test_analyze.py -q 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'analyze'`.

- [ ] **Step 3: Write `notebooks/analyze.py`**

Move the loading and statistics code out of the notebook's code cells 1, 3, 7, 9, 14, 22 (see the notebook on the branch: `git show HEAD:examples/pi05/libero_shared_autonomy/analyze_experiments.ipynb`) into functions:

```python
"""Load and summarize runs written by experiment.py.

A run directory holds `config.yaml` (resolved settings, schedule and the
matrices in force), `trials.jsonl` (one record per completed trial) and per
trial a `.npz` of step arrays. Runs that share suite, seed and task_order see
the same task schedule, so trial i is the same task across them: comparisons
are paired.
"""

import json
from itertools import combinations
from math import atan2, degrees, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from lerobot.policies.pi05.steering import DEADBAND

MODE_SHORT = {
    "shared_flow_control": "FC",
    "shared_reverse_flow_steering": "RFS",
    "shared_override": "override",
    "teleop": "teleop",
    "policy": "policy",
}
AXES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]


def find_runs(root: Path) -> list[Path]:
    """Run directories under `root` that recorded at least one trial, oldest first."""
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "trials.jsonl").exists())


def label_for(config: dict) -> str:
    """Short condition label from what actually varied: mode, corruption M, adapter F."""
    mode = MODE_SHORT.get(config["mode"], config["mode"])
    if config["mode"] == "shared_flow_control":
        mode += f" tau={config['tau']}"
    tags = []
    if config.get("corruption_matrix") is not None:
        tags.append("+M")
    if config.get("flow_adapter_matrix") is not None:
        tags.append("+F")
    return mode + "".join(tags)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    trials = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines() if line.strip()]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    df = pd.DataFrame(trials)
    df["run"] = run_dir.name
    df["label"] = label_for(config)
    df["mode"] = config["mode"]
    df["corrupted"] = config.get("corruption_matrix") is not None
    df["adapted"] = config.get("flow_adapter_matrix") is not None
    df["run_dir"] = str(run_dir)
    return df, config


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, dict]]:
    frames, configs = [], {}
    for run_dir in run_dirs:
        df, config = load_run(run_dir)
        frames.append(df)
        configs[Path(run_dir).name] = config
    return pd.concat(frames, ignore_index=True), configs


def describe_rotation(matrix: np.ndarray) -> str:
    """Say whether a 3x3 is a rotation about z, and by how much."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        return ""
    if np.allclose(matrix, np.eye(3)):
        return "identity (no-op)"
    orthonormal = np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-3) and np.isclose(np.linalg.det(matrix), 1, atol=1e-3)
    angle = degrees(atan2(matrix[1, 0], matrix[0, 0]))
    return f"rotation about z by {angle:+.0f} deg" if orthonormal else f"not a rotation (det={np.linalg.det(matrix):.3f})"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def success_table(trials: pd.DataFrame, configs: dict[str, dict]) -> pd.DataFrame:
    """Per run: trials, successes, rate, Wilson 95% CI, median steps of successes."""
    rows = []
    for run in configs:
        group = trials[trials["run"] == run]
        n, k = len(group), int(group["success"].sum())
        low, high = wilson(k, n)
        rows.append(
            {
                "run": run,
                "label": label_for(configs[run]),
                "mode": configs[run]["mode"],
                "corrupted": configs[run].get("corruption_matrix") is not None,
                "adapted": configs[run].get("flow_adapter_matrix") is not None,
                "trials": n,
                "successes": k,
                "success_rate": k / n if n else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "median_steps_success": group.loc[group["success"].astype(bool), "steps"].median(),
            }
        )
    return pd.DataFrame(rows).set_index("run")


def compare(trials: pd.DataFrame, a: str, b: str) -> dict:
    """Paired comparison of runs a and b on the trials both completed (McNemar exact test)."""
    pair = trials.pivot_table(index="trial", columns="run", values="success", aggfunc="first")[[a, b]].dropna()
    a_only = int((pair[a].astype(bool) & ~pair[b].astype(bool)).sum())
    b_only = int((~pair[a].astype(bool) & pair[b].astype(bool)).sum())
    discordant = a_only + b_only
    p = stats.binomtest(b_only, discordant, 0.5).pvalue if discordant else np.nan
    return {
        "A": a,
        "B": b,
        "paired_trials": len(pair),
        "A_rate": float(pair[a].mean()),
        "B_rate": float(pair[b].mean()),
        "A_only": a_only,
        "B_only": b_only,
        "discordant": discordant,
        "mcnemar_p": p,
    }


def paired_comparisons(trials: pd.DataFrame) -> pd.DataFrame:
    runs = list(dict.fromkeys(trials["run"]))
    return pd.DataFrame([compare(trials, a, b) for a, b in combinations(runs, 2)])


def input_activity(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-trial operator engagement from the step arrays, joined onto `trials`."""

    def one(row) -> pd.Series:
        z = np.load(Path(row["run_dir"]) / row["steps_file"])
        raw, served = z["user_translation_raw"], z["user_translation"]
        active = np.abs(raw).max(axis=1) >= DEADBAND
        return pd.Series(
            {
                "n_steps": len(raw),
                "commanding_frac": float(active.mean()),
                "mean_speed_when_active": float(np.linalg.norm(raw[active], axis=1).mean()) if active.any() else 0.0,
                "mean_corruption_shift": float(np.linalg.norm((served - raw)[active], axis=1).mean()) if active.any() else 0.0,
                "gripper_closed_frac": float((z["user_gripper"] > 0).mean()),
                "reads_this_trial": int(z["user_reads"].sum()),
            }
        )

    return trials.join(trials.apply(one, axis=1), rsuffix="_npz")
```

- [ ] **Step 4: Run the test**

```bash
uv run --no-sync pytest tests/examples/pi05_libero_shared_autonomy/test_analyze.py -q 2>&1 | tail -3
uv run --no-sync ruff check examples/pi05/libero_shared_autonomy/notebooks tests/examples && uv run --no-sync ruff format examples/pi05/libero_shared_autonomy/notebooks tests/examples
```

Expected: pass, ruff clean.

- [ ] **Step 5: Rebuild the notebook on top of `analyze.py`, outputs cleared**

```bash
git mv examples/pi05/libero_shared_autonomy/analyze_experiments.ipynb examples/pi05/libero_shared_autonomy/notebooks/analyze_experiments.ipynb
```

Then run this one-off script from the repo root (do not commit it) to replace the loading/statistics cells with calls into `analyze.py`, keep the plotting cells and markdown, and clear all outputs:

```bash
uv run --no-sync python - <<'EOF'
import nbformat
from pathlib import Path

path = Path("examples/pi05/libero_shared_autonomy/notebooks/analyze_experiments.ipynb")
nb = nbformat.read(path, as_version=4)
cells = nb.cells

def code(src):
    return nbformat.v4.new_code_cell(src.strip("\n"))

cells[1] = code('''
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path.cwd()))  # notebooks/ on the path when the kernel starts here
sys.path.insert(0, str(Path.cwd() / "examples/pi05/libero_shared_autonomy/notebooks"))  # or at the repo root
import analyze

CANDIDATES = [Path("../../../../outputs/pi05_libero_experiments"), Path("outputs/pi05_libero_experiments")]
RUNS_ROOT = next((p for p in CANDIDATES if p.exists()), None)
if RUNS_ROOT is None:
    raise FileNotFoundError(f"no experiment output directory found; tried {[str(p) for p in CANDIDATES]}")
RUN_DIRS = analyze.find_runs(RUNS_ROOT)
print(f"{len(RUN_DIRS)} runs with data:")
for p in RUN_DIRS:
    print(" ", p.name)
''')
cells[3] = code('''
trials, configs = analyze.load_runs(RUN_DIRS)
LABEL = {run: analyze.label_for(c) for run, c in configs.items()}
ORDER = list(LABEL)
print(f"{len(trials)} trials over {trials['run'].nunique()} runs")
trials[["run", "label", "trial", "task_id", "success", "steps", "duration_s"]].head()
''')
cells[7] = code('''
AXES = analyze.AXES
for run in ORDER:
    config = configs[run]
    print(f"=== {LABEL[run]}  ({run})")
    M, F = config.get("corruption_matrix"), config.get("flow_adapter_matrix")
    if M is None:
        print("  M: none, the operator's command is passed through untouched")
    else:
        M = np.array(M)
        print(f"  M: {analyze.describe_rotation(M)}")
        display(pd.DataFrame(M, index=AXES[:3], columns=AXES[:3]))
    if F is None:
        print("  F: none, the flow reversal uses the policy's own velocity field")
    else:
        F = np.array(F)
        zero_rows = [AXES[i] for i in range(7) if np.allclose(F[i], 0)]
        print(f"  F: translation block is {analyze.describe_rotation(F[:3, :3])}")
        if zero_rows:
            print(f"     rows that are entirely zero: {', '.join(zero_rows)} -> the reversal does not move those dims")
        display(pd.DataFrame(F, index=AXES, columns=AXES))
    print()
''')
cells[9] = code('''
summary = analyze.success_table(trials, configs)
summary.style.format({"success_rate": "{:.0%}", "ci95_low": "{:.0%}", "ci95_high": "{:.0%}"})
''')
cells[14] = code('''
wide = trials.pivot_table(index="trial", columns="run", values="success", aggfunc="first")[ORDER]
tasks = trials.drop_duplicates("trial").set_index("trial")["task_id"]
wide.columns = [LABEL[r] for r in wide.columns]
display(
    wide.assign(task_id=tasks.reindex(wide.index))
    .set_index("task_id", append=True)
    .style.format(lambda v: "" if pd.isna(v) else ("success" if v else "fail"))
)
pairwise = analyze.paired_comparisons(trials)
pairwise["A"], pairwise["B"] = pairwise["A"].map(LABEL), pairwise["B"].map(LABEL)
pairwise.style.format({"A_rate": "{:.0%}", "B_rate": "{:.0%}", "mcnemar_p": "{:.3f}"})
''')
cells[22] = code('''
activity = analyze.input_activity(trials)
activity.groupby("label", sort=False)[
    ["commanding_frac", "mean_speed_when_active", "mean_corruption_shift", "gripper_closed_frac"]
].mean().round(3)
''')
for c in cells:
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None
nb.metadata.pop("widgets", None)
nbformat.write(nb, path)
print("rewritten", path)
EOF
```

Then open the notebook once against real runs (Task 12 produces one) or against the synthetic runs from the test to confirm every cell executes:

```bash
uv run --no-sync --with jupyter jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
  --output /tmp/executed.ipynb examples/pi05/libero_shared_autonomy/notebooks/analyze_experiments.ipynb
```

Expected: exits 0 when `outputs/pi05_libero_experiments` holds at least two runs. If no runs exist yet, defer this check to Task 12 Step 4 and note it.

- [ ] **Step 6: Add the nbstripout hook**

In `.pre-commit-config.yaml`, after the `pyupgrade` repo block:

```yaml
  ##### Notebooks #####
  - repo: https://github.com/kynan/nbstripout
    rev: 0.8.1
    hooks:
      - id: nbstripout
        files: ^examples/pi05/.*\.ipynb$
```

Then:

```bash
uv run --no-sync pre-commit run nbstripout --all-files
git status --short
```

Expected: the hook passes and reports no change (outputs were already cleared).

- [ ] **Step 7: Commit**

```bash
git add -A examples/pi05/libero_shared_autonomy/notebooks examples/pi05/libero_shared_autonomy/analyze_experiments.ipynb .pre-commit-config.yaml tests/examples
git commit -m "Move experiment analysis into notebooks/analyze.py with a stripped notebook

Statistics (Wilson CI, paired McNemar, operator activity) are importable and
tested; the notebook only plots. nbstripout keeps outputs out of git.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 11: Documentation and spec touch-up

**Files:**
- Create: `examples/pi05/README.md`
- Rewrite: `examples/pi05/libero_shared_autonomy/README.md`
- Delete: `examples/pi05/libero_shared_autonomy/README_interactive.md`, `README_experiment.md`
- Modify: `docs/superpowers/specs/2026-09-01-pi05-examples-reorg-design.md` (one paragraph)

- [ ] **Step 1: Write the index**

`examples/pi05/README.md`:

```markdown
# pi0.5 pipelines

Two pipelines built on LeRobot's pi0.5 policy.

| Folder | What it does | Start here |
| --- | --- | --- |
| [`feeding_finetune/`](feeding_finetune/) | LoRA fine-tune pi0.5 on the OmniGibson Kinova feeding task from NWB recordings | `bash examples/pi05/feeding_finetune/train.sh` |
| [`libero_shared_autonomy/`](libero_shared_autonomy/) | Evaluate, drive and study shared-autonomy steering of pi0.5 on LIBERO with a SpaceMouse or keyboard | `./examples/pi05/libero_shared_autonomy/setup.sh` |

The two do not share code: the feeding pipeline is a dataset converter plus a
`lerobot-train` launch; the LIBERO pipeline is a set of runners around
`lerobot.policies.pi05.steering`. Each folder's README is self-contained.
```

- [ ] **Step 2: Rewrite the LIBERO README**

Build `examples/pi05/libero_shared_autonomy/README.md` from the current file plus the two READMEs being removed, with this section order and content:

1. `# pi0.5 + LIBERO shared autonomy` and the current intro paragraph (checkpoint link).
2. `## Install`: the current text, paths updated to `examples/pi05/libero_shared_autonomy/`, plus one sentence that every launcher sources `env.sh` for the cache/GL variables.
3. `## Smoke evaluation`: the current text and the variable table unchanged.
4. `## Interactive prompting`: the current paragraph on the live view, then the mode list (the five bullets from the current README, with the "Partial reversal" paragraph from `README_interactive.md`'s section of that name condensed to one bullet under `shared_reverse_flow_steering`), then the two input-source bullets. End with: "The REPL command reference is `interactive.py --help`; settings can also come from `configs/interactive.yaml` (`interactive.sh --config`)."
5. `## Perturbing the operator`: new, short. Three sub-bullets: input noise (`--input-noise` / `noise`), command corruption (`control.corruption` / `--corruption FILE` / `corruption FILE`), reversal adapter (`control.reversal_adapter` / `--reversal-adapter FILE` / `adapter FILE`). Show the YAML forms exactly as in the spec's section 6, and say that live-loaded files use the same forms.
6. `## Experiments`: from `README_experiment.md`: the launch lines (paths updated, `--config configs/experiment/<condition>.yaml`), "The prompt vs. the task", a "Conditions" paragraph listing the five files and that each `extends: base.yaml`, the configuration table with the `control.deterministic_corruption` / `control.flow_reversal_adapter` rows replaced by `control.corruption` / `control.reversal_adapter` and the `experiment.name` row removed (the run name is the config file's stem), "Running a session", and "What gets recorded" with its two tables verbatim.
7. `## Analysis`: `notebooks/analyze.py` functions in one sentence each, the jupyter launch line updated to the new path, and the note that outputs are stripped by pre-commit.
8. `## Fine-tuning`: the current `lerobot-train` snippet unchanged.
9. `## Troubleshooting`: the section from `README_interactive.md` verbatim.

Then:

```bash
git rm examples/pi05/libero_shared_autonomy/README_interactive.md examples/pi05/libero_shared_autonomy/README_experiment.md
grep -rn 'README_interactive\|README_experiment\|config_experiment\|deterministic_corruption\.yaml\|flow_reversal_adapter\.yaml\|teleop_input\|spacemouse\.py' examples/ AGENT_GUIDE.md CLAUDE.md
```

Expected: no matches.

- [ ] **Step 3: Amend the spec**

In the spec's section 5.4, replace the `resolve_paths` bullet with: "No path resolution is needed: after matrices moved inline, no config key names a file. `policy.path` may be a Hub id and is passed through untouched." Also change section 10 step 1 from "Fast-forward" to "Merge (a regular merge; `main` gained the spec commit after the branch point)".

- [ ] **Step 4: Lint the docs and commit**

```bash
uv run --no-sync pre-commit run --all-files 2>&1 | tail -8
git add -A examples/pi05 docs/superpowers/specs
git commit -m "Document the reorganized pi0.5 examples

One README per pipeline plus an index; the REPL reference lives in
interactive.py --help.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Expected: pre-commit passes (prettier may reflow the Markdown tables; re-add after it does).

---

### Task 12: Runtime verification on the GPU and merge

**Files:**
- None created; produces run directories under `outputs/` (ignored by git)

- [ ] **Step 1: Full test suite for the touched areas**

```bash
uv run --no-sync pytest tests/examples tests/policies/pi0_pi05 -q 2>&1 | tail -3
uv run --no-sync pre-commit run --all-files 2>&1 | tail -3
```

Expected: all pass or skip; pre-commit clean.

- [ ] **Step 2: One short interactive rollout in every mode**

```bash
cd /home/user/Projects/lerobot
printf 'mode policy\n\nmode shared_override\n\nmode shared_flow_control 5\n\nmode shared_reverse_flow_steering 5\n\nmode teleop\n\nquit\n' \
  | ./examples/pi05/libero_shared_autonomy/interactive.sh --config --suite libero_spatial --task-id 0 2>&1 | tee /tmp/claude-1000/-home-user-Projects-lerobot/d61c6ba8-caa7-414e-9def-bca8d7e14bec/scratchpad/interactive_smoke.log | grep -E 'Running|SUCCESS|no success|Error|Traceback|steering applied|guidance applied'
```

Expected: five `Running [...]` lines, five outcome lines, no `Traceback`. The teleop rollout runs 900 steps with nobody driving (about 45 s). `shared_flow_control` and `shared_reverse_flow_steering` report 0 guided steps / 0 steered chunks because no one pushed the stick; that is correct.

- [ ] **Step 3: One steered rollout with the adapter active**

Also from a config, to exercise the inline matrix path:

```bash
printf 'adapter\ncorruption\n\nquit\n' | ./examples/pi05/libero_shared_autonomy/interactive.sh --config \
  --mode shared_reverse_flow_steering --n-reverse-steps 5 \
  --corruption <(echo 'rotation_z_deg: 20') 2>&1 | grep -E 'Command corruption|Reversal adapter|Running|SUCCESS|no success|Traceback'
```

Expected: `Command corruption: ... M = 20` style line (label is the file name, here the process-substitution path), `Reversal adapter: off.`, one rollout, no traceback. Then with the adapter:

```bash
cat > /tmp/claude-1000/-home-user-Projects-lerobot/d61c6ba8-caa7-414e-9def-bca8d7e14bec/scratchpad/adapter.yaml <<'EOF'
translation: corruption
orientation: zero
gripper: zero
EOF
printf 'adapter\n\nquit\n' | ./examples/pi05/libero_shared_autonomy/interactive.sh --config \
  --mode shared_reverse_flow_steering --corruption <(echo 'rotation_z_deg: 20') \
  --reversal-adapter /tmp/claude-1000/-home-user-Projects-lerobot/d61c6ba8-caa7-414e-9def-bca8d7e14bec/scratchpad/adapter.yaml 2>&1 | grep -E 'Reversal adapter|Running|Traceback'
```

Expected: `Reversal adapter: reverse integration uses x_t += h * F @ v, F = adapter.yaml = [+0.94 -0.34 ...` and one rollout.

- [ ] **Step 4: A two-trial experiment and the notebook**

```bash
printf '\n\n' | ./examples/pi05/libero_shared_autonomy/experiment.sh \
  --config examples/pi05/libero_shared_autonomy/configs/experiment/reverse_flow_5steps_rotz20.yaml --n-trials 2 2>&1 | tail -12
ls -t outputs/pi05_libero_experiments | head -1
```

Expected: two trials run without a driver, `trials.jsonl` has two lines, `config.yaml` contains `corruption_matrix` and `flow_adapter_matrix` with the 20-degree rotation, and `trial_000.npz` / `trial_000.mp4` exist. Then verify the recorded fields and run the notebook if it was deferred in Task 10:

```bash
RUN=outputs/pi05_libero_experiments/$(ls -t outputs/pi05_libero_experiments | head -1)
uv run --no-sync python -c "
import json, numpy as np, yaml, sys
run = sys.argv[1]
rows = [json.loads(l) for l in open(f'{run}/trials.jsonl')]
assert len(rows) == 2 and rows[0]['n_reverse_steps'] == 5 and rows[0]['deterministic_corruption'] == 'rotation_z_deg=20', rows[0]
cfg = yaml.safe_load(open(f'{run}/config.yaml'))
assert abs(cfg['corruption_matrix'][0][0] - 0.9397) < 1e-3 and cfg['flow_adapter_matrix'][6][6] == 0.0
z = np.load(f'{run}/trial_000.npz'); assert z['action'].shape[1] == 7 and 'user_reads' in z
print('run ok')" "$RUN"
```

Expected: `run ok`.

- [ ] **Step 5: Merge**

```bash
git switch main
git merge --no-ff reorg-pi05-examples -m "Reorganize the pi0.5 examples under examples/pi05

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git log --oneline -3
```

Do not push; the user decides when to publish.

---

## Self-review notes

- Spec section 4 (steering module) is Task 3; section 5.1 Task 5; 5.2 Task 7; 5.3 and 5.5 Task 8; 5.4 Task 6; 5.6 Task 9; 5.7 Task 10; section 6 Tasks 3, 5, 6; section 7 Task 6; section 8 Tasks 2 and 11; section 9 Tasks 3, 4, 5, 6, 9, 10, 12; section 10 Tasks 0 and 12.
- Deviations from the spec, both recorded in Task 11 Step 3: no `resolve_paths` (no path keys remain), and the initial merge is a regular merge rather than a fast-forward.
- Names used across tasks: `ReversalAdapter` (steering), `CommandCorruption`, `build_corruption`, `read_matrix_spec` (teleop), `spec_label`, `load_interactive_settings`, `load_experiment_settings` (config), `LiveView`, `ACTION_LABELS` (live_view), `Session`, `RolloutResult`, `VIDEO_FPS` (session).
