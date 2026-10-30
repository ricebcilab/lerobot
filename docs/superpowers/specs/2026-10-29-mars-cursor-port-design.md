# MARS project: port the cursor-control shared-autonomy work into lerobot

**Date:** 2026-10-29
**Status:** design, awaiting Nishal's approval
**Source:** `ricebcirig/analysis/nishal/cursor_control_shared_autonomy` at `bd2a48f2` (the "cursor project" below)
**Target:** this repo (`ricebcilab/lerobot`, public), `main`

## 1. Intent

A collaborator at another site will run MARS (Modular Adapters for Reversal Steering) with a SpaceMouse,
first on the 2D cursor task and then on their own robot. The cursor task is their on-ramp: on day one they
clone this repo, download the weights, run a full human-study session with their SpaceMouse, and send back
the per-arm metrics. If MARS beats the other arms the way it does for our participants, their setup is
right, and they move on to the robot with the same MARS code.

Today the cursor work lives in one person's analysis folder in a different, private repo, while the robot
side (pi0.5 in LIBERO) lives here. Nishal wants one home for the shared project: the method once, both
tasks on top of it, and the paper figures and results next to them, so nobody jumps between two repos.

The repo is public and that is fine before the paper is out (Nishal's decision). Results, data and
checkpoints stay gitignored.

## 2. Goals

1. One implementation of the shared-autonomy method, task-generic, in
   `src/lerobot/policies/pi05/steering.py`, used by both the cursor and the LIBERO examples.
2. The full cursor project runs from this repo: environment, policy training, simulated users, user- and
   task-module training, sweeps and the grid, the human-study app (SpaceMouse sessions, logging,
   per-participant analysis), reports and the paper figures.
3. A fresh clone plus downloaded weights runs a human session on the first day and prints the per-arm
   metrics (success, time, wrong target, timeout per arm, MARS against teleop, flow control and FRS).
4. The existing LIBERO shared-autonomy example keeps working, moved under the same project and wired
   onto the shared core.

## 3. Non-goals

- The user module's rank knobs (`step_rank`, `cross_step_rank`) are left out of the port, and so is
  everything that uses them: `research/adapter_ranks_*.py`, the two rank tests in `test_adapter.py`, the
  trainer flags, and the `adapter_ranks` checkpoints and results (not copied). They stay in ricebcirig.
- No robot-side task-module trainer for LIBERO or a real robot. The cursor task-module trainer comes over
  as is; making it work for pi0.5 is later work.
- No change to any method's behavior or default values.
- The ricebcirig copy stays alive and is not deleted or frozen (Nishal's decision). The two copies may
  drift; this repo becomes the one new work goes into.
- No git history of the cursor project is carried over; the port is a fresh copy.

## 4. Anti-goals (a failure even if it runs)

- The grid's or the study's numbers change. The ported code, run on the existing checkpoints and the same
  seeds, must reproduce the cursor project's results (see §8).
- Two copies of the method in this repo (one for pi0.5, one for the cursor).
- Participant data, results or checkpoints committed to git.
- Machine-specific paths (`/home/user/...`, the `t15` venv, `brand-rice`) in committed code.

## 5. Placement (Nishal's decisions are marked)

```
src/lerobot/
  policies/pi05/steering.py        ALL shared-autonomy method code (Nishal): task-generic core + pi0.5 wrappers
  policies/<cursor policy>/        the small cursor flow-matching policy as a new policies/ entry (Nishal)
  teleoperators/spacemouse/        SpaceMouse teleoperator (my call: follows teleoperators/gamepad/)
  envs/cursor.py                   the cursor environment (my call: next to envs/libero.py)
examples/pi05/mars/                the project (Nishal: one project named MARS under examples/pi05/, task subfolders)
  README.md                        what MARS is, day-one quick start, links to the two tasks and the figures
  cursor/                          study app, sessions, simulated users, trainers, sweeps, analysis, reports, configs, research
  libero/                          git mv of examples/pi05/libero_shared_autonomy, imports rewired
  figures/                         paper figure builders (build_figures.py) and their committed PDFs/PNGs
  results/   checkpoints/          gitignored, same layout as the cursor project's
examples/pi05/README.md            index: libero_shared_autonomy entry replaced by mars
docs/superpowers/specs/            this document
```

Nishal said no separate `mars/` folder under `src/`; every piece of method code goes in `steering.py`, and
trainers and the teleop source go wherever lerobot's existing folders fit best.

**Rule for the split (my call):** code in `src/` must not import from `examples/`. So `src/` gets only what
knows nothing about the cursor's channel, simulated users, demos or sessions; anything that does stays in
`examples/pi05/mars/cursor/`, even if it is "method-shaped".

### 5.1 `steering.py`: the core

The file today is pi0.5-specific (LIBERO action dims at module level, hooks into pi0.5's sampler). The
cursor project's `cursor_sa/steering.py` (493 lines) holds the same methods written against any policy's
velocity field: the user and task modules, the reversal adapter, `reverse_flow` / `forward_flow`, flow
control (Kao et al.: `tau`, `ic_gain`, `lambda`), FRS / MARS chunks and the `SharedController`.

The merged file has a task-generic core (functions and modules that take a velocity field and a number of
action dims) and the existing pi0.5 wrappers (`FlowControlPolicy`, `FlowReversalSteeringPolicy`) built on
it. Where the two copies are the same method (`LearnedUserModule` and the cursor `UserModule` are both one
n × n matrix per chunk step, zero-initialized), they become one class parametrized by `n_dims`; where they
differ (the cursor task module reads obstacle tokens, the pi0.5 one does not), the difference stays as an
option of the one class. The LIBERO action-dim constants stay but stop being the core's defaults.

The cursor `steering.py` imports two helpers from `cursor_sa/channel.py`, which itself imports the cursor
env: `is_active` (the deadband check that leaves an idle operator's rows to the plain policy) and
`corrupt_chunk` (applies a user's corruption to expert deflections, for training data). So that
`steering.py` imports nothing from the cursor (my call):

- `steered_chunk` (one chunk under policy / flow control / FRS / MARS) goes to `steering.py` and does the
  deadband check itself with a deadband argument (`steering.py` already has `DEADBAND` for LIBERO, also
  0.05). Its conversion from deflections to screen-unit actions (`to_action`, which needs the cursor's `dt`
  and `max_speed`) is done by the caller or passed in.
- `bci_chunk`, `to_action`, `to_deflection`, `observation` and `SharedController` (the cursor's closed
  loop: layouts, history, chunk execution) stay in the cursor example. The module's training losses
  (flow-matching field loss for the task module, the user module's reversal loss) are method code and go in
  `steering.py`; the data they train on does not.

### 5.2 The cursor policy

`cursor_sa/policy.py` (`FlowPolicy`, 157 lines) becomes a `policies/` entry with lerobot's file pattern
(`configuration_*.py`, `modeling_*.py`, `README.md`). **A thin wrapper, not the full policy contract (my
call):** a `PreTrainedConfig` / `PreTrainedPolicy` pair registered in `policies/factory.py`, saving as
`config.json` + `model.safetensors` so it loads from the Hub with `from_pretrained`, around the existing
`FlowPolicy` with its own dict observation (layouts, position, history, obstacles), `horizon`, `hist` and
`action_scale`. No processor pipeline, no dataset features, no `lerobot-train`. Its trainer
(`train/train_policy.py`) stays with the example in `examples/pi05/mars/cursor/train/`. Old `policy.pt` files load
through a one-time converter.

### 5.3 Other source pieces

- **SpaceMouse:** `cursor_sa/devices.py` and `examples/pi05/libero_shared_autonomy/teleop.py` each read the
  SpaceMouse from `/dev/hidraw`; one `Teleoperator` subclass replaces both. The keyboard fallback stays with
  the examples. Both readers hard-code one older model (HID id `0000046D:0000C62B`, report id 1, full scale
  350). The collaborator's model is unknown; they run Linux with a GPU (Nishal). So the teleoperator stays on
  Linux `/dev/hidraw`, finds any 3Dconnexion device (Logitech `046D` and 3Dconnexion `256F` vendor ids), and
  takes the device id and full-scale count from its config, defaulting to today's values so our setup
  behaves as before. The README says how to check a new device's full-scale count (the deflection at a
  full push should read about 1), since that scale defines "full push" in the study.
- **Cursor env:** `cursor_sa/env.py` (509 lines, layouts, obstacles, targets) goes to `envs/cursor.py`. Registering it as an
  `EnvConfig` subclass is left to the builder.
- **Trainers (my call):** both cursor trainers (`train_policy.py`, `train_adapter.py`) stay with the example
  in `examples/pi05/mars/cursor/train/`, because they read the cursor's simulated users, demos, channel and
  sessions. No `src/` trainer CLI. Their loss and module code comes from `steering.py`. The LIBERO
  `train_user_module.py` stays as it is, importing the merged core. A task-generic trainer CLI is later
  work, together with a robot task-module trainer.
- **Module checkpoints (my call):** user and task modules keep the `.pt` format `save_adapter_module` writes
  (kind, config, weights, meta), loaded with `weights_only=True`. Cursor module checkpoints, which have no
  `kind` / `n_dims`, load through the same one-time converter.

### 5.4 Example code

Everything else in the cursor project is task code and goes to `examples/pi05/mars/cursor/`, keeping its
internal layout: `agents.py` (simulated users), `channel.py`, `demos.py`, `session.py`, `records.py`,
`renderer.py`, `analysis.py`, `config.py`, `paths.py`, `run_experiment.py`, `run_sweep.py`,
`analyze_results.py`, `configs/`, `reports/` (builders and their PDFs), `research/` (the scripts) and `tests/`.
The paper figures go to `examples/pi05/mars/figures/`. The cursor README's research notes come along as the
cursor README.

Cursor tests move into lerobot's `tests/` where they exercise `src/` code (steering, policy, env,
teleoperator), and stay with the example otherwise (session, sweeps, figures). They run under lerobot's pytest.

## 6. Data, results and weights

- `examples/pi05/mars/results/` and `examples/pi05/mars/checkpoints/` are gitignored. Nishal copies the cursor project's
  `results/` (156 MB: p1–p7 study sessions, simulation sweeps, research outputs) and the needed checkpoints
  into them, so the figures and reports rebuild from this repo.
- The weights the collaborator needs (the cursor policy, the study's user modules, the grid's task and user
  modules for the five seeds) go to one public Hugging Face Hub repo under the `ricebcilab` org, with a
  one-command download into `examples/pi05/mars/checkpoints/`. The builder prepares the upload and runs it only
  on Nishal's command at that time (Nishal).
- Participant data never leaves the gitignored folders.

## 7. Dependencies and install

The cursor project needs numpy, torch, PyYAML, matplotlib, reportlab, Pillow, scipy and pygame, all
version-pinned elsewhere in `pyproject.toml` except reportlab. A `mars` optional extra
(`pip install -e ".[mars]"`) covers the cursor app, reports and figures; `hidapi` or plain `/dev/hidraw`
access for the SpaceMouse follows the gamepad extra's pattern. The README states the udev rule a SpaceMouse
needs on Linux.

## 8. Done means

1. **Same numbers.** Seed 0 only (Nishal): the ported code re-runs the grid's test stage (500 trials,
   seed 800) for every cell and method at the saved depths (`depths.yaml`), on the existing checkpoints, and
   each cell's success lies within the original run's 95% CI. `test.json` was merged from the `grid`,
   `grid_swap/{A,B}_UxT` and `retrained_mars_user` runs; the re-run covers all of their methods. Seeds 1–4 are
   copied, not re-run. The figure builder then rebuilds Fig 1–3 from the copied results with the same
   `stats.json` values, and Fig 1 (computed live from checkpoints) matches. Bit-exact is not expected
   (lerobot pins a different torch).
2. **LIBERO unchanged.** The LIBERO example's tests and a short interactive run with each control mode
   behave as before the move.
3. **Day one works.** On a fresh clone and a fresh environment, with only `pip install -e ".[mars]"`, the
   Hub download and a SpaceMouse: one documented command starts a human session, the session completes,
   and one documented command prints the per-arm metrics.
4. **Tests pass.** The cursor project's tests (adapter, training, session, sweep, grid, obstacles, cursor
   flow, figures) pass in their new homes, alongside lerobot's existing policy and teleoperator tests.
5. **No machine paths** in committed files.

## 9. Constraints

- GPU work (re-running the grid to check §8.1) runs on GPU 1 only, at most 3 trainings and 6 processes
  at once; GPU 0 drives the desktop.
- No commit or push without Nishal's command at that time; a plan's per-stage commits are not an order.
- lerobot's code style (ruff, license headers on new `src/` files, type hints in `src/`) applies to
  ported `src/` code; example code keeps the cursor project's style.

## 10. Left to the builder

- How the merged `steering.py` is organized inside, the names of the unified classes, and whether old names
  stay as aliases.
- Whether old cursor checkpoints load directly or through a one-time converter, as long as §8.1 holds.
- The cursor policy's folder name under `policies/`.
- Whether the cursor env is registered as an `EnvConfig`.
- The order of work and how to split it into reviewable steps.
