# Pi0.5 + LIBERO local setup

This folder provides a reproducible local setup for LeRobot's Pi0.5 policy and
the vanilla LIBERO benchmark. It uses the published
[`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned)
checkpoint by default.

## Install

From the repository root:

```bash
./examples/pi05_libero/setup.sh
```

The setup script:

- installs the locked `pi` and `libero` extras into `.venv`;
- writes a non-interactive LIBERO config under `.cache/libero`;
- downloads `lerobot/libero-assets` there; and
- links the installed `hf-libero` package to those project-local assets.

The evaluation launcher also keeps Hugging Face model downloads and Matplotlib
cache data under the project's ignored `.cache` directory. It leaves `HF_HOME`
unchanged so an existing `hf auth login` credential remains available for the
gated PaliGemma tokenizer used by Pi0.5.

Both `.venv` and `.cache` are ignored by git. Re-running the script is safe.

## Smoke evaluation

The default launcher evaluates task 0 of LIBERO-Spatial for one episode:

```bash
./examples/pi05_libero/eval.sh
```

This downloads the Pi0.5 checkpoint on first use. A CUDA GPU with a working
driver is strongly recommended; Pi0.5 is too large for a practical CPU rollout.
The PaliGemma tokenizer is gated: first accept its Hugging Face license and run
`hf auth login`. The initial rollout may spend several minutes compiling the
model; set `PI05_COMPILE_MODEL=false` to disable compilation.

The launcher is configured with environment variables. For example, run the
four standard suites with 10 episodes per task:

```bash
LIBERO_TASKS=libero_spatial,libero_object,libero_goal,libero_10 \
LIBERO_TASK_IDS= \
LIBERO_EPISODES=10 \
./examples/pi05_libero/eval.sh
```

Useful overrides:

| Variable             | Default                         | Meaning                                      |
| -------------------- | ------------------------------- | -------------------------------------------- |
| `PI05_MODEL_ID`      | `lerobot/pi05_libero_finetuned` | Local checkpoint or Hub model ID             |
| `LIBERO_TASKS`       | `libero_spatial`                | Comma-separated LIBERO suites                |
| `LIBERO_TASK_IDS`    | `[0]`                           | Task IDs; set empty to run every task        |
| `LIBERO_EPISODES`    | `1`                             | Episodes per task                            |
| `LIBERO_BATCH_SIZE`  | `1`                             | Parallel evaluation environments             |
| `PI05_ACTION_STEPS`  | `10`                            | Actions executed from each predicted chunk   |
| `PI05_COMPILE_MODEL` | `true`                          | Enable the checkpoint's `torch.compile` path |
| `LIBERO_OUTPUT_DIR`  | `outputs/pi05_libero_eval`      | Evaluation output directory                  |

For headless execution the launcher sets `MUJOCO_GL=egl`. Override it (for
example, with `MUJOCO_GL=osmesa`) only if the machine's rendering setup requires
another MuJoCo backend.

## Interactive prompting

To type free-text instructions and watch the policy react, use the interactive
launcher instead of the batch evaluation:

```bash
./examples/pi05_libero/interactive.sh
```

Full reference for its flags, REPL commands, modes and inputs:
[README_interactive.md](README_interactive.md).

To run scripted, recorded trials from a YAML config instead of the REPL
(shared-autonomy user studies), use `./examples/pi05_libero/experiment.sh` —
see [README_experiment.md](README_experiment.md).

It loads the policy once, opens a live view at `http://localhost:8765` (VSCode
forwards the port automatically) showing the camera stream plus the policy's
per-step action vector (labeled end-effector deltas and gripper command), and
drops into a REPL: type any instruction
(or press Enter for the scene's built-in one) to run a rollout in the current
LIBERO scene. Each rollout is also saved as an MP4 under
`outputs/pi05_libero_interactive/`. Use `tasks` to list the current suite's
scenes, `task <suite> <id>` to switch scene, and `quit` to exit.

Besides the default `mode policy` (model-only control), four teleop modes
let you drive or steer the arm with a 3Dconnexion SpaceMouse and/or the
keyboard:

- `mode teleop` — you drive the arm (x/y/z and the gripper); the model is
  not involved.
- `mode shared_override` — pi0.5 drives, but your x/y/z replaces the
  model's translation in the executed action.
- `mode shared_flow_control [tau]` — pi0.5 drives; while you are pushing, your
  translation (normalized to the model's action space) is written into dims
  0-2 of `x_t` for the first `tau` of the 10 flow-matching denoising steps of
  each action chunk, and the remaining steps denoise freely. The executed
  action is entirely the model's output, steered through the early flow. Idle
  input = pure policy.
- `mode shared_reverse_flow_steering` — Flow Reversal Steering
  ([Tang et al. 2026](https://arxiv.org/abs/2606.13675)): while you are
  pushing, a reference chunk that servos in your direction at uniform velocity
  (rotation zero, gripper held at the model's last command) is integrated
  _backward_ through the policy's own velocity field for the same 10 steps to
  find the latent noise that maps to it; the normal forward flow then runs from
  that noise instead of random noise. The executed action is entirely the
  model's output — the reference only picks the starting noise, so you get the
  generalist action mode nearest your intent. Idle input = pure policy. Costs
  one extra denoising pass per steered chunk. After each rollout the REPL
  prints how many chunks were steered and how far the executed translation
  landed from the reference (in action-std units).

Input sources (both active at once; the SpaceMouse wins while deflected):

- **SpaceMouse** — read directly from `/dev/hidraw`, no extra install; x/y/z
  from the stick, any button toggles the gripper. Connected the first time
  you switch to a teleop mode; if none is plugged in, the mode still works
  with the keyboard alone.
- **Keyboard** — captured by the live-view page (click it to give it focus):
  <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd> move forward/back/left/right,
  <kbd>PgUp</kbd>/<kbd>PgDn</kbd> (or <kbd>W</kbd>/<kbd>S</kbd>) move up/down,
  <kbd>Space</kbd> toggles the gripper, hold <kbd>Shift</kbd> for full speed
  (keys move at half speed by default). Arrow directions follow the SpaceMouse
  axis tuning in `spacemouse.py`, so `↑` moves the arm like pushing the stick
  forward. The page shows the held keys and gripper state under the action bars.

Flags such as
`--policy-path`, `--suite`, `--task-id`, `--port`, and `--compile` are
forwarded to `interactive.py` (see `--help`).

## Fine-tuning

A basic Pi0.5 LIBERO fine-tuning job can be launched with:

```bash
uv run --locked --extra pi --extra libero lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.device=cuda \
  --output_dir=outputs/pi05_libero_train \
  --job_name=pi05_libero_train \
  --steps=6000 \
  --batch_size=1
```

Increase the batch size or use distributed training according to available
VRAM. The published LeRobot result used an effective batch size of 256 on eight
H100 GPUs; the command above deliberately starts with a safer per-process batch
size of 1.
