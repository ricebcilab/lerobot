# Shared-autonomy steering

`lerobot.policies.steering` is laid out as lerobot's other policies are:

- `configuration_steering.py`: `SteeringConfig` (registered `"steering"`) and the constants its validation and its
  adapter specs read (the modes, the scene sources and token geometries, and the action layout a fixed adapter F is
  built in). Its adapter fields (`adapter_matrix`, `adapter_user`, `adapter_task`, `adapter_blocks`) are what a
  `reversal_adapter` spec parses into (`SteeringConfig.from_spec`). `SteeringConfig.build_adapter` builds a spec's
  fixed F, and `checked_matrix`, `rotation` and `translation` are the matrix helpers of the specs.
- `modeling_steering.py`: `SteeringPolicy`, the lerobot policy, and the cores it wraps (below).
- `processor_steering.py`: `make_steering_pre_post_processors`, its two steps (`SteeringSceneTokensProcessorStep`,
  `SteeringCommandProcessorStep`) and `SceneEncoder`, the engine of the scene step.

The modules' training loop is not the package's: it lives in the trainers that run it.

## The methods

Two ways for an operator's translation command to steer the policy without touching the executed action, which is
always the model's own output:

- `FlowControlSteering`, shared flow control (Kao et al. 2026). The command is scaled by `ic_gain` into the noise
  prior's units. It is written into the translation dims of the first `tau` chunk steps of `x_t`, at the start of
  each of the first `lam` Euler steps (`lam` 1 = the initial condition only). The flow itself runs unmodified.
- `FlowReversalSteering`, Flow Reversal Steering (Tang et al. 2026, arXiv:2606.13675). A uniform-velocity reference
  chunk built from the command is integrated _backward_ through the policy's velocity field to the latent noise that
  maps to it. The forward flow then starts from that noise.

Both read an operator `TeleopSource` and take the checkpoint's action statistics from the postprocessor
(`action_mean_std`). `_ChunkSteering.check_steps` is the one check of their step knobs. `ReversalAdapter`, a `MatrixHolder`, is MARS
(Modular Adapters for Reversal Steering, the method formerly called FRS-RA). It has two optional parts:

- a _user module_ that adapts the velocity field of the reversal only: a fixed 7x7 matrix F, typically the known
  command corruption, or a learned `LearnedUserModule` U(x_t, t);
- a _task module_ (A(x_t, t, v) v + b) that adapts the field in both directions, so it reshapes every chunk the
  policy produces.

The forward field is task(v) and the backward field user(task(v)). The modules are trained apart and composed here.
FRS with a user module alone is MARS-U, with a task module alone MARS-T.

Elements of the reference the operator does not command are _delegated_ to the policy (`FlowReversalSteering.delegation_mask`):

- the dims outside `operator_dims` (which of translation / rotation / gripper the operator commands);
- the padding dims;
- the steps beyond `pinned_steps`.

After the reversal their latent is replaced by the unsteered one (fresh Gaussian noise for a full reversal). This is
the noise-space in-painting of Tang et al., App. D.1. Their human interface steers a 5-step reference
(`action_repeats`) in a 15-step chunk of which 10 steps are executed, and sets the other 10 latent steps to
N(0, I). The reference covers half the executed steps, and the policy fills in the rest. The core's default pins the
executed steps instead; set `pinned_steps` to match the paper's ratio.

## The flows and the modules

Under the cores sits `Flow(velocity, adapter, noise_time, batch_time)`, a task-generic integrator over any velocity
field and any number of action dims. `Flow.integrate` is the one Euler integrator for either flow-time convention.
The sign of `h` sets the direction of flow time, and `backward` decides whether the user part of the adapter acts.

| convention                     | reversal (action toward noise) | forward pass (noise toward action)                                                              |
| ------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------- |
| pi0.5 (t=1 noise, t=0 action)  | `t0=0, h=+1/N`                 | `t0=1, h=-1/N`                                                                                  |
| cursor (t=0 noise, t=1 action) | `t0=1, h=-1/N`                 | `t0=1-n/N, h>0` from the reversal's landing point (`t0=0, h=+1/N` for a whole chunk from noise) |

`Flow` also holds:

- `reverse` and `forward` at the policy's own step;
- FRS's `latent` and `schedule`;
- the trainers' validation `Flow.regenerate` and `Flow.heading_cosine`;
- `Flow.uniform_reference`;
- `Flow.paired_chunks` (the steered chunk and the policy's own plan from the same noise).

The modules and their losses:

- `AdaptedField` takes fixed or per-trial matrices, user and task modules. A user module acts through
  `LearnedUserModule.apply_correction`, i.e. its `velocity`.
- `LearnedUserModule` carries its rank knobs. `LearnedTaskModule` reads obstacles when it is given them.
- Their training losses are `LearnedUserModule.field_loss` and `LearnedTaskModule.field_loss`.
- `AdapterModules` is the composed adapter of a policy that takes an observation. It carries the checkpoints:
  `save_module`, `load_module`, `load` and `from_cursor_checkpoint`.

The cores are not pi0.5-specific. They steer any flow policy with pi0.5's sampler interface: a `config`, and a
`predict_action_chunk` taking `noise_fn`, `flow_start_time`, `num_forward_steps`, `x_t_hook` and `velocity_hook`.
They work in either flow-time convention, and `steered_chunk(batch, commands)` serves a batch of commands. The cursor
example drives its own policy through them (`FlowPolicy.sample_chunk`), so the robot and the cursor run one
implementation of each method.

## The scene

`LearnedTaskModule`'s `token_source` picks where its scene tokens come from:

- `discs` (the default) encodes the obstacle discs, `obs["obstacles"]`, relative to the cursor `obs["pos"]`.
- The image sources read `obs["scene"][source]`, a per-layout array. `SceneEncoder.scene_array(layouts, source, size,
encoder)` builds it from the rendered layout (`render_layouts`). `LearnedTaskModule.spec(source, size)` gives the
  module's kwargs for it.
  - `cnn` reads the render itself (uint8 `[B, 3, S, S]`) through a small CNN stem trained with the module.
  - `vlm_image` reads pi0.5's frozen SigLIP tokens of the render (`embed_image`): the patch tokens after the
    multimodal projector, float16 `[B, 256, 2048]`. Every render is resized to 224 px first, so the count does not
    depend on the render size.
  - `vlm_prefix` reads the same image positions in the last hidden state of pi0.5's language model on the prefix:
    the image, then the fixed prompt "Move the cursor to the target.", without state tokens.

`SceneEncoder` computes both from `lerobot/pi05_base`. It loads the policy (and imports pi0.5 and `transformers`) only
on its first call, so the steering package imports without them, and `close()` frees it. The checkpoint is read from
the local Hugging Face cache: on a machine without a valid HF token, run with `HF_HUB_OFFLINE=1`.

## The policy

`SteeringPolicy` puts a mode's core and the adapter behind lerobot's policy contract. `SteeringConfig` has no dataset
features, as `CursorFlowConfig`.

- Its weights are the adapter modules'. The base policy is passed in or loaded from `base_policy_path`.
- `predict_action_chunk` returns the steered chunk for `batch["command"]`. `select_action` serves the chunk's
  executed steps one per call. `forward` returns the adapter's training loss.
- A task module's scene settings become the config's (`SteeringConfig.scene_settings`).
- The processor factory adds the scene step for a `vlm_image` / `vlm_prefix` task module.
- The cursor example's `tools/convert_checkpoints.py --steering` writes a `SteeringPolicy` folder and its pipelines
  beside each `adapter.pt`.
