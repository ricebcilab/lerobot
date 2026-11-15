# Shared-autonomy steering

`lerobot.policies.steering` is laid out as lerobot's other policies are:

- `configuration_steering.py`: `SteeringConfig` (registered `"steering"`) and the constants its validation and its
  adapter specs read (the modes, the scene sources and token geometries, and the action layout a fixed adapter F is
  built in). Its adapter fields (`adapter_matrix`, `adapter_user`, `adapter_task`, `adapter_blocks`) are what a
  `reversal_adapter` spec parses into (`SteeringConfig.from_spec`). `SteeringConfig.build_adapter` builds a spec's
  fixed F, for any `n_dims` (7, the operator space, by default; 2 on the cursor). The matrices inside a spec are the
  operator readout's grammar (`lerobot.teleoperators.readout`: `matrix_spec`, `corruption_spec`, `checked_matrix`).
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
- `FlowReversalSteering`, Flow Reversal Steering (Tang et al. 2026, arXiv:2606.13675). A reference chunk built from
  the command is integrated _backward_ through the policy's velocity field to the latent noise that maps to it. The
  forward flow then starts from that noise. The reference is the action space's rule (`reference_chunk`): in the 7-D
  operator space (pi0.5's) a uniform-velocity chunk with the gripper held (`Flow.uniform_reference`); in an action
  space without that gripper dim (the cursor's two dims) the command held at `reference_step` (`max_speed * dt`)
  action units per control step, `(per_step * reference_step - mean) / std`.

Both take the checkpoint's action statistics from the postprocessor (`action_mean_std`). A core gets the operator's
command per call (`steered_chunk(batch, commands)`, one chunk; `chunk_for(batch, commands)`, a rollout's next chunk
with the state a rollout keeps: FRS's held gripper and its counts), or pulls it from a `TeleopSource` once per chunk
as a policy of its own (`select_action`). `FlowControlSteering.check_steps` (both cores have it) is the one check of
their step knobs.
`AdapterModules`, the one adapter holder of the cores, the policy and the LIBERO session, is MARS (Modular Adapters
for Reversal Steering, the method formerly called FRS-RA). It has two optional parts:

- a _user module_ that adapts the velocity field of the reversal only: a fixed matrix F, typically the known
  command corruption, or a learned `LearnedUserModule` U(x_t, t). F is in the operator's (environment) units and is
  normalized at use by the postprocessor's action statistics (`normalized_matrix(std)`, S^-1 F S; a uniform std,
  the cursor's scalar action scale, leaves it as it is);
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
- `AdapterModules` holds the parts: its `field(obs, std=..., like=...)` is the `AdaptedField` the flows apply
  (the task module bound to the observation, F normalized by `std` and placed on `like`'s dtype and device). It acts
  on `n_dims` dims (2 on the cursor); `AdapterModules(n_dims=None)` is the 7-D holder of the pi0.5 cores and the
  LIBERO session, each module on its own dims. It describes itself (`describe`, `label`, `user_label`,
  `task_label`), loads modules into its slots (`load_modules`) and carries the checkpoints: `save_module`,
  `load_module` and `load`.

The cores are not pi0.5-specific. They steer any flow policy with pi0.5's sampler interface: a `config`, and a
`predict_action_chunk` taking `noise_fn`, `flow_start_time`, `num_forward_steps`, `x_t_hook` and `velocity_hook`.
They work in either flow-time convention, and `steered_chunk(batch, commands)` serves a batch of commands. The cursor
policy offers the interface itself (`CursorFlowPolicy`), so the robot and the cursor run one implementation of each
method, through one policy: the cursor example's controller, sweeps and agents take their chunks from
`SteeringPolicy` over it, and the LIBERO session's shared modes are a `SteeringPolicy` over pi0.5 with the session's
adapter in its slot and the command in the batch.

## The scene

`LearnedTaskModule`'s `token_source` picks where its scene tokens come from:

- `discs` (the default) encodes the obstacle discs, `obs["obstacles"]`, relative to the cursor `obs["pos"]`.
- The image sources read `obs["scene"][source]`, a per-scene array. The pre-processor's scene step
  (`SteeringSceneTokensProcessorStep`) encodes it from the scene's pictures, `observation.scene_images` (uint8
  `[B, 3, S, S]`, `S` the task module's image size, `SCENE_IMAGE_SIZE` = 224 by default). The package draws no
  scene: the cursor's pictures are its layouts' render, drawn by the cursor's own step
  (`CursorSceneRenderProcessorStep`, in `lerobot.policies.cursor_flow`, where `render_scene_array` is a layout
  batch's array). `LearnedTaskModule.spec(source, size)` gives the module's kwargs for it.
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

- Its weights are the adapter modules'. The base policy is passed in, or loaded from `base_policy_path` (a folder or
  a Hub repo id) together with that folder's processors (a converted cursor folder has them:
  `convert_checkpoints.py`).
- Every base is driven one way, through pi0.5's sampler interface: a `config` with `num_inference_steps`,
  `chunk_size`, `max_action_dim`, `n_action_steps` and `noise_time`, `predict_action_chunk(batch, **hooks)` (the
  normalized chunk), `reset()` and `velocity_field(batch)`. `PI05Policy` and `CursorFlowPolicy` offer it
  (`CursorFlowPolicy.wrapping(model)` puts a bare `FlowPolicy` behind it); a base without it is refused by name.
  The flow's settings are the base's config's: `SteeringConfig` does not repeat them.
- `frs` needs the base's postprocessor, for its action statistics, on every base (a cursor base's is
  `make_cursor_flow_pre_post_processors(base.config)[1]`); pass it with the base, or with
  `set_base_policy(policy, postprocessor=...)`.
- `predict_action_chunk` returns the steered chunk for `batch["command"]`, the operator's deflection: `(B, n)`,
  `(B, H, n)` per chunk step, or `(n,)` for a single unbatched trial. `n` is at most `n_dims` (a pi0.5 base's
  translation is 3). The chunk is normalized, as the base's: the base's postprocessor unnormalizes it.
  `select_action` is a rollout's: it serves the chunk's first `n_action_steps` steps (the base's) one per call, takes
  the command of the batch that starts a chunk, keeps the base's own plan for the step served (`last_policy_action`,
  from the same noise) and the core's rollout state (`chunk_for`); `reset` starts a new rollout. Mode `"none"` is the
  base's own chunk through the task module, the command ignored; so is a batch without a command, in any mode, and
  `noise` (pi0.5's keyword) starts that chunk from a given noise (a steered chunk draws its own: a command with
  `noise` raises).
- The base gets the batch untouched less the steering policy's own keys (`command` and a training batch's). The
  adapter binds to its own view of it, the task module's input: the `observation.` prefix stripped, the scene step's
  inputs dropped, `observation.scene_tokens` at `scene[scene_source]`, a single unbatched trial given a batch of one.
- `forward` returns the adapter's training loss on the example trainer's batches: `kind`, `chunk` (in action units,
  normalized by the postprocessor's statistics, so a task batch needs the postprocessor too) or `latent` and
  `bci_target`, `noise`, `time`, `weight`. The pre-processor carries these through untouched, but does not build
  them.
- A task module's scene settings become the config's (`SteeringConfig.scene_settings`).
- The processor factory adds the scene step for a `vlm_image` / `vlm_prefix` task module. The step reads pictures;
  the cursor example puts the cursor's render step before it (`cursor_steering_pre_post_processors`).
- The cursor example's `tools/convert_checkpoints.py --steering` writes a `SteeringPolicy` folder and its pipelines
  beside each `adapter.pt`.

`make_policy` cannot build a steering policy: it needs dataset or env features, and the base must be passed in
(the `cursor_flow` precedent). Build it with `SteeringPolicy(config, base_policy=...)`, or
`get_policy_class("steering")`, or `from_pretrained`, and its processors with `make_steering_pre_post_processors`,
or with `make_pre_post_processors(config, pretrained_path=...)` for a saved folder.

On the cursor's base:

```python
from lerobot.policies.cursor_flow import CursorFlowPolicy, make_cursor_flow_pre_post_processors
from lerobot.policies.steering import SteeringConfig, SteeringPolicy, make_steering_pre_post_processors

base = CursorFlowPolicy.from_pretrained("path/to/cursor_policy")  # the policy whose chunks are steered
_, base_postprocessor = make_cursor_flow_pre_post_processors(base.config)  # its action statistics
config = SteeringConfig(mode="frs", adapter_user="path/to/adapter.pt", device="cpu")  # MARS-U
policy = SteeringPolicy(config, base_policy=base, postprocessor=base_postprocessor)
preprocessor, _ = make_steering_pre_post_processors(policy.config)

# obs: the cursor's observation dict (pos, hist, targets, mask, goal, ...); deflection: (B, 2) in [-1, 1]
batch = preprocessor({**{f"observation.{k}": v for k, v in obs.items()}, "command": deflection})
action = base_postprocessor(policy.select_action(batch))  # (B, 2), screen units per step
```

From a converted folder (`convert_checkpoints.py --steering`), or any folder `save_pretrained` and the pipelines'
`save_pretrained` wrote:

```python
from lerobot.policies.factory import make_pre_post_processors

policy = SteeringPolicy.from_pretrained(
    "path/to/adapters/<name>", base_policy=base, postprocessor=base_postprocessor
)
preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path="path/to/adapters/<name>")
```
