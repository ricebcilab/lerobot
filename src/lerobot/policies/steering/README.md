# Shared-autonomy steering

`lerobot.policies.steering`, laid out as lerobot's policies are: `configuration_steering.py` (`SteeringConfig`, the action space, the operator's view of it, matrix and adapter specs: `AdapterSpec`, `MatrixHolder`), `modeling_steering.py` (`SteeringPolicy`, the lerobot policy, and the cores it wraps: the MARS adapters, `Flow`, the integrator, and the methods' cores) and `processor_steering.py` (`SceneEncoder`, the scene-token and command steps, `make_steering_pre_post_processors`). The module-level functions these classes replaced stay as deprecated aliases for one round; the modules' training loop lives in the trainers that run it.

Two ways for an operator's translation command to steer the policy without
touching the executed action, which is always the model's own output:

- `FlowControlSteering`: shared flow control (Kao et al. 2026): the command,
  scaled by `ic_gain` into the noise prior's units, is written into the
  translation dims of the first `tau` chunk steps of `x_t` at the start of
  each of the first `lam` Euler steps (`lam` 1 = the initial condition
  only); the flow itself runs unmodified.
- `FlowReversalSteering`: a uniform-velocity reference chunk built from
  the command is integrated _backward_ through the policy's velocity field to
  the latent noise that maps to it (Flow Reversal Steering, Tang et al. 2026,
  arXiv:2606.13675); the forward flow then starts from that noise.

Both read an operator `TeleopSource` and the checkpoint's action statistics
from the postprocessor. `Flow.reverse` is the integrator. `ReversalAdapter`
is MARS (Modular Adapters for Reversal Steering, the method formerly called
FRS-RA): a _user module_ that adapts the velocity field of the reversal only
(a fixed 7x7 matrix F, typically the known command corruption, or a learned
`LearnedUserModule` U(x*t, t)) and/or a \_task module* (A(x_t, t, v) v + b)
that adapts the field in both directions, so it reshapes every chunk the
policy produces. The forward field is task(v) and the backward field
user(task(v)); the modules are trained apart and composed here. FRS with a
user module alone is MARS-U, with a task module alone MARS-T.

Elements of the reference the operator does not command (`operator_dims`
selects which of translation / rotation / gripper they do), the padding dims and
the steps beyond `pinned_steps` are _delegated_ to the policy: after the
reversal their latent is replaced by the unsteered one (fresh Gaussian noise for
a full reversal). This is the noise-space in-painting of Tang et al., App. D.1,
whose human interface steers a 5-step reference (`action_repeats`) in a
15-step chunk of which 10 steps are executed, setting the other 10 latent steps
to N(0, I): the reference covers half the executed steps, and the policy fills in
the rest. The wrapper's default pins the executed steps instead; set
`pinned_steps` to match the paper's ratio.

Under the pi0.5 wrappers sits a task-generic core that takes any velocity field
`velocity(x, t)` and any number of action dims: `Flow.integrate` (one Euler
integrator for either flow-time convention; the sign of `h` sets the
direction of flow time, `backward` whether the user part of the adapter acts):

- pi0.5 (t=1 noise, t=0 action): the reversal (action toward noise) runs
  `t0=0, h=+1/N`, the forward pass (noise toward action) `t0=1, h=-1/N`;
- the cursor (t=0 noise, t=1 action): the reversal runs `t0=1, h=-1/N`, the
  forward pass from the reversal's landing point `t0=1-n/N` with `h>0`
  (`t0=0, h=+1/N` for a whole chunk from noise).

Also `AdaptedField` (fixed or per-trial matrices, user and task modules; a user
module acts through `LearnedUserModule.apply_correction`, i.e. its `velocity`), the
module classes (`LearnedUserModule` with its rank knobs, `LearnedTaskModule`, which
reads obstacles when it is given them) and their checkpoints (`AdapterModules.save_module` / `load_module`), and the module
training losses `LearnedUserModule.field_loss` and `LearnedTaskModule.field_loss`.

The steering cores (`FlowControlSteering`, `FlowReversalSteering`) are
not pi0.5-specific: they steer any flow policy with pi0.5's sampler interface (a
`config` and `predict_action_chunk` taking `noise_fn`, `flow_start_time`,
`num_forward_steps`, `x_t_hook` and `velocity_hook`), in either flow-time
convention, and `steered_chunk(batch, commands)` serves a batch of commands. The
cursor example drives its own policy through them (`FlowPolicy.sample_chunk`), so
the robot and the cursor run one implementation of each method.

`LearnedTaskModule`'s `token_source` picks where its scene tokens come from:
`discs` (the default) encodes the obstacle discs, `obs["obstacles"]`, relative to
the cursor `obs["pos"]`; the image sources read `obs["scene"][source]`, a per-layout array that
`SceneEncoder.scene_array(layouts, source, size, encoder)` (in `processor_steering`) builds from the rendered
layout (`render_layouts`), and `scene_spec(source, size)` gives the module's
kwargs for it. `cnn` reads the render itself (uint8 `[B, 3, S, S]`) through a
small CNN stem trained with the module; `vlm_image` reads pi0.5's frozen SigLIP
tokens of the render (`embed_image`: the patch tokens after the multimodal
projector, float16 `[B, 256, 2048]`: every render is resized to 224 px first, so
the count does not depend on the render size); `vlm_prefix` reads the same
image positions in the last hidden state of pi0.5's language model on the prefix
(the image, then the fixed prompt "Move the cursor to the target.", without
state tokens). `SceneEncoder` computes both from `lerobot/pi05_base`; it loads the
policy (and imports pi0.5 and `transformers`) only on its first call, so the
steering package imports without them, and `close()` frees it. The checkpoint is
read from the local Hugging Face cache: on a machine without a valid HF token,
run with `HF_HUB_OFFLINE=1`.

`SteeringPolicy` puts a mode's core and the adapter behind lerobot's policy contract (`SteeringConfig`, registered
`"steering"`, has no dataset features, as `CursorFlowConfig`): its weights are the adapter modules', the base policy
is passed in or loaded from `base_policy_path`, `predict_action_chunk` returns the steered chunk for
`batch["command"]` and `forward` the adapter's training loss. The cursor example's `tools/convert_checkpoints.py
--steering` writes a `SteeringPolicy` folder beside each `adapter.pt`.
