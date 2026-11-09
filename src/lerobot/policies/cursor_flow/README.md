# Cursor flow policy

`cursor_flow` is the flow-matching policy of the MARS cursor task: a simulated 2D cursor-reaching task with
targets and optional obstacles, used to study shared autonomy (`examples/pi05/mars/cursor/`). It is a small
transformer (about 0.9M parameters at the defaults) that outputs chunks of `horizon` per-step cursor
displacements.

- `FlowPolicy` (`modeling_cursor_flow.py`) is the network itself, moved verbatim from the cursor example. Its
  observation is a dict of tensors: cursor position `pos` (B, 2), the last `hist` displacements `hist`
  (B, h, 2), `targets` (B, N, 2) with `mask` and `goal` (B, N), and, for a policy built with
  `obstacle_tokens`, `obstacles` (B, K, 3) with `obstacle_mask` (B, K). The flow runs from t = 0 (noise) to
  t = 1 (action). `mode="regress"` is the baseline that predicts the chunk directly.
- `CursorFlowPolicy` is a thin `PreTrainedPolicy` around it. It saves as `config.json` + `model.safetensors`,
  so it loads with `from_pretrained`. It offers pi0.5's sampler interface, which `SteeringPolicy` drives every
  base through: the config's `num_inference_steps` (10), `chunk_size` (`horizon`), `n_action_steps` (8),
  `max_action_dim` (2) and `noise_time` (0.0, the cursor's flow convention); `predict_action_chunk(batch, **hooks)`,
  the plain policy chunk (`FlowPolicy.sample_chunk`, taking pi0.5's hooks), normalized; `select_action`, its first
  `n_action_steps` steps one per call (`reset` empties the queue); and `velocity_field(batch)`. The batch is the
  observation dict, plain or under `observation.`, a single unbatched trial (a `(2,)` `pos`) given a batch of one.
  `CursorFlowPolicy.wrapping(model)` puts an existing `FlowPolicy` behind the interface without copying it.
- `make_cursor_flow_pre_post_processors(config)` (`processor_cursor_flow.py`, also the factory's
  `make_pre_post_processors` for a `CursorFlowConfig`) makes its pipelines: the postprocessor's unnormalizer scales a
  chunk to screen units, `chunk * action_scale` (mean 0, std `action_scale`), and carries the action statistics the
  steering cores read.
- The cursor's scenes as pictures, for a steering policy's scene step (`processor_cursor_flow.py`):
  `CursorSceneRenderProcessorStep` reads the trials' layouts (`observation.layouts`) and writes their render
  (`observation.scene_images`, `render_layouts` at `render_size`) and a digest of what it is drawn from
  (`observation.scene_digest`), the key the scene step keeps its tokens under; `render_scene_array` is a layout
  batch's scene array for a task module's source (the render for `cnn`, its tokens for `vlm_image` /
  `vlm_prefix`).

It has no dataset features, and `lerobot-train` does not train it. It is trained by the example's own trainer,
`examples/pi05/mars/cursor/train/train_policy.py`, which writes the cursor project's `policy.pt` format.
`examples/pi05/mars/cursor/tools/convert_checkpoints.py` converts those files (and the cursor's adapter
checkpoints) in place to the formats lerobot loads. A converted policy folder also holds the policy's pipelines
(`policy_preprocessor.json`, `policy_postprocessor.json` and the unnormalizer's statistics), so a steering policy that
loads it by path has its postprocessor; a folder converted before them gets them on a re-run.

```python
from lerobot.policies.cursor_flow import CursorFlowPolicy, make_cursor_flow_pre_post_processors

policy = CursorFlowPolicy.from_pretrained("path/to/checkpoints/policies/flow_goaldrop", strict=True)
_, postprocessor = make_cursor_flow_pre_post_processors(policy.config)
chunk = postprocessor(policy.predict_action_chunk(obs))  # (B, 16, 2) screen units per step
```
