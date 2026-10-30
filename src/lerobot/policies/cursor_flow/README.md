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
  so it loads with `from_pretrained`. `predict_action_chunk` returns the plain policy chunk (10 Euler steps of
  `lerobot.policies.pi05.steering.integrate`) in screen units, and `select_action` returns its first step.
  The cursor code uses `.model` directly, together with the MARS steering in `lerobot.policies.pi05.steering`.

It has no processor pipeline and no dataset features, and `lerobot-train` does not train it. It is trained by
the example's own trainer, `examples/pi05/mars/cursor/train/train_policy.py`, which writes the cursor
project's `policy.pt` format. `examples/pi05/mars/cursor/convert_checkpoints.py` converts those files (and the
cursor's adapter checkpoints) in place to the formats lerobot loads.

```python
from lerobot.policies.cursor_flow import CursorFlowPolicy

policy = CursorFlowPolicy.from_pretrained("path/to/checkpoints/policies/flow_goaldrop", strict=True)
chunk = policy.predict_action_chunk(obs)  # (B, 16, 2) screen units per step
```
