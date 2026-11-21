# What the hardware said

Fill this in before the first block of trials, from `run.sh probe` and the jog test. It records what the code
cannot know: the station's contract, the arm's limits, and the operator's device. Keep it next to the run
outputs so a later reader can tell which assumptions a session ran under.

## Checkpoint

- path:
- type, `chunk_size`, `n_action_steps`, `num_inference_steps`:
- `max_action_dim`, `use_relative_actions`:
- image keys:
- action statistics: present / absent, and where they came from (`norm_stats.json` path):

## Station

- host, `control_rate`:
- `arms`, `action_keys`:
- cameras, and the map onto the checkpoint's image keys:
- joint limits: from the manifest / overrides (paste them):
- does `step(None)` hold the arm? yes / no:
- gripper travel and which end is closed:
- `joint_velocity` units, and how you confirmed them (a jog at a known command):

## Operator

- SpaceMouse HID id, full scale (a full push reads ~1.0 at this count):
- the axis map, confirmed by jogging each direction:
- gain, deadband:

## The task

- task text given to the VLA:
- prompt used for the policy floor (`"do something"` unless changed):
- start pose, and whether the station restores it (`reset()`):
- are the objects' placements repeatable? If not, how trials are paired:

## Deviations

Anything this session ran with that differs from `DESIGN.md`, and why.
