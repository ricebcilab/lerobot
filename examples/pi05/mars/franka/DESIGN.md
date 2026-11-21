# MARS on the Franka: decisions

This folder steers a π0.5-DROID checkpoint on an N-Droids Franka through c3po and the MARS methods from
`lerobot.policies.steering`. It is an example, in the sense of `../libero/`: nothing here is policy code.

`interfaces.py` is frozen. These decisions are frozen with it. `../../../../franka-mars-plan.pdf` (in the
workspace, not the repository) carries the reasoning.

## What the station takes

c3po's Franka station accepts joint velocities and a gripper position, not Cartesian targets:

```
action:  {arm}/joint_velocity  (7,) rad/s      {arm}/gripper_position  (1,)
obs:     {arm}/joint_position  (7,)            {arm}/gripper_position  (1,)  + cameras
```

This is read off a previous project's c3po adapter on this arm, and slice 0 confirms it from the manifest.
It removes the integration step: the station integrates the velocity, so the runner holds one action per
`ceil(control_rate / action_rate)` station ticks instead of sending position targets.

## The action and the command

- **Action**, 8 numbers in the checkpoint's units: `(qdot_1..qdot_7, gripper)`. Unnormalized by the
  checkpoint's postprocessor, then clamped, then split into the station's two keys.
- **Command**, 7 numbers in `[-1, 1]`: `qdot_desired / qdot_full`, where `qdot_full` is the joint speed a full
  stick deflection asks for. The Jacobian turns the stick's twist into `qdot_desired`.
- The **gripper is not in the command**. It is delegated to the policy (`gripper_dim=7`), because the
  operator's gripper command is a button, not a continuous signal.

## The steering configuration, per arm

```
SteeringConfig(
    mode=...,                     # "none" | "flow_control" | "frs"
    n_dims=7,                     # commanded joints; the user module's width
    gripper_dim=7,                # the action's gripper index
    operator_dims=None,           # the stick constrains every commanded joint
    reference_step=qdot_full,     # FRS: rad/s at a full deflection
    pinned_steps=n_action_steps,  # the executed prefix
    adapter_user=..., adapter_matrix=...,
)
```

| arm | mode | knobs |
| --- | --- | --- |
| `teleop` | no policy | — |
| `floor` | `none` | prompt `"do something"` |
| `ceiling` | `none` | the task prompt |
| `fc` | `flow_control` | `tau`, `lam`, `ic_gain=1.0`, command pre-normalized |
| `frs` | `frs` | `n_reversal_steps`, `pinned_steps` |
| `mars_fixed` | `frs` | + `adapter_matrix` = `J⁺ M J` at the start pose |
| `mars_learned` | `frs` | + `adapter_user` = a `user.pt` |

The three `src/` additions this needs (`reference_step` and `gripper_dim` in `SteeringConfig`, the FRS core
keeping the full action statistics) are additive and default-preserving; the LIBERO and cursor suites pin
that.

## The Jacobian

`jacobian.py` ports the validated chain from the previous project's `inverse_kinematics.py`: the
mujoco_menagerie `panda.xml` link transforms, the flange offset, the geometric Jacobian with its angular
rows, and the damped least-squares solve with `1e-2` damping. Do not hand-derive a DH table; that project
records a hardware fault caused by one. The port keeps the comments that name the fault, because they are the
reason the constants are what they are.

## Records

The same schema as `../libero/`: `trials.jsonl` plus `trial_XXX.npz`, with LIBERO's keys, so
`../libero/notebooks/analyze.py` reads a session without changes. `config.yaml` is flat, with `mode`,
`tau`, `n_reversal_steps`, `reversal_adapter`, and `prompt` at the top level, which is what `method_of`
reads.

## Not here

- No task module: no pi0.5 trainer exists, and the two tasks below are pick-and-place.
- No `lerobot.robots` class: c3po is the transport.
- No ROS, no pinocchio, no async inference.

## The tasks

1. **Two blocks**, pick and place. The arm starts from the same joint configuration every trial, so trials
   are paired when the blocks are also placed the same way.
2. **Hotpot**, after the four methods work: a pot in the middle, plates of toy ingredients around it, and the
   operator steers the VLA to put a prompted ingredient in the pot. No liquids, nothing hot.
