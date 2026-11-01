"""Operator steering of the goal-free cursor flow policy.

Binds the shared-autonomy modes of lerobot's pi0.5 steering
(src/lerobot/policies/pi05/steering.py), with the same names
and knobs so cursor and LIBERO results can be compared:

    policy        the policy alone
    teleop        the command alone moves the cursor, every step
    flow_control  Kao et al. 2026: the command, at the noise prior's scale
                  (`ic_gain` sigma for a full push), is written into the first
                  `tau` chunk steps of x_t at the start of each of the first `lam`
                  (lambda) Euler steps; the flow itself runs unmodified. lambda 1
                  writes the initial condition only (Kao et al.'s method), lambda 0
                  writes nothing (the policy alone), lambda > 1 keeps re-writing as
                  the flow proceeds (lerobot's pi0.5 shared_flow_control, whose
                  "tau" is this lambda, at tau 16 with the command in action units:
                  ic_gain = action_gain(policy)); the recorded study ran that at
                  lambda 1
    frs           Flow Reversal Steering (Tang et al. 2026): the command's chunk
                  is integrated backward `n_reverse_steps` steps, then forward
                  again (LeRobot: shared_flow_reversal_steering); a partial
                  reversal is this module's own variant; `pinned_steps` steers
                  only the reference's first chunk steps and hands the rest of the
                  latent back to the policy (Tang et al., App. D; None = the whole
                  chunk, what the study and the grid ran); with a reversal adapter
                  the backward pass follows the adapted field (MARS)

The command's chunk moves at constant velocity in the commanded direction,
MAX_SPEED at full deflection. In the shared modes the executed action is always
the policy's own output, and a command inside DEADBAND means the plain policy.
The command is read once per chunk, as in LeRobot.

A reversal adapter (`ReversalAdapter`) has up to two modules that change the
policy's velocity v at every Euler step:

    task module   v -> A(x_t, t, v, obstacles) v + b(...), in BOTH directions: it steers
                  the frozen policy's chunk distribution into a task's (around obstacles)
    user module   v -> U(x_t, t) v, in the BACKWARD pass only: it undoes the user's
                  command corruption (MARS's fixed F = M, `ReversalAdapter(matrix=F)`,
                  or a trained `UserModule`)

so the forward field is task(v) and the backward field user(task(v)). The two are
trained apart and composed here. Flow time runs from t=0 (noise) to t=1 (action),
the reverse of pi0.5.

The methods themselves are lerobot's: the cursor's chunks come from the same
`FlowControlPolicy` and `FlowReversalSteeringPolicy` wrappers the robot (pi0.5)
uses, driving the cursor policy through pi0.5's sampler interface
(`CursorSampler`, over `FlowPolicy.sample_chunk`); the Euler flows and the user
and task modules are lerobot's steering core too. This module binds them to the
cursor policy's observation, its two action dims and its screen units.
"""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.pi05.steering import (AdaptedField, FlowControlPolicy, FlowReversalSteeringPolicy,
                                             LearnedTaskModule, LearnedUserModule, integrate, load_adapter_module,
                                             task_velocity_hook)

from .readout import DEADBAND, corrupt_chunk, is_active

MODES = ("policy", "teleop", "flow_control", "frs")
GAIN_MODES = ("step", "time")  # how speed_gain slows the cursor: smaller steps, or each planned step over more control steps
MAX_SPEED = 1.2  # screens/s at full deflection, about the demos' peak reach speed


def method_name(mode, has_adapter=False):
    """The method token used in configs, logs and figures: frs with an adapter is mars."""
    return "mars" if mode == "frs" and has_adapter else mode


# ---------------------------------------------------------------- units and observations


def to_action(deflection, policy, dt=0.02, max_speed=MAX_SPEED):
    """Deflections in [-1, 1] (full deflection = max_speed) to the policy's normalized action units."""
    return deflection * (max_speed * dt) / policy.action_scale


def to_deflection(action, policy, dt=0.02, max_speed=MAX_SPEED):
    """The inverse of `to_action`: normalized policy actions to deflections per step."""
    return action * policy.action_scale / (max_speed * dt)


def bci_chunk(intended, corruption, sigma, policy, dt=0.02, max_speed=MAX_SPEED):
    """The BCI's version of an intended chunk (normalized policy units, [B, H, 2]): each
    step as the operator's deflection, through the readout (`corruption` M, noise
    std `sigma` [B]), and back. The torch twin of what an expert_chunk agent delivers."""
    return to_action(corrupt_chunk(to_deflection(intended, policy, dt, max_speed), corruption, sigma),
                     policy, dt, max_speed)


def _f32(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


def observation(layouts, pos, hist, device="cpu", goal_given=False):
    """The policy's observation of a batch of trials: cursor positions [B, 2], their last
    displacements [B, h, 2], the targets, the goal flags (all False unless goal_given), and the
    task's obstacles (which the policy ignores and a task module reads)."""
    goal = layouts.goal_mask() if goal_given else np.zeros(layouts.mask.shape, dtype=bool)
    return dict(pos=_f32(pos, device), hist=_f32(hist, device), targets=_f32(layouts.targets, device),
                mask=torch.as_tensor(layouts.mask, device=device), goal=torch.as_tensor(goal, device=device),
                obstacles=_f32(layouts.obstacles, device), obstacle_mask=torch.as_tensor(layouts.obstacle_mask, device=device))


def hide_goal(obs):
    """The same observation with every goal flag cleared."""
    return {**obs, "goal": torch.zeros_like(obs["goal"])}


# ---------------------------------------------------------------- reversal adapters


def UserModule(horizon=16, d=256, tdim=64, n_dims=2, time_embedding="mars", step_rank=None, cross_step_rank=0,
               cross_step_dims=None):
    """The core's `LearnedUserModule` with the cursor's dims and defaults: U(x_t, t), one 2x2 matrix per chunk
    step applied to the policy's velocity in the backward pass; it undoes one user's command corruption.
    Initialized to the identity. `step_rank` and `cross_step_rank` (`cross_step_dims`) set its flexibility as the
    core documents; `velocity(x, t, v)` applies the whole correction (the core's `apply_user_module`),
    `forward(x, t)` gives the per-step matrices alone [B, H, 2, 2]."""
    return LearnedUserModule(horizon, n_dims=n_dims, d=d, tdim=tdim, time_embedding=time_embedding, step_rank=step_rank,
                             cross_step_rank=cross_step_rank, cross_step_dims=cross_step_dims)


# A(x_t, t, v, obstacles) v + b(...): the core's task module (two dims by default), bound to each observation
TaskModule = LearnedTaskModule


class ReversalAdapter(nn.Module):
    """The reversal adapter: a user part (`matrix`, MARS's fixed F, one 2x2 or one per trial [B, 2, 2], and/or
    `user`, a UserModule or any (x, t) -> [B, H, 2, 2] callable) and/or a task module, composed at test time.
    `adapt` applies the task module in both directions and the user part in the backward pass: forward task(v),
    backward user(task(v)). `field(obs)` is the core's `AdaptedField`, which `adapt` and the flows use; `load`
    reads a checkpoint in lerobot's `save_adapter_module` format (one module, `kind` user or task)."""

    def __init__(self, user=None, task=None, matrix=None):
        super().__init__()
        self.user, self.task = user, task
        self.matrix = None if matrix is None else torch.as_tensor(matrix, dtype=torch.float32)

    @property
    def kind(self):
        user = self.user is not None or self.matrix is not None
        return {(True, True): "composed", (True, False): "user", (False, True): "task", (False, False): "none"}[
            (user, self.task is not None)]

    def field(self, obs):
        """The core's `AdaptedField` for this observation: the fixed matrix on its device, the user module, the
        task module bound to `obs`."""
        matrix = None if self.matrix is None else self.matrix.to(obs["pos"].device)
        task = self.task.bind(obs) if self.task is not None and hasattr(self.task, "bind") else self.task
        return AdaptedField(matrix, self.user, task, n_dims=2)

    def adapt(self, obs, x, t, v, backward):
        return self.field(obs).adapt(x, t, v, backward)

    @classmethod
    def load(cls, path, device="cpu"):
        module, kind, _ = load_adapter_module(path, device)
        return cls(**{kind: module}).to(device).eval()

def as_adapter(adapter, device="cpu"):
    """None or a ReversalAdapter as is; a TaskModule in the task slot; a UserModule or any (x, t) callable in the
    user slot; a matrix F ([2, 2] or [B, 2, 2]) as the fixed matrix."""
    if adapter is None or isinstance(adapter, ReversalAdapter):
        return adapter
    if isinstance(adapter, TaskModule):
        return ReversalAdapter(task=adapter)
    if callable(adapter):
        return ReversalAdapter(user=adapter)
    return ReversalAdapter(matrix=torch.as_tensor(adapter, dtype=torch.float32, device=device))

def _field(adapter, obs, device):
    """The core's form of an adapter (see `as_adapter`) for this observation, or None."""
    adapter = as_adapter(adapter, device)
    return None if adapter is None else adapter.field(obs)


# ---------------------------------------------------------------- flows and steered chunks


def flow(policy, obs, x, t0, h, n, adapter=None):
    """n Euler steps of signed size h from flow time t0 through the policy's field, adapted by
    `adapter` (the task module in both directions, the user module when h < 0). Differentiable,
    for training; `reverse_flow` and `forward_flow` are its no-grad forms. The core's `integrate`."""
    return integrate(x, lambda x_t, t: policy(obs, x_t, t), t0, h, n, _field(adapter, obs, x.device), backward=h < 0,
                     grad=True, batch_time=True)


@torch.no_grad()
def reverse_flow(policy, obs, x, n, steps, adapter=None):
    """Euler-integrate the flow backward from a chunk (t=1) toward noise: n steps of 1/steps."""
    return flow(policy, obs, x, 1.0, -1.0 / steps, n, adapter)


@torch.no_grad()
def forward_flow(policy, obs, x, t0, t1, steps, adapter=None):
    """Euler-integrate the (task-adapted) flow from t0 to t1 in `steps` steps."""
    return flow(policy, obs, x, t0, (t1 - t0) / steps, steps, adapter)


def action_gain(policy, dt=0.02, max_speed=MAX_SPEED):
    """The `ic_gain` that writes the command in the policy's action units (a full push = max_speed * dt), as the
    lab's variant did: `to_action` for a deflection of 1."""
    return float(max_speed * dt / float(policy.action_scale))


class CursorSampler:
    """The cursor policy behind pi0.5's sampler interface (`FlowPolicy.sample_chunk`), so lerobot's steering
    wrappers drive it exactly as they drive pi0.5: its `config` gives the flow's Euler steps, the chunk, the two
    action dims, the steps executed per replan and the flow time of noise (0, the cursor's convention)."""

    def __init__(self, policy, steps=10, execute=8):
        self.policy = policy
        self.config = SimpleNamespace(num_inference_steps=steps, chunk_size=policy.horizon, max_action_dim=2,
                                      n_action_steps=execute, noise_time=0.0)

    def reset(self):
        pass

    def predict_action_chunk(self, obs, **hooks):
        """A normalized chunk [B, H, 2] (pi0.5's hooks: noise_fn, flow_start_time, num_forward_steps, x_t_hook,
        velocity_hook; and noise)."""
        return self.policy.sample_chunk(obs, self.config.num_inference_steps, **hooks)


def steering_wrapper(sampler, mode, tau=5, n_reverse_steps=None, dt=0.02, max_speed=MAX_SPEED, ic_gain=1.0, lam=1,
                     pinned_steps=None):
    """lerobot's wrapper for a shared mode on the cursor's sampler, every action dim commanded (None in modes
    policy and teleop). frs's reference is the command held at constant velocity over the chunk, or a command per
    step, in action units (`to_action`)."""
    if mode == "flow_control":
        return FlowControlPolicy(sampler, None, tau=tau, lam=lam, ic_gain=ic_gain, deadband=DEADBAND, dims=slice(None))
    if mode == "frs":
        policy, horizon = sampler.policy, sampler.config.chunk_size

        def reference(commands):
            per_step = commands if commands.dim() == 3 else commands[:, None].expand(-1, horizon, -1)
            return to_action(per_step, policy, dt, max_speed)

        return FlowReversalSteeringPolicy(sampler, None, n_reversal_steps=n_reverse_steps, operator_dims=None,
                                          pinned_steps=pinned_steps, deadband=DEADBAND, reference_fn=reference)
    return None


def _chunk(sampler, wrapper, obs, commands, field, noise=None):
    """A normalized chunk [B, H, 2]: the wrapper's for the commands, or the policy's own (with the task module)."""
    if wrapper is None:
        hook = task_velocity_hook(field)
        return sampler.predict_action_chunk(obs, noise=noise, **({} if hook is None else {"velocity_hook": hook}))
    return wrapper.steered_chunk(obs, commands, field=field)[0]


@torch.no_grad()
def steered_chunk(policy, obs, command, mode, tau=5, n_reverse_steps=None, steps=10, dt=0.02,
                  max_speed=MAX_SPEED, adapter=None, reference=None, ic_gain=1.0, lam=1, pinned_steps=None):
    """One action chunk [B, H, 2] in screen units per step, for commands [B, 2]: lerobot's steering wrapper for
    `mode` on this policy (`steering_wrapper`), its adapter bound to `obs`. The reference the shared modes steer
    toward is the command held constant over the chunk, or `reference` [B, H, 2] (a deflection per step, e.g. the
    chunk an expert_chunk agent means). Rows whose command (or whole reference) is inside the deadband get the
    plain policy chunk, from the same noise."""
    sampler = CursorSampler(policy, steps)
    wrapper = steering_wrapper(sampler, mode, tau, n_reverse_steps, dt, max_speed, ic_gain, lam, pinned_steps)
    chunk = _chunk(sampler, wrapper, obs, command if reference is None else reference,
                   _field(adapter, obs, command.device))
    return chunk * policy.action_scale


class SharedController:
    """Turns operator commands into cursor actions for a batch of trials.

    Each control step: `act(command)` returns the actions [B, 2] to apply, then
    `record(displacement)` feeds back how far the cursor actually moved (the
    policy's velocity history). The policy is never told the goal.

    `speed_gain` scales the cursor's speed in every mode (policy, steered, and
    teleop), so 0.5 halves it on screen; `gain_mode` says how. "step" scales
    every executed step by the gain and gives the policy's history the
    displacement divided by it, so the policy plans as if it moved at its normal
    speed while its position advances at the gain's (the study's controller; on
    a curved route the two disagree and the policy loses the route). "time"
    plays every planned step over 1/gain control steps, so the cursor follows
    the planned path at the gain's speed and the history holds one entry per
    planned step, the motion the policy was trained on (chunks then come 1/gain
    times less often per second; with 1/gain not an integer a chunk's last
    control step is shorter). At gain 1 the two are the same. `adapter` is the
    reversal adapter (see `as_adapter`), or None; its task module shapes every
    chunk, its user module the reversals of frs. `ic_gain` is flow_control's
    scale of the command in the noise prior, `lam` the Euler steps it is written at,
    `pinned_steps` the chunk steps frs's reference steers (None = all).
    """

    def __init__(self, policy, layouts, mode, tau=5, n_reverse_steps=None, steps=10, execute=8,
                 dt=0.02, max_speed=MAX_SPEED, speed_gain=1.0, adapter=None, device="cpu", gain_mode="step",
                 ic_gain=1.0, lam=1, pinned_steps=None):
        assert mode in MODES, mode
        assert speed_gain > 0, speed_gain
        assert gain_mode in GAIN_MODES, gain_mode
        self.gain, self.gain_mode, self.adapter = speed_gain, gain_mode, as_adapter(adapter, device)
        self.ic_gain, self.lam, self.pinned_steps = ic_gain, lam, pinned_steps
        self.policy, self.mode, self.tau, self.n_reverse_steps = policy, mode, tau, n_reverse_steps
        self.steps, self.execute, self.dt, self.max_speed, self.device = steps, execute, dt, max_speed, device
        self.layouts = layouts
        self.pos = layouts.start.copy()
        self.hist = np.zeros((len(layouts), policy.hist, 2))
        self.chunk = None
        # control steps per chunk, and the count of them; == cycle means a new chunk at the next act
        self.cycle = execute if gain_mode == "step" else int(np.ceil(execute / speed_gain - 1e-9))
        self.t = self.cycle
        self.parts = []                       # the planned steps the last control step played, as (index, share)
        self.moved, self.played = {}, {}      # per planned step under way: the displacement so far, the share so far
        self.chunks = 0
        self.steered_chunks = np.zeros(len(layouts), dtype=int)
        self.sampler = CursorSampler(policy, steps, execute)  # lerobot's wrapper for the mode drives the policy
        self.wrapper = steering_wrapper(self.sampler, mode, tau, n_reverse_steps, dt, max_speed, ic_gain, lam,
                                        pinned_steps)

    def steps_of(self, t):
        """The planned steps control step t of a chunk plays, as (index, share of the step) pairs: in mode
        "step" planned step t scaled by the gain, in mode "time" the chunk's span [t, t + 1) x gain."""
        if t >= self.cycle:
            return []
        if self.gain_mode == "step":
            return [(t, self.gain)]
        a, b = t * self.gain, min((t + 1) * self.gain, self.execute)
        shares = [(i, min(b, i + 1) - max(a, i)) for i in range(int(a + 1e-9), int(np.ceil(b - 1e-9)))]
        return [(i, s) for i, s in shares if s > 1e-9]

    def observation(self, goal_given=False):
        """The policy's observation of the current state; the goal is hidden unless asked for."""
        return observation(self.layouts, self.pos, self.hist, self.device, goal_given)

    @torch.no_grad()
    def sample_chunk(self, obs, z=None):
        """A chunk [B, H, 2] (normalized units) from the task-adapted forward flow, from noise z."""
        return _chunk(self.sampler, None, obs, None, _field(self.adapter, obs, obs["pos"].device), noise=z)

    def act(self, command, reference=None):
        """Actions [B, 2] for this control step. `reference` [B, H, 2] (a deflection per step)
        replaces the constant-velocity reference at a replan, see `steered_chunk`."""
        if self.t >= self.cycle:  # a new chunk
            self.t = 0
            if self.mode != "teleop":
                obs = self.observation()
                commands = _f32(command if reference is None else reference, self.device)
                with torch.no_grad():
                    chunk = _chunk(self.sampler, self.wrapper, obs, commands, _field(self.adapter, obs, self.device))
                self.chunk = (chunk * self.policy.action_scale).cpu().numpy()
                self.chunks += 1
                if self.mode != "policy":
                    # the trials steering: the command outside the deadband, or any step of the reference chunk
                    active = is_active(np.asarray(command if reference is None else reference))
                    self.steered_chunks += active if reference is None else active.any(-1)
        self.parts = self.steps_of(self.t)
        self.t += 1
        if self.mode == "teleop":
            return command * (self.max_speed * self.dt * self.gain)
        return sum(share * self.chunk[:, i] for i, share in self.parts)

    def record(self, displacement):
        """How far the cursor moved this control step: the position, and the policy's history, one entry per
        planned step (the displacement attributed to it, scaled to the whole step) once the step is over."""
        self.pos = self.pos + displacement
        total = sum(share for _, share in self.parts)
        for i, share in self.parts:
            self.moved[i] = self.moved.get(i, 0) + displacement * (share / total)
            self.played[i] = self.played.get(i, 0) + share
        ongoing = {i for i, _ in self.steps_of(self.t)}  # the next control step continues these
        for i in sorted(set(self.moved) - ongoing):
            self.hist = np.concatenate([self.hist[:, 1:], (self.moved.pop(i) / self.played.pop(i))[:, None]], 1)

    def planned(self):
        """Rest of the current chunk [B, n, 2] as positions, for display (empty in teleop)."""
        if self.chunk is None or self.mode == "teleop" or self.t >= self.cycle:
            return np.zeros((len(self.pos), 0, 2))
        moves = [sum(share * self.chunk[:, i] for i, share in self.steps_of(t)) for t in range(self.t, self.cycle)]
        return self.pos[:, None] + np.cumsum(np.stack(moves, 1), 1)
