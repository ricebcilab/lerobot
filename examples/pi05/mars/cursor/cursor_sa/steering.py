"""Operator steering of the goal-free cursor flow policy.

Binds the shared-autonomy modes of lerobot's pi0.5 steering
(src/lerobot/policies/steering/), with the same names
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
                  ic_gain = to_action(1, policy), a full deflection's); the recorded study
                  ran that at lambda 1
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

A reversal adapter (lerobot's `AdapterModules`, two dims by default; `as_adapter`
coerces a matrix, a module or None into one) has up to two modules that change the
policy's velocity v at every Euler step:

    task module   v -> A(x_t, t, v, obstacles) v + b(...), in BOTH directions: it steers
                  the frozen policy's chunk distribution into a task's (around obstacles);
                  a `LearnedTaskModule`, bound to each observation
    user module   v -> U(x_t, t) v, in the BACKWARD pass only: it undoes the user's
                  command corruption (MARS's fixed F = M, `AdapterModules(matrix=F)`,
                  or a trained `LearnedUserModule(16, n_dims=2)`)

so the forward field is task(v) and the backward field user(task(v)). The two are
trained apart and composed here. Flow time runs from t=0 (noise) to t=1 (action),
the reverse of pi0.5.

The methods themselves are lerobot's: every chunk comes from lerobot's
`SteeringPolicy` (mode none for the policy alone, flow_control, frs) over the
cursor policy (`CursorFlowPolicy`, `as_policy`), which offers pi0.5's sampler
interface, so it is the same `FlowControlSteering` and `FlowReversalSteering`
cores the robot (pi0.5) uses that steer it; the policy's postprocessor
(`make_cursor_flow_pre_post_processors`) turns the normalized chunks into screen
units. The Euler flows and the user and task modules are lerobot's steering core
too. This module binds them to the cursor policy's observation, its two action dims
and its screen units. A steering policy's pipelines on the cursor are
`cursor_steering_pre_post_processors`: lerobot's, its scene-tokens step reading the
pictures the cursor's render step draws from the trials' layouts.
"""

import numpy as np
import torch

from lerobot.policies.cursor_flow import make_cursor_flow_pre_post_processors
from lerobot.policies.cursor_flow.processor_cursor_flow import CursorSceneRenderProcessorStep
from lerobot.policies.steering import (AdapterModules, Flow, SceneEncoder, SteeringConfig, SteeringPolicy,
                                       SteeringSceneTokensProcessorStep, as_adapter, make_steering_pre_post_processors)

from lerobot.envs.cursor import render_views
from .policy import as_policy
from lerobot.teleoperators.readout import DEADBAND, Readout, is_active

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
    return to_action(Readout(corruption).chunk(to_deflection(intended, policy, dt, max_speed), sigma),
                     policy, dt, max_speed)


def _f32(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


def static_arrays(layouts, device="cpu", sources=None):
    """What every observation of `layouts` repeats, on `device`: the targets, the mask, the goal flags given and hidden
    (`goal`: {True: ..., False: ...}), the obstacles and their mask, and `scene`, the layouts' scene arrays of `sources`
    (None: all of them; the arrays keep their dtype, uint8 render or float16 tokens), None when the layouts have no
    scene or `sources` keeps none of it."""
    scene = None if layouts.scene is None else {k: torch.as_tensor(v, device=device) for k, v in layouts.scene.items()
                                                if sources is None or k in sources}
    return dict(targets=_f32(layouts.targets, device), mask=torch.as_tensor(layouts.mask, device=device),
                goal={False: torch.as_tensor(np.zeros(layouts.mask.shape, dtype=bool), device=device),
                      True: torch.as_tensor(layouts.goal_mask(), device=device)},
                obstacles=_f32(layouts.obstacles, device),
                obstacle_mask=torch.as_tensor(layouts.obstacle_mask, device=device),
                scene=None if sources is not None and not scene else scene)


def observation(layouts, pos, hist, device="cpu", goal_given=False, scene=None, static=None):
    """The policy's observation of a batch of trials: cursor positions [B, 2], their last
    displacements [B, h, 2], the targets, the goal flags (all False unless goal_given), and the
    task's obstacles (which the policy ignores and a task module reads) and, when the layouts have one, their scene;
    `scene` (source -> array, e.g. a scene provider's tokens of this step) joins the layouts' own. `static` is
    `static_arrays(layouts, device, ...)` made once for many observations (a controller's); without it they are
    made here, every layout scene array included."""
    static = static_arrays(layouts, device) if static is None else static
    obs = dict(pos=_f32(pos, device), hist=_f32(hist, device), targets=static["targets"], mask=static["mask"],
               goal=static["goal"][bool(goal_given)], obstacles=static["obstacles"], obstacle_mask=static["obstacle_mask"])
    if static["scene"] is not None or scene is not None:
        obs["scene"] = {**(static["scene"] or {}), **{k: torch.as_tensor(v, device=device) for k, v in (scene or {}).items()}}
    return obs


def view_scene_fn(module, encoder):
    """The scene provider (`SharedController(scene_fn=...)`) of a vlm_views task module: (layouts, pos [B, 2], active
    [B] bool or None = all) -> {"vlm_views": float16 [B, 512, 2048]} on the encoder's device, the running trials'
    two views at their positions (`render_views` at the module's wrist_span and image_size) encoded by `encoder`
    (`SceneEncoder.view_tokens`; the module applies its own reduction to these raw tokens), zeros for the finished rows, nothing
    rendered when none runs. Leaves torch's CPU and CUDA generators as it found them (the encoder's lazy first load
    builds a model from them)."""

    def scene(layouts, pos, active=None):
        rows = np.arange(len(layouts)) if active is None else np.flatnonzero(active)
        if not len(rows):
            return {"vlm_views": torch.zeros(len(layouts), module.scene_tokens, module.scene_dim, dtype=torch.float16,
                                             device=getattr(encoder, "device", "cpu"))}
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            views = render_views(layouts if active is None else layouts[rows], np.asarray(pos)[rows], module.wrist_span,
                                 module.image_size)
            tokens = SceneEncoder.view_tokens(views, encoder)
        if len(rows) == len(layouts):
            return {"vlm_views": tokens}
        out = torch.zeros(len(layouts), *tokens.shape[1:], dtype=tokens.dtype, device=tokens.device)
        out[torch.as_tensor(rows, device=tokens.device)] = tokens
        return {"vlm_views": out}

    return scene


def hide_goal(obs):
    """The same observation with every goal flag cleared."""
    return {**obs, "goal": torch.zeros_like(obs["goal"])}


# ---------------------------------------------------------------- flows and steered chunks


def flow(policy, obs, x, t0, h, n, adapter=None):
    """n Euler steps of signed size h from flow time t0 through the policy's field, adapted by
    `adapter` (see `as_adapter`; the task module in both directions, the user module when h < 0),
    bound to `obs`. Differentiable, for training; `reverse_flow` and `forward_flow` are its no-grad
    forms. The core's `Flow.integrate`."""
    adapter = as_adapter(adapter, x.device)
    return Flow(lambda x_t, t: policy(obs, x_t, t), None if adapter is None else adapter.field(obs), noise_time=0.0,
                batch_time=True).integrate(x, t0, h, n, backward=h < 0, grad=True)


@torch.no_grad()
def reverse_flow(policy, obs, x, n, steps, adapter=None):
    """Euler-integrate the flow backward from a chunk (t=1) toward noise: n steps of 1/steps."""
    return flow(policy, obs, x, 1.0, -1.0 / steps, n, adapter)


@torch.no_grad()
def forward_flow(policy, obs, x, t0, t1, steps, adapter=None):
    """Euler-integrate the (task-adapted) flow from t0 to t1 in `steps` steps."""
    return flow(policy, obs, x, t0, (t1 - t0) / steps, steps, adapter)


def cursor_steering_pre_post_processors(config):
    """The steering policy's pipelines on the cursor: lerobot's (`make_steering_pre_post_processors(config)`), the
    scene-tokens step of a vlm_image / vlm_prefix task module preceded by the cursor's render step
    (`CursorSceneRenderProcessorStep` at the tokens step's render_size), which draws the trials' layouts
    (`observation.layouts`) as the pictures the tokens step encodes. Without a tokens step they are lerobot's as
    they are."""
    pre, post = make_steering_pre_post_processors(config)
    steps = list(pre.steps)
    at = next((i for i, step in enumerate(steps) if isinstance(step, SteeringSceneTokensProcessorStep)), None)
    if at is not None:
        pre.steps = [*steps[:at], CursorSceneRenderProcessorStep(render_size=steps[at].render_size), *steps[at:]]
    return pre, post


def _steering(policy, config, adapter, device):
    """lerobot's `SteeringPolicy` for `config` (`SharedController.steering_config`) over `policy` (a
    `CursorFlowPolicy`), its adapter `adapter` (see `as_adapter`; None: none), and the policy's postprocessor
    (`make_cursor_flow_pre_post_processors`), which the steering policy's FRS reference reads and which turns its
    normalized chunks into screen units: (steering policy, postprocessor)."""
    postprocessor = make_cursor_flow_pre_post_processors(policy.config)[1]
    steering = SteeringPolicy(config, base_policy=policy, postprocessor=postprocessor)
    adapter = as_adapter(adapter, device)
    steering.adapter = AdapterModules() if adapter is None else adapter
    return steering, postprocessor


@torch.no_grad()
def steered_chunk(policy, obs, command, mode, tau=5, n_reverse_steps=None, steps=10, dt=0.02,
                  max_speed=MAX_SPEED, adapter=None, reference=None, ic_gain=1.0, lam=1, pinned_steps=None):
    """One action chunk [B, H, 2] in screen units per step (on the CPU, where the policy's postprocessor leaves it),
    for commands [B, 2]: one call of lerobot's `SteeringPolicy` for `mode` (`SharedController.steering_config`) on
    this policy (`as_policy`: a `FlowPolicy` or a `CursorFlowPolicy`, sampled in `steps` Euler steps), its adapter
    bound to `obs`, the chunk unnormalized by the policy's postprocessor. The reference the shared modes steer
    toward is the command held constant over the chunk, or `reference` [B, H, 2] (a deflection per step, e.g. the
    chunk an expert_chunk agent means). Rows whose command (or whole reference) is inside the deadband get the
    plain policy chunk, from the same noise; modes policy and teleop give the policy's own chunk."""
    steering, postprocessor = _steering(
        as_policy(policy, num_inference_steps=steps),
        SharedController.steering_config(mode, tau, n_reverse_steps, dt, max_speed, ic_gain, lam, pinned_steps,
                                         command.device),
        adapter, command.device)
    return postprocessor(steering.predict_action_chunk({**obs, "command": command if reference is None else reference}))


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
    `pinned_steps` the chunk steps frs's reference steers (None = all). `scene_fn` (layouts, pos, active) -> scene
    arrays (e.g. `view_scene_fn`) adds a scene of the current positions to every observation, asked for again only
    when the positions or `active` (None, or a bool [B] of the trials still running, which the caller sets) have
    changed, so observations at one position share it.

    The chunks are lerobot's `SteeringPolicy`'s (`self.steering`, `steering_config`'s mode: none for the policy
    alone, flow_control or frs) over the cursor policy (`self.policy`, `as_policy` of `policy`, a `CursorFlowPolicy`
    or a bare `FlowPolicy`, with `steps` Euler steps and `execute` steps per chunk on a config of its own), the
    adapter in the steering policy's adapter slot, every chunk unnormalized by the policy's postprocessor
    (`self.postprocessor`) into screen units. Teleop's actions are the command's, but its steering policy (mode
    none) still serves `sample_chunk`, the expert_chunk agent's intent.
    """

    def __init__(self, policy, layouts, mode, tau=5, n_reverse_steps=None, steps=10, execute=8,
                 dt=0.02, max_speed=MAX_SPEED, speed_gain=1.0, adapter=None, device="cpu", gain_mode="step",
                 ic_gain=1.0, lam=1, pinned_steps=None, scene_fn=None):
        assert mode in MODES, mode
        assert speed_gain > 0, speed_gain
        assert gain_mode in GAIN_MODES, gain_mode
        self.gain, self.gain_mode, self.adapter = speed_gain, gain_mode, as_adapter(adapter, device)
        self.policy, self.mode = as_policy(policy, num_inference_steps=steps, n_action_steps=execute), mode
        self.execute, self.dt, self.max_speed, self.device = execute, dt, max_speed, device
        self.layouts = layouts
        # what every observation repeats, on the device once (`observation` adds the positions and histories), with
        # the layouts' scene arrays the task module reads: none without one, every one for a module without a source
        task = None if self.adapter is None else self.adapter.task
        source = getattr(task, "token_source", None)
        sources = () if task is None else tuple(layouts.scene or ()) if source is None else (source,)
        self._static = static_arrays(layouts, device, sources)
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
        self.steering, self.postprocessor = _steering(
            self.policy, self.steering_config(mode, tau, n_reverse_steps, dt, max_speed, ic_gain, lam, pinned_steps,
                                              device), self.adapter, device)
        if self.steering.config.mode != "none":
            _ = self.steering.core  # built now: its knobs are checked when the controller is made
        self.scene_fn, self.active = scene_fn, None
        self._scene = None                    # the last scene asked for: (positions, active, arrays)

    @staticmethod
    def steering_config(mode, tau=5, n_reverse_steps=None, dt=0.02, max_speed=MAX_SPEED, ic_gain=1.0, lam=1,
                        pinned_steps=None, device="cpu"):
        """The `SteeringConfig` of a controller mode: modes policy and teleop are the steering policy's mode none
        (the policy's own chunk through the task module), flow_control and frs their own, every action dim
        commanded (the cursor's two), the readout's deadband; frs's reference is the command held at `max_speed *
        dt` per step, in action units. The flow's Euler steps and the chunk are the base policy's config's
        (`as_policy`)."""
        return SteeringConfig(mode="none" if mode in ("policy", "teleop") else mode, tau=tau, lam=lam, ic_gain=ic_gain,
                              n_reversal_steps=n_reverse_steps, pinned_steps=pinned_steps, deadband=DEADBAND,
                              n_dims=2, max_speed=max_speed, dt=dt, device=str(device))

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
        """The policy's observation of the current state (`observation`, the static arrays uploaded once in
        `__init__`); the goal is hidden unless asked for. With a `scene_fn` its scene of these positions joins it, reused while the
        positions and `active` are unchanged."""
        scene = None
        if self.scene_fn is not None:
            cached = self._scene
            if (cached is None or not np.array_equal(cached[0], self.pos) or (cached[1] is None) != (self.active is None)
                    or (self.active is not None and not np.array_equal(cached[1], self.active))):
                active = None if self.active is None else np.array(self.active, copy=True)
                cached = self._scene = (self.pos.copy(), active, self.scene_fn(self.layouts, self.pos, self.active))
            scene = cached[2]
        return observation(self.layouts, self.pos, self.hist, self.device, goal_given, scene, static=self._static)

    @torch.no_grad()
    def sample_chunk(self, obs, z=None):
        """A chunk [B, H, 2] in screen units per step (on the CPU) from the task-adapted forward flow, from noise z
        (None: drawn): the steering policy's chunk for no command (the policy's own), unnormalized by the policy's
        postprocessor."""
        return self.postprocessor(self.steering.predict_action_chunk(obs, noise=z))

    def act(self, command, reference=None):
        """Actions [B, 2] for this control step. `reference` [B, H, 2] (a deflection per step)
        replaces the constant-velocity reference at a replan, see `steered_chunk`."""
        if self.t >= self.cycle:  # a new chunk
            self.t = 0
            if self.mode != "teleop":
                obs = self.observation()
                commands = _f32(command if reference is None else reference, self.device)
                with torch.no_grad():
                    chunk = self.steering.predict_action_chunk({**obs, "command": commands})
                self.chunk = self.postprocessor(chunk).numpy()
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
