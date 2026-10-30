# Frozen reference for equivalence tests: a frozen copy of the original cursor project's cursor_sa/steering.py at
# ricebcirig commit bd2a48f2 (analysis/nishal/cursor_control_shared_autonomy; the user module's step_rank and
# cross_step_rank knobs), with its relative imports made absolute and its one machine path dropped. Do not edit; tests
# compare the lerobot steering core's ranked user module (src/lerobot/policies/pi05/steering.py) against it.

"""Operator steering of the goal-free cursor flow policy.

Mirrors the shared-autonomy modes of the lab's LeRobot pi0.5 setup
(src/lerobot/policies/pi05/steering.py in this repo), with the same names
and knobs so cursor and LIBERO results can be compared:

    policy        the policy alone
    teleop        the command alone moves the cursor, every step
    flow_control  Kao et al. 2026: the command, at the noise prior's scale
                  (`ic_gain` sigma for a full push), is written into the first
                  `tau` chunk steps of x_t at the start of each of the first `lam`
                  (lambda) Euler steps; the flow itself runs unmodified. lambda 1
                  writes the initial condition only (Kao et al.'s method), lambda 0
                  writes nothing (the policy alone), lambda > 1 keeps re-writing as
                  the flow proceeds (the lab's LeRobot shared_flow_control, whose
                  "tau" is this lambda, at tau 16 with the command in action units:
                  ic_gain = action_gain(policy)); the recorded study ran that at
                  lambda 1
    frs           Flow Reversal Steering (Tang et al. 2026): the command's chunk
                  is integrated backward `n_reverse_steps` steps, then forward
                  again (LeRobot: shared_flow_reversal_steering); a partial
                  reversal is this module's own variant; with a reversal adapter
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
                  command corruption (`FixedAdapter` holds MARS's fixed F = M,
                  `UserModule` a trained one)

so the forward field is task(v) and the backward field user(task(v)). The two are
trained apart and composed here. Flow time runs from t=0 (noise) to t=1 (action),
the reverse of pi0.5.
"""

import numpy as np
import torch
import torch.nn as nn

from cursor_sa.channel import corrupt_chunk, is_active
from cursor_sa.policy import mlp, time_embedding

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
    step as the operator's deflection, through the input channel (`corruption` M, noise
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


def active_rows(command, reference=None):
    """Which trials are steering: the command [B, 2] is outside the deadband, or any step
    of the reference chunk [B, H, 2] is."""
    src = command if reference is None else reference
    active = is_active(src.cpu().numpy() if torch.is_tensor(src) else np.asarray(src))
    return active if reference is None else active.any(-1)


# ---------------------------------------------------------------- reversal adapters


class FixedAdapter:
    """MARS's fixed F: one 2x2 matrix (or one per trial, [B, 2, 2]) at every step and time."""

    def __init__(self, F, device="cpu"):
        self.F = torch.as_tensor(F, dtype=torch.float32, device=device)

    def __call__(self, x, t):
        B, H = x.shape[:2]
        return self.F.to(x.device).expand(B, 2, 2)[:, None].expand(B, H, 2, 2)


class UserModule(nn.Module):
    """U(x_t, t): one 2x2 matrix per chunk step, applied to the policy's velocity in the backward pass; it undoes
    one user's command corruption. Initialized to the identity. Two knobs set its flexibility:

    step_rank r (default None: every step's matrix free): each step's correction mixes r matrices shared by the
        chunk, A_s = I + sum_k c_sk B_k, with the coefficients c and the matrices B from (x_t, t); r = 4 spans
        every 2x2 matrix, as the default does.
    cross_step_rank q (default 0: none): a rank-q coupling between the chunk's steps on top of the per-step
        matrices, U V^T on the stacked chunk: step s adds U_s m with m = sum_s' V_s'^T v_s' (U, V [H, 2, q] from
        (x_t, t)); U starts at zero, so a fresh module is still the identity.

    `velocity(x, t, v)` applies the whole correction; `forward(x, t)` gives the per-step matrices alone [B, H, 2, 2]
    (all of it when cross_step_rank is 0)."""

    def __init__(self, horizon=16, d=256, tdim=64, step_rank=None, cross_step_rank=0):
        super().__init__()
        self.config = dict(horizon=horizon, d=d, tdim=tdim)
        if step_rank is not None or cross_step_rank:   # older checkpoints keep their config as it was
            self.config.update(step_rank=step_rank, cross_step_rank=cross_step_rank)
        self.horizon, self.tdim, self.step_rank, self.cross_step_rank = horizon, tdim, step_rank, cross_step_rank
        out = 4 * horizon if step_rank is None else 4 * step_rank + horizon * step_rank
        self.net = nn.Sequential(nn.Linear(2 * horizon + tdim, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, out))
        head = self.net[-1]
        if step_rank is None:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        else:   # the coefficients start at zero, the shared matrices do not (else neither would get a gradient)
            nn.init.zeros_(head.weight[4 * step_rank:])
            nn.init.zeros_(head.bias[4 * step_rank:])
        if cross_step_rank:
            self.cross = nn.Linear(d, 2 * horizon * 2 * cross_step_rank)   # U then V
            n_u = horizon * 2 * cross_step_rank
            nn.init.zeros_(self.cross.weight[:n_u])
            nn.init.zeros_(self.cross.bias[:n_u])

    def _hidden(self, x, t):
        return self.net[:-1](torch.cat([x.flatten(1), time_embedding(t, self.tdim)], -1))

    def _matrices(self, h):
        out, eye = self.net[-1](h), torch.eye(2, device=h.device)
        if self.step_rank is None:
            return eye + out.view(-1, self.horizon, 2, 2)
        r = self.step_rank
        basis, coef = out[:, :4 * r].view(-1, r, 2, 2), out[:, 4 * r:].view(-1, self.horizon, r)
        return eye + torch.einsum("bsk,bkij->bsij", coef, basis)

    def forward(self, x, t):
        return self._matrices(self._hidden(x, t))

    def velocity(self, x, t, v):
        h = self._hidden(x, t)
        out = torch.einsum("bsij,bsj->bsi", self._matrices(h), v)
        if self.cross_step_rank:
            uv = self.cross(h).view(-1, 2, self.horizon, 2, self.cross_step_rank)
            m = torch.einsum("bsar,bsa->br", uv[:, 1], v)
            out = out + torch.einsum("bsar,br->bsa", uv[:, 0], m)
        return out


def user_velocity(user, x, t, v):
    """A user module's correction of the velocity v: its own `velocity` (a UserModule, with any cross-step coupling),
    else its per-step matrices (a FixedAdapter or any (x, t) -> [B, H, 2, 2] callable)."""
    if hasattr(user, "velocity"):
        return user.velocity(x, t, v)
    return torch.einsum("bsij,bsj->bsi", user(x, t), v)


class TaskModule(nn.Module):
    """A(x_t, t, v, obstacles) v + b(...): one affine map of the policy's velocity per chunk step,
    in both directions of the flow, from the obstacles as seen from the cursor. The chunk's steps
    attend (masked cross-attention) to one token per obstacle (position relative to the cursor,
    radius, distance, in units of `scale`) and a learned null token, so a trial without obstacles
    has something to attend to. The head is zero-initialized: a fresh module is the identity."""

    def __init__(self, horizon=16, d=128, tdim=64, heads=4, layers=2, scale=0.2):
        super().__init__()
        self.config = dict(horizon=horizon, d=d, tdim=tdim, heads=heads, layers=layers, scale=scale)
        self.horizon, self.tdim, self.scale = horizon, tdim, scale
        self.token_in = mlp(4, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.query_in = nn.Linear(2 + 2 + tdim + 2, d)
        self.step_emb = nn.Parameter(0.02 * torch.randn(horizon, d))
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.norm_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.norm_k = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                                 for _ in range(layers)])
        self.out = nn.Linear(d, 6)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def tokens(self, obs, B):
        """The obstacle tokens [B, 1 + K, d] with the null token first, and their padding mask."""
        null = self.null.expand(B, 1, -1)
        obstacles = obs.get("obstacles")
        if obstacles is None or obstacles.shape[1] == 0:
            return null, None
        rel = (obstacles[..., :2] - obs["pos"][:, None]) / self.scale
        feats = torch.cat([rel, obstacles[..., 2:3] / self.scale, rel.norm(dim=-1, keepdim=True)], -1)
        tokens = torch.cat([null, self.token_in(feats)], 1)
        mask = obs.get("obstacle_mask")
        if mask is None:
            return tokens, None
        return tokens, torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device), ~mask], 1)

    def forward(self, obs, x, t, v):
        B, H = x.shape[:2]
        tokens, pad = self.tokens(obs, B)
        temb = time_embedding(t, self.tdim)[:, None].expand(-1, H, -1)
        pos = (2 * obs["pos"] - 1)[:, None].expand(-1, H, -1)
        q = self.query_in(torch.cat([x, v, temb, pos], -1)) + self.step_emb
        for attn, norm_q, norm_k, ff in zip(self.attn, self.norm_q, self.norm_k, self.ff):
            k = norm_k(tokens)
            q = q + attn(norm_q(q), k, k, key_padding_mask=pad, need_weights=False)[0]
            q = q + ff(q)
        out = self.out(q)
        return torch.eye(2, device=x.device) + out[..., :4].view(B, H, 2, 2), out[..., 4:]


class ReversalAdapter(nn.Module):
    """The reversal adapter: a user module (any (x, t) -> [B, H, 2, 2] callable: a UserModule or a
    FixedAdapter) and/or a task module, composed at test time. `adapt` applies the task module in
    both directions and the user module in the backward pass: forward task(v), backward
    user(task(v)). A checkpoint holds one module (`kind` user or task; the first adapters, saved
    without a kind, are user modules)."""

    def __init__(self, user=None, task=None):
        super().__init__()
        self.user, self.task = user, task

    @property
    def kind(self):
        return {(True, True): "composed", (True, False): "user", (False, True): "task", (False, False): "none"}[
            (self.user is not None, self.task is not None)]

    def adapt(self, obs, x, t, v, backward):
        if self.task is not None:
            A, b = self.task(obs, x, t, v)
            v = torch.einsum("bsij,bsj->bsi", A, v) + b
        if backward and self.user is not None:
            v = user_velocity(self.user, x, t, v)
        return v

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        kind = ckpt.get("kind", "user")
        module = (UserModule if kind == "user" else TaskModule)(**ckpt["config"])
        module.load_state_dict(ckpt["state_dict"])
        return cls(**{kind: module.to(device).eval()}).to(device).eval()


def as_adapter(adapter, device="cpu"):
    """None or a ReversalAdapter as is; a TaskModule in the task slot; a matrix F ([2, 2] or
    [B, 2, 2]), a FixedAdapter, a UserModule or any (x, t) callable in the user slot."""
    if adapter is None or isinstance(adapter, ReversalAdapter):
        return adapter
    if isinstance(adapter, TaskModule):
        return ReversalAdapter(task=adapter)
    if callable(adapter):
        return ReversalAdapter(user=adapter)
    return ReversalAdapter(user=FixedAdapter(adapter, device))


# ---------------------------------------------------------------- flows and steered chunks


def flow(policy, obs, x, t0, h, n, adapter=None):
    """n Euler steps of signed size h from flow time t0 through the policy's field, adapted by
    `adapter` (the task module in both directions, the user module when h < 0). Differentiable,
    for training; `reverse_flow` and `forward_flow` are its no-grad forms."""
    adapter = as_adapter(adapter, x.device)
    for i in range(n):
        t = torch.full((len(x),), t0 + i * h, device=x.device)
        v = policy(obs, x, t)
        if adapter is not None:
            v = adapter.adapt(obs, x, t, v, backward=h < 0)
        x = x + h * v
    return x


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


@torch.no_grad()
def flow_control_chunk(policy, obs, z, deflection, steer, tau, lam, steps, adapter=None, gain=1.0):
    """Flow control (Kao et al. 2026), normalized chunk from noise z: the command's chunk of deflections [B, H, 2],
    scaled by `gain` to the noise prior's units, is written into the first `tau` chunk steps of the steering rows at
    the start of each of the first `lam` (lambda) Euler steps; the rest of the chunk keeps its noise and the flow
    runs unmodified (the task module in its field, as in every forward pass). lambda 1: the initial condition only,
    Kao et al.'s method; lambda 0: nothing written, the policy alone; lambda > 1: the lab's variant."""
    adapter = as_adapter(adapter, z.device)
    written = gain * deflection[:, :tau]
    x, h = z.clone(), 1.0 / steps
    for i in range(steps):
        if i < lam:
            x[:, :tau] = torch.where(steer, written, x[:, :tau])
        t = torch.full((len(x),), i * h, device=x.device)
        v = policy(obs, x, t)
        if adapter is not None:
            v = adapter.adapt(obs, x, t, v, backward=False)
        x = x + h * v
    return x


@torch.no_grad()
def frs_chunk(policy, obs, z, ref, steer, n_reverse_steps, steps, adapter=None):
    """Normalized chunk: the reference reversed n of the N steps (landing at t = 1 - n/N),
    then run forward in N - n steps, so a partial reversal costs the same N evaluations
    as a plain chunk (as in LeRobot); a full reversal (n = N) runs forward in N steps.
    Rows that do not steer get the plain chunk from z."""
    n = steps if n_reverse_steps is None else n_reverse_steps
    if not 1 <= n <= steps:
        raise ValueError(f"n_reverse_steps must be in [1, {steps}], got {n}")
    latent = reverse_flow(policy, obs, ref.clone(), n, steps, adapter)
    steered = forward_flow(policy, obs, latent, 1.0 - n / steps, 1.0, steps if n == steps else steps - n, adapter)
    plain = forward_flow(policy, obs, z, 0.0, 1.0, steps, adapter) if not steer.all() else steered
    return torch.where(steer, steered, plain)


@torch.no_grad()
def steered_chunk(policy, obs, command, mode, tau=5, n_reverse_steps=None, steps=10, dt=0.02,
                  max_speed=MAX_SPEED, adapter=None, reference=None, ic_gain=1.0, lam=1):
    """One action chunk [B, H, 2] in screen units per step, for commands [B, 2].

    The reference chunk the shared modes steer toward is the command held
    constant over the chunk, or `reference` [B, H, 2] (a deflection per step,
    e.g. the chunk an expert_chunk agent means) when given: flow_control writes
    it, scaled by `ic_gain`, into x_t (see `flow_control_chunk`); frs converts
    it to action units. Rows whose command (or whole reference) is inside the
    deadband get the plain policy chunk, drawn from the same noise they would
    have had in mode "policy".
    """
    B, H = len(command), policy.horizon
    z = torch.randn(B, H, 2, device=command.device)
    steer = torch.as_tensor(active_rows(command, reference), device=command.device)[:, None, None]
    if mode == "policy" or not steer.any():
        return forward_flow(policy, obs, z, 0.0, 1.0, steps, adapter) * policy.action_scale
    deflection = command[:, None].expand(-1, H, -1) if reference is None else reference
    if mode == "flow_control":
        if not 0 <= tau <= H:
            raise ValueError(f"tau must be in [0, {H}] chunk steps for flow_control, got {tau}")
        if not 0 <= lam <= steps:
            raise ValueError(f"lambda must be in [0, {steps}] Euler steps for flow_control, got {lam}")
        x = flow_control_chunk(policy, obs, z, deflection, steer, tau, lam, steps, adapter, ic_gain)
    elif mode == "frs":
        x = frs_chunk(policy, obs, z, to_action(deflection, policy, dt, max_speed), steer, n_reverse_steps, steps, adapter)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    return x * policy.action_scale


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
    scale of the command in the noise prior, `lam` the Euler steps it is written at.
    """

    def __init__(self, policy, layouts, mode, tau=5, n_reverse_steps=None, steps=10, execute=8,
                 dt=0.02, max_speed=MAX_SPEED, speed_gain=1.0, adapter=None, device="cpu", gain_mode="step",
                 ic_gain=1.0, lam=1):
        assert mode in MODES, mode
        assert speed_gain > 0, speed_gain
        assert gain_mode in GAIN_MODES, gain_mode
        self.gain, self.gain_mode, self.adapter = speed_gain, gain_mode, as_adapter(adapter, device)
        self.ic_gain, self.lam = ic_gain, lam
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
        if z is None:
            z = torch.randn(len(self.layouts), self.policy.horizon, 2, device=obs["pos"].device)
        return forward_flow(self.policy, obs, z, 0.0, 1.0, self.steps, self.adapter)

    def act(self, command, reference=None):
        """Actions [B, 2] for this control step. `reference` [B, H, 2] (a deflection per step)
        replaces the constant-velocity reference at a replan, see `steered_chunk`."""
        if self.t >= self.cycle:  # a new chunk
            self.t = 0
            if self.mode != "teleop":
                ref = None if reference is None else _f32(reference, self.device)
                self.chunk = steered_chunk(self.policy, self.observation(), _f32(command, self.device), self.mode,
                                           self.tau, self.n_reverse_steps, self.steps, self.dt, self.max_speed,
                                           self.adapter, ref, ic_gain=self.ic_gain, lam=self.lam).cpu().numpy()
                self.chunks += 1
                if self.mode != "policy":
                    self.steered_chunks += active_rows(command, reference)
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
