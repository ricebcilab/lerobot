"""Simulated checks behind the steering design choices, saved for the report.

    python research/design_checks.py

Writes results/research/design_checks/checks.json with:

  euler_steps   network evaluations per trial and success vs Euler steps per
                chunk, with the goal given (flow) and hidden (flow_goaldrop)
  first_chunk   goal hidden: how 64 first chunks from the same start divide
                among the targets, vs Euler steps
  memory        goal hidden: what keeps the policy on its target (velocity
                history, open-loop chunk execution)
  latency       ms per steered chunk on GPU and CPU
  replanning    steering vs teleop with a noisy simulated user, replanning
                every 8, 4, 2 or 1 steps
  speed_gain    speed gain 1 vs 0.5, with the policy history unscaled or not
  corruption    the shipped clean and *_gain05y configs at input noise 0 and 0.3

Flow control here is the lab's variant these checks were recorded with: tau 16, lambda 5, the command in
action units (ic_gain = action_gain(policy)); see cursor_sa/steering.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
import time
from pathlib import Path

import numpy as np
import torch

from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.analysis import choice_stats, eligible, spread, trial_metrics
from cursor_sa.channel import InputChannel, adapter_spec, matrix_spec
from cursor_sa.env import CursorEnv, TaskConfig, sample_layouts
from cursor_sa.paths import RESULTS, policy_path
from cursor_sa.policy import FlowPolicy
from cursor_sa.records import logged
from cursor_sa.steering import SharedController, observation, steered_chunk, action_gain
from evaluate_policy_goal_given import rollout

HERE = Path(__file__).parent
TASK = TaskConfig()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(name, device=DEV):
    return FlowPolicy.load(policy_path(name), device)


def policy_rollout(policy, layouts, execute=8, steps=10, hide_goal=False, zero_hist=False, seed=0):
    """The policy alone, with the memory ablations: zero_hist hides its velocity history."""
    torch.manual_seed(seed)
    return rollout(policy, layouts, TASK, execute, steps, DEV, hide_goal, zero_hist)


def evaluations(env, path, layouts, execute, steps):
    """Median network evaluations until the cursor first enters the target it finally
    selects, and until the selection completes (after the 0.3 s hold)."""
    ok = env.selected >= 0
    tgt = layouts.targets[np.arange(len(layouts)), np.maximum(env.selected, 0)]
    entry = (np.linalg.norm(path - tgt[:, None], axis=-1) < TASK.target_radius).argmax(1)
    chunks = lambda s: np.ceil(s / execute)
    return dict(chunks_to_entry=float(np.median(chunks(entry[ok]))),
                evals_to_entry=float(np.median(chunks(entry[ok]) * steps)),
                chunks_to_select=float(np.median(chunks(env.steps[ok]))),
                evals_to_select=float(np.median(chunks(env.steps[ok]) * steps)))


def euler_steps():
    flow, regress, free = load("flow"), load("regress"), load("flow_goaldrop")
    lay = sample_layouts(np.random.default_rng(1234 + 4), 500, 4, TASK)
    given = {}
    for steps in (1, 2, 4, 10):
        env, path = policy_rollout(flow, lay, steps=steps)
        m = trial_metrics(env, path, lay, TASK)
        given[steps] = dict(success=float(env.success.mean()), time_s=float(np.median(m["time_s"][env.success])),
                            bow_sd=float(m["bow"][env.success].std()), **evaluations(env, path, lay, 8, steps))
    env, path = policy_rollout(regress, lay)
    m = trial_metrics(env, path, lay, TASK)
    given["regression"] = dict(success=float(env.success.mean()), time_s=float(np.median(m["time_s"][env.success])),
                               bow_sd=float(m["bow"][env.success].std()))
    hidden = {}
    for steps in (10, 4, 2, 1):
        hidden[steps] = goal_free_condition(free, steps=steps)
    return dict(given=given, hidden=hidden, setup="4 targets; goal given: 500 layouts; goal hidden: 200 layouts x 20 runs")


def goal_free_condition(policy, execute=8, steps=10, zero_hist=False):
    lay = sample_layouts(np.random.default_rng(4321 + 4), 200, 4, TASK)
    rep = lay[np.repeat(np.arange(200), 20)]
    env, path = policy_rollout(policy, rep, execute, steps, hide_goal=True, zero_hist=zero_hist)
    st = choice_stats(env.selected.reshape(200, 20), lay, TASK, np.random.default_rng(0))
    ok = env.selected >= 0
    chosen = rep[np.arange(len(rep))]
    chosen.goal = np.maximum(env.selected, 0)  # path efficiency toward the target it chose
    m = trial_metrics(env, path, chosen, TASK)
    out = dict(selected_any=st["selected_any"], spread=st["spread"], spread_ideal=st["spread_ideal"],
               time_s=float(np.median(env.steps[ok]) * TASK.dt) if ok.any() else None,
               path_efficiency=float(np.median(m["efficiency"][ok])) if ok.any() else None)
    if ok.any():
        out["evals_to_select"] = evaluations(env, path, rep, execute, steps)["evals_to_select"]
    return out


def memory():
    free = load("flow_goaldrop")
    return {name: goal_free_condition(free, **kw) for name, kw in [
        ("normal (history + 8-step execution)", {}),
        ("history zeroed at each replan", dict(zero_hist=True)),
        ("replan every step (history kept)", dict(execute=1)),
        ("execute all 16 steps", dict(execute=16)),
        ("replan every step and history zeroed", dict(execute=1, zero_hist=True))]}


def first_chunk():
    free = load("flow_goaldrop")
    L, K = 200, 64
    lay = sample_layouts(np.random.default_rng(4321 + 4), L, 4, TASK)
    rep = lay[np.repeat(np.arange(L), K)]
    obs = observation(rep, rep.start, np.zeros((len(rep), free.hist, 2)), DEV)
    to_target = rep.targets - rep.start[:, None]
    to_target /= np.linalg.norm(to_target, axis=-1, keepdims=True)
    elig = eligible(lay, TASK)
    out = {}
    for steps in (1, 2, 4, 10):
        torch.manual_seed(0)
        disp = free.sample(obs, steps=steps).sum(1).cpu().numpy()
        heading = disp / np.maximum(np.linalg.norm(disp, axis=-1, keepdims=True), 1e-9)
        cos = np.where(np.repeat(elig, K, 0), (to_target * heading[:, None]).sum(-1), -9)
        counts = (cos.argmax(1).reshape(L, K)[..., None] == np.arange(4)).sum(1)
        out[steps] = dict(mean_length=float(np.linalg.norm(disp, axis=-1).mean()), spread=spread(counts, elig))
    return out


def latency():
    out = {}
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        pol = load("flow_goaldrop", dev)
        lay = sample_layouts(np.random.default_rng(0), 1, 4, TASK)
        obs = SharedController(pol, lay, "policy", device=dev).observation()
        cmd = torch.tensor([[0.7, 0.2]], device=dev)
        for mode in ("flow_control", "frs"):
            for _ in range(3):
                steered_chunk(pol, obs, cmd, mode)
            if dev == "cuda":
                torch.cuda.synchronize()
            tic = time.perf_counter()
            for _ in range(20):
                steered_chunk(pol, obs, cmd, mode)
            if dev == "cuda":
                torch.cuda.synchronize()
            out[f"{dev} {mode}"] = 1000 * (time.perf_counter() - tic) / 20
    return out


def steer(policy, mode, noise, corruption=None, adapter=None, layout_seed=7, n=200, n_targets=4, cls=SharedController,
          channel_seed=1, **kw):
    """A simulated user steering through the input channel; returns success and timing.
    Its own loop rather than session.run_trials so `cls` can swap in a variant controller."""
    lay = sample_layouts(np.random.default_rng(layout_seed), n, n_targets, TASK)
    env = CursorEnv(lay, TASK)
    torch.manual_seed(0)
    user = ExpertAgent(kw.pop("lag", 0))
    user.reset(n, InputChannel(corruption, noise, np.random.default_rng(channel_seed)))
    ctl = cls(policy, lay, mode, adapter=adapter, device=DEV, **kw)
    peak = np.zeros(n)
    while not env.done.all():
        _, served, _ = user.step(env.pos, lay, ctl)
        prev = env.pos.copy()
        env.step(ctl.act(served))
        ctl.record(env.pos - prev)
        peak = np.maximum(peak, np.linalg.norm(env.pos - prev, axis=1) / TASK.dt)
    ok = env.success
    return dict(success=float(ok.mean()), selected_any=float((env.selected >= 0).mean()),
                wrong=float(((env.selected >= 0) & ~ok).mean()), timeout=float((env.selected < 0).mean()),
                time_s=float(np.median(env.steps[ok]) * TASK.dt) if ok.any() else None,
                peak_speed=float(np.median(peak)))


def replanning():
    free = load("flow_goaldrop")
    arms = [("policy", "policy", {}), ("teleop", "teleop", {})]
    arms += [(f"flow control, tau 5, execute {e}", "flow_control", dict(tau=16, lam=5, ic_gain=action_gain(free), execute=e)) for e in (8, 4, 2, 1)]
    arms += [(f"FRS, execute {e}", "frs", dict(execute=e)) for e in (8, 2)]
    return {name: {str(s): steer(free, mode, s, **kw) for s in (0.3, 0.6)} for name, mode, kw in arms}


class ActualHistory(SharedController):
    """Alternative: the policy sees the scaled (actual on-screen) displacement."""

    def record(self, displacement):
        self.pos = self.pos + displacement
        self.hist = np.concatenate([self.hist[:, 1:], displacement[:, None]], 1)


def speed_gain():
    free = load("flow_goaldrop")
    out = {}
    for mode in ("policy", "teleop", "flow_control", "frs"):
        for label, cls, gain in [("gain 1.0", SharedController, 1.0), ("gain 0.5, unscaled history", SharedController, 0.5),
                                 ("gain 0.5, actual history", ActualHistory, 0.5)]:
            if mode == "teleop" and cls is ActualHistory:
                continue
            out[f"{mode} | {label}"] = {str(s): steer(free, mode, s, cls=cls, tau=16, lam=5, ic_gain=action_gain(free), execute=2, speed_gain=gain)
                                        for s in (0.3, 0.6)}
    return out


def corruption():
    out = {}
    for arm in ("teleop", "flow_control", "frs", "teleop_gain05y", "flow_control_gain05y", "frs_gain05y",
                "mars_fixed_gain05y"):
        c = config.load_arm(arm + ".yaml")
        policy = load(c["model"])
        m = matrix_spec(c["corruption"])
        f = adapter_spec(c["reversal_adapter"], m) if c["mode"] == "frs" else None
        out[arm] = {str(s): steer(policy, c["mode"], s, corruption=m, adapter=f, layout_seed=11, n=300, channel_seed=2,
                                  n_targets=tuple(c["targets"]), tau=c["tau"], n_reverse_steps=c["n_reverse_steps"],
                                  steps=c["flow_steps"], execute=c["execute"], speed_gain=c["speed_gain"])
                    for s in (0.0, 0.3)}
    return out


def main():
    out = RESULTS / "research" / "design_checks"
    out.mkdir(parents=True, exist_ok=True)
    with logged(out / "log.txt"):
        results, tic = {}, time.time()
        for name, fn in [("latency", latency), ("first_chunk", first_chunk), ("euler_steps", euler_steps),
                         ("memory", memory), ("replanning", replanning), ("speed_gain", speed_gain),
                         ("corruption", corruption)]:
            results[name] = fn()
            print(f"{name} done ({time.time() - tic:.0f}s)", flush=True)
            with open(out / "checks.json", "w") as f:
                json.dump(results, f, indent=1)
        print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
