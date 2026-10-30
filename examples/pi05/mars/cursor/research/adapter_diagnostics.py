"""Diagnostics of the learned reversal adapter: why it transfers intent under one
corruption and not another.

    python research/adapter_diagnostics.py

For each corruption, at the start of a reach (where the target is chosen), the
goal-given policy's chunk is corrupted and reversed by every adapter, and the
share of regenerated chunks heading to the intended target is counted. Also
fits the objective's best *constant* 2x2 adapter (it shrinks), and tries a
norm-preserving variant of the learned H. Writes
results/research/adapter_diagnostics/diagnostics.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cursor_sa import config  # noqa: E402
from cursor_sa.channel import matrix_spec  # noqa: E402
from cursor_sa.demos import DemoSampler, make_demos  # noqa: E402
from cursor_sa.env import TaskConfig, sample_layouts  # noqa: E402
from cursor_sa.paths import RESULTS, adapter_path, policy_path  # noqa: E402
from cursor_sa.policy import FlowPolicy  # noqa: E402
from cursor_sa.records import logged  # noqa: E402
from cursor_sa.steering import ReversalAdapter, bci_chunk, hide_goal, observation, reverse_flow, user_velocity  # noqa: E402

OUT = RESULTS / "research" / "adapter_diagnostics"


def heads_to(chunk, layouts):
    """Index of the target each chunk heads for: the best match between the chunk's total
    displacement and the direction from the cursor start to each target."""
    d = chunk.sum(1).cpu().numpy()
    d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
    to = layouts.targets - layouts.start[:, None]
    to /= np.linalg.norm(to, axis=-1, keepdims=True) + 1e-9
    return np.where(layouts.mask, (to * d[:, None]).sum(-1), -9).argmax(1)


@torch.no_grad()
def reverse_norm(policy, obs, x, adapter, steps=10):
    """Learned H, with each step's adapted velocity rescaled to the policy velocity's norm."""
    h = 1.0 / steps
    for i in range(steps):
        t = torch.full((len(x),), 1.0 - i * h, device=x.device)
        v = policy(obs, x, t)
        a = user_velocity(adapter, x, t, v)
        x = x - h * a * (v.norm(dim=-1, keepdim=True) / a.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    return x


def constant_adapter(policy, corruption, device, steps=3000, sigma_max=1.0, seed=0):
    """The adapter's training objective (goal-hidden pairs) with H restricted to one constant 2x2 matrix."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    layouts = sample_layouts(rng, 5000, (1, 8), TaskConfig())
    sampler = DemoSampler(layouts, *make_demos(rng, layouts, TaskConfig()), policy.horizon, policy.hist, device)
    F = nn.Parameter(torch.eye(2, device=device))
    opt = torch.optim.Adam([F], lr=1e-2)
    for _ in range(steps):
        free = hide_goal(sampler.sample(512)[0])
        noise = torch.randn(512, policy.horizon, 2, device=device)
        bci = bci_chunk(policy.integrate(free, noise), corruption, sigma_max * torch.rand(512, device=device), policy)
        t = torch.rand(512, device=device)
        x = (1 - t[:, None, None]) * noise + t[:, None, None] * bci
        with torch.no_grad():
            v = policy(free, x, t)
        loss = ((v @ F.T - (bci - noise)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return F.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corruptions", type=yaml.safe_load,
                   default={"rot90": {"rotation_deg": 90}, "gain025y": {"scale": [1.0, 0.25]}},
                   help="label -> matrix spec; adapters are read from checkpoints/adapters/<label>_<goal|free>/")
    p.add_argument("--layouts", type=int, default=2000)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with logged(OUT / "log.txt"):
        device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
        base = config.load_arm("base.yaml")
        policy = FlowPolicy.load(policy_path(base["model"]), device)
        layouts = sample_layouts(np.random.default_rng(7), args.layouts, (2, 8), TaskConfig())
        cond = observation(layouts, layouts.start, np.zeros((len(layouts), policy.hist, 2)), device, goal_given=True)
        free = hide_goal(cond)
        out = {"reach_start": {}, "constant_adapter": {}}
        for label, spec in args.corruptions.items():
            m = torch.as_tensor(matrix_spec(spec), dtype=torch.float32, device=device)
            torch.manual_seed(0)
            intended = policy.integrate(cond, torch.randn(len(layouts), policy.horizon, 2, device=device))
            want = heads_to(intended, layouts)
            f_const = constant_adapter(policy, m, device)
            out["constant_adapter"][label] = dict(M=m.cpu().numpy().tolist(), F=f_const.cpu().numpy().tolist())
            learned = {d: ReversalAdapter.load(adapter_path(f"{label}_{d}"), device).user for d in ("goal", "free")}
            for sigma in (0.0, 1.0):
                bci = bci_chunk(intended, m, torch.full((len(layouts),), sigma, device=device), policy)
                chunks = {"policy alone": policy.integrate(free, torch.randn_like(bci))}
                for name, a in [("FRS", None), ("MARS, F = M", m), ("constant F from the objective", f_const),
                                ("learned H, goal pairs", learned["goal"]), ("learned H, goal-free pairs", learned["free"])]:
                    chunks[name] = policy.integrate(free, reverse_flow(policy, free, bci, 10, 10, a))
                chunks["learned H (goal-free pairs), norm-preserving"] = policy.integrate(
                    free, reverse_norm(policy, free, bci, learned["free"]))
                key = f"{label} | noise {sigma}"
                out["reach_start"][key] = {k: float((heads_to(c, layouts) == want).mean()) for k, c in chunks.items()}
                print(f"{label} noise {sigma}: " + "  ".join(f"{k} {100 * v:.0f}%" for k, v in out["reach_start"][key].items()),
                      flush=True)
        with open(OUT / "diagnostics.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {OUT / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
