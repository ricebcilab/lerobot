"""Train the cursor flow policy, or the regression baseline, on synthetic demos.

    python train/train_policy.py --mode flow
    python train/train_policy.py --mode regress
    python train/train_policy.py --mode flow --name flow_goaldrop --goal_dropout 0.25   # the study's policy
    python train/train_policy.py --name flow_goaldrop_disc_large --goal_dropout 0.25 --targets 4 4 --reach 0.35 0.5 \
        --obstacle '{kind: disc, radius: 0.08, along: 0.5, offset: [0.012, 0.024]}'

--obstacle trains on one obstacle task, a share --p_free of the layouts without the obstacle, as that
task's module is trained (train_adapter.py, same geometry and demonstrations), and gives the policy
obstacle tokens: the grid's retrained policy for that task (`retrained_mars_user`), against the frozen
policy steered onto it by the task module. --init fine-tunes a trained policy instead of starting from
scratch: its architecture, weights and action scale, with an obstacle head new to it (the obstacle
tokens' input layer and type embedding) starting fresh (the study's policy fine-tuned this way did worse
in closed loop than one trained from scratch: README, the grid's checks).

Saves the EMA weights to checkpoints/policies/<name>/policy.pt and the loss curve
to checkpoints/policies/<name>/train_log.json. convert_checkpoints.py turns policy.pt into
the lerobot policy directory (config.json + model.safetensors); FlowPolicy.load reads both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cursor_sa.demos import DemoSampler, make_demos
from cursor_sa.env import obstacle_spec
from cursor_sa.paths import policy_dir, policy_path
from cursor_sa.policy import FlowPolicy, load_checkpoint
from train.train_adapter import layouts_for


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="flow", choices=["flow", "regress"])
    p.add_argument("--name", help="output folder name (default: the mode)")
    p.add_argument("--n_demos", type=int, default=20000)
    p.add_argument("--targets", type=int, nargs=2, default=[1, 8], help="min and max targets per trial")
    p.add_argument("--reach", type=yaml.safe_load, nargs=2, default=[0.2, None], help="min and max reach (null: none)")
    p.add_argument("--obstacle", type=yaml.safe_load, help="an obstacle task to train on; gives the policy obstacle tokens")
    p.add_argument("--p_free", type=float, default=0.25, help="with --obstacle: share of layouts without it (as a task module)")
    p.add_argument("--init", help="fine-tune this policy (under checkpoints/policies/): its architecture, weights and "
                   "action scale; an obstacle head new to it starts fresh")
    p.add_argument("--val_demos", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--hist", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--goal_dropout", type=float, default=0.0,
                   help="fraction of samples trained with no goal flag (goal-free policy)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.obstacle = obstacle_spec(args.obstacle, "--obstacle")
    base = load_checkpoint(policy_path(args.init)) if args.init else None
    if base is not None:  # the base policy's architecture
        for key in ("horizon", "hist", "d_model", "layers", "heads", "mode"):
            setattr(args, key, base["config"][key])

    out = policy_dir(args.name or args.mode)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tic = time.time()
    data = {}
    for split, n in [("train", args.n_demos), ("val", args.val_demos)]:
        layouts, task = layouts_for(args, args.obstacle, rng, n, args.p_free)
        data[split] = (layouts, *make_demos(rng, layouts, task, allow_failed=args.obstacle is not None))
    actions = data["train"][1]
    moving = actions[np.linalg.norm(actions, axis=-1) > 0]
    action_scale = float(np.sqrt((moving ** 2).mean()))
    samplers = {k: DemoSampler(*v, args.horizon, args.hist, device) for k, v in data.items()}
    print(f"{args.n_demos} demos ({data['train'][2].sum()} samples) in {time.time() - tic:.0f}s; "
          f"action scale {action_scale:.4f}", flush=True)

    if base is None:
        model = FlowPolicy(horizon=args.horizon, hist=args.hist, d_model=args.d_model, layers=args.layers,
                           heads=args.heads, mode=args.mode, action_scale=action_scale,
                           obstacle_tokens=args.obstacle is not None).to(device)
    else:  # its weights and action scale (the units its users' modules work in); a new obstacle head starts fresh
        model = FlowPolicy(**{**base["config"], "obstacle_tokens": args.obstacle is not None}).to(device)
        missing, unexpected = model.load_state_dict(base["state_dict"], strict=False)
        assert not unexpected and all(k.startswith("obstacle_") for k in missing), (missing, unexpected)
        action_scale = float(model.action_scale)
        print(f"fine-tuning {args.init} (action scale {action_scale:.4f}); new: {sorted(missing)}", flush=True)
    ema = copy.deepcopy(model).requires_grad_(False)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * s / args.steps)))

    log, running, tic = [], 0.0, time.time()
    for step in range(1, args.steps + 1):
        loss = model.loss(*samplers["train"].sample(args.batch, args.goal_dropout))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
        running += loss.item()

        if step % args.log_every == 0:
            with torch.no_grad():
                val = np.mean([ema.loss(*samplers["val"].sample(args.batch, args.goal_dropout)).item()
                               for _ in range(10)])
            log.append(dict(step=step, train=running / args.log_every, val=float(val)))
            print(f"step {step:6d}  train {running / args.log_every:.4f}  val(ema) {val:.4f}  "
                  f"{time.time() - tic:.0f}s", flush=True)
            running = 0.0

    torch.save(dict(config=model.config, state_dict=ema.state_dict(), args=vars(args)), out / "policy.pt")
    with open(out / "train_log.json", "w") as f:
        json.dump(dict(args=vars(args), action_scale=action_scale, log=log), f, indent=1)
    print(f"saved {out / 'policy.pt'}")


if __name__ == "__main__":
    main()
