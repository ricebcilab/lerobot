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
in closed loop than one trained from scratch under the study's controller: README, the grid's checks).
--demo_seconds trains on as many demonstrations as fit in that much demonstrating, drawn as the adapter
trainer draws them (train_adapter.budget_demos), so a task module given the same budget and seed sees the same
ones. --eval_steps picks how long the policy trains: at those steps and the last it scores the EMA weights alone
with the goal given on layouts of its task (`goal_given_scorer`, apart from the sweeps' trials) and keeps the best.

Saves the EMA weights (with --eval_steps, the best scored) to checkpoints/policies/<name>/policy.pt and the loss
curve (and the scores) to checkpoints/policies/<name>/train_log.json. convert_checkpoints.py turns policy.pt into
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

from cursor_sa import config
from cursor_sa.agents import make_agent
from cursor_sa.demos import DemoSampler, make_demos
from cursor_sa.env import obstacle_spec, sample_layouts
from cursor_sa.paths import policy_dir, policy_path
from cursor_sa.policy import FlowPolicy, load_checkpoint
from cursor_sa.session import arm_task, readout_for, run_trials
from train.train_adapter import budget_demos, layouts_for


def goal_given_scorer(args, device):
    """The share of --eval_trials layouts of the task that a policy alone hits with the goal given (the expert_chunk
    agent with the readout off: grid_task_check's oracle) under --eval_arm's controller. The layouts and the trials'
    noise come from --eval_seed, and scoring leaves the training's random state as it was."""
    arm = config.load_arm(args.eval_arm, {"mode": "teleop", "obstacle": args.obstacle})
    layouts = sample_layouts(np.random.default_rng(args.eval_seed), args.eval_trials, tuple(arm["targets"]), arm_task(arm))

    def score(policy):
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.manual_seed(args.eval_seed + 1)
            readout = readout_for(arm, np.random.default_rng(args.eval_seed + 1), off=True)
            records = run_trials(arm, layouts, make_agent("expert_chunk"), readout, policy.eval(), {}, device)
        policy.train()
        return float(np.mean([r["outcome"] == "hit" for r in records]))

    return score


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="flow", choices=["flow", "regress"])
    p.add_argument("--name", help="output folder name (default: the mode)")
    p.add_argument("--n_demos", type=int, default=20000)
    p.add_argument("--demo_seconds", type=float, help="instead of --n_demos, as many demonstrations as fit in this much "
                   "demonstrating (train_adapter.budget_demos)")
    p.add_argument("--eval_steps", type=int, nargs="+", default=[], help="score the EMA weights at these steps and the "
                   "last, alone with the goal given (goal_given_scorer), and save the best (on a tie, the later)")
    p.add_argument("--eval_trials", type=int, default=200, help="with --eval_steps: the layouts scored")
    p.add_argument("--eval_seed", type=int, default=900, help="with --eval_steps: their seed, apart from the sweeps' "
                   "trials (800 and 810)")
    p.add_argument("--eval_arm", default="condition_grid.yaml", help="with --eval_steps: the controller and task geometry "
                   "they run under")
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
        if split == "train" and args.demo_seconds is not None:  # a time budget; n_demos becomes the count it held
            layouts, task, *demos = budget_demos(args, args.obstacle, rng, args.demo_seconds, args.p_free)
            args.n_demos = len(layouts)
            data[split] = (layouts, *demos)
        else:
            layouts, task = layouts_for(args, args.obstacle, rng, n, args.p_free)
            data[split] = (layouts, *make_demos(rng, layouts, task, allow_failed=args.obstacle is not None))
    actions = data["train"][1]
    moving = actions[np.linalg.norm(actions, axis=-1) > 0]
    action_scale = float(np.sqrt((moving ** 2).mean()))
    samplers = {k: DemoSampler(*v, args.horizon, args.hist, device) for k, v in data.items()}
    print(f"{args.n_demos} demos ({data['train'][2].sum()} samples, {data['train'][2].sum() * task.dt:.0f} s of "
          f"demonstrating) in {time.time() - tic:.0f}s; action scale {action_scale:.4f}", flush=True)

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

    scored = set(args.eval_steps) | ({args.steps} if args.eval_steps else set())
    score = goal_given_scorer(args, device) if scored else None
    scores, best = [], None  # best: (success, step, the EMA weights then)
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
        if step in scored:
            success = score(ema)
            scores.append(dict(step=step, success=success))
            if best is None or success >= best[0]:
                best = (success, step, copy.deepcopy(ema.state_dict()))
            print(f"step {step:6d}  alone with the goal given {100 * success:.1f}%  {time.time() - tic:.0f}s", flush=True)

        if step % args.log_every == 0:
            with torch.no_grad():
                val = np.mean([ema.loss(*samplers["val"].sample(args.batch, args.goal_dropout)).item()
                               for _ in range(10)])
            log.append(dict(step=step, train=running / args.log_every, val=float(val)))
            print(f"step {step:6d}  train {running / args.log_every:.4f}  val(ema) {val:.4f}  "
                  f"{time.time() - tic:.0f}s", flush=True)
            running = 0.0

    weights = ema.state_dict() if best is None else best[2]
    torch.save(dict(config=model.config, state_dict=weights, args=vars(args)), out / "policy.pt")
    scored_log = {} if best is None else dict(eval=scores, best_step=best[1])
    with open(out / "train_log.json", "w") as f:
        json.dump(dict(args=vars(args), action_scale=action_scale, log=log, **scored_log), f, indent=1)
    print(f"saved {out / 'policy.pt'}" + ("" if best is None else f" (step {best[1]}: {100 * best[0]:.1f}% alone, goal given)"))


if __name__ == "__main__":
    main()
