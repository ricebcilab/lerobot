"""The portal: run an experiment protocol for a participant, or play one arm freely.

    python run_experiment.py --protocol study.yaml --participant p1              # a person on the SpaceMouse
    python run_experiment.py --protocol study.yaml --participant p1 --resume     # continue an interrupted visit
    python run_experiment.py --protocol study.yaml --participant sim             # the expert agent, headless
    python run_experiment.py --protocol study.yaml --participant sim --watch     # the expert, rendered
    python run_experiment.py --arm study_mars.yaml                                  # free play, one arm

A participant is p<N> (a person) or sim (the expert agent). Both run the same
protocol through the same mechanism (cursor_sa/session.py) and write the same
records to results/<participant>/<protocol>_s<session>/. Free play (no
protocol) logs to results/<participant>/free/<timestamp>.jsonl, shows the
condition, and enables the setting keys:

    1-4   mode: policy, teleop, flow_control, frs        [ ]  depth (tau / n_reverse_steps)
    - =   speed gain                                     space  skip the trial        esc  quit

If a window cannot open (e.g. over SSH), set DISPLAY=:1 first.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cursor_sa import config
from cursor_sa.agents import make_agent
from cursor_sa.env import sample_layouts
from cursor_sa.paths import RESULTS
from cursor_sa.records import append_record, load_run, summarize, trial_record
from cursor_sa.session import Aborted, Run, arm_task, channel_for, load_checkpoints, run_trials

ARM_FLAGS = {  # free-play overrides of arm parameters
    "--model": ("model", str), "--flow-steps": ("flow_steps", int), "--execute": ("execute", int),
    "--mode": ("mode", str), "--tau": ("tau", int), "--lambda": ("lambda", int), "--ic-gain": ("ic_gain", float),
    "--n-reverse-steps": ("n_reverse_steps", yaml.safe_load),
    "--speed-gain": ("speed_gain", float), "--gain-mode": ("gain_mode", str), "--input-noise": ("input_noise", float),
    "--corruption": ("corruption", yaml.safe_load), "--reversal-adapter": ("reversal_adapter", yaml.safe_load),
    "--velocity-tau": ("velocity_tau", float), "--curl": ("curl", float), "--targets": ("targets", None),
    "--target-radius": ("target_radius", float), "--min-target-sep": ("min_target_sep", float),
    "--reach": ("reach", yaml.safe_load), "--dwell-s": ("dwell_s", float), "--time-limit-s": ("time_limit_s", float),
    "--obstacle": ("obstacle", yaml.safe_load),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", help="protocol config, e.g. study.yaml (bare names are looked up in configs/)")
    p.add_argument("--arm", help="free play with this arm config instead of a protocol")
    p.add_argument("--participant", help="p<N> for a person (default p0 in free play), sim for the expert agent")
    p.add_argument("--session", type=int, default=1, help="visit number; a new one gets new layouts")
    p.add_argument("--resume", action="store_true", help="continue an interrupted visit from its next trial")
    p.add_argument("--watch", action="store_true", help="render a sim run in real time")
    p.add_argument("--no-blind", action="store_true", help="show the condition to the participant")
    p.add_argument("--allow-keyboard", action="store_true", help="arrow keys may drive the cursor (study runs)")
    p.add_argument("--results", default=str(RESULTS), help="results folder")
    p.add_argument("--cpu", action="store_true", help="run the policy on the CPU")
    over = p.add_argument_group("free-play overrides of the arm", "each wins over the arm config")
    for flag, (name, typ) in ARM_FLAGS.items():
        if typ is None:
            over.add_argument(flag, dest=name, type=int, nargs=2, default=argparse.SUPPRESS, metavar=("MIN", "MAX"))
        else:
            over.add_argument(flag, dest=name, type=typ, default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if bool(args.protocol) == bool(args.arm):
        p.error("give exactly one of --protocol or --arm")
    if args.protocol and not args.participant:
        p.error("--participant is required with a protocol (p<N> or sim)")
    args.participant = args.participant or "p0"
    if args.participant != "sim" and not (args.participant.startswith("p") and args.participant[1:].isdigit()):
        p.error("--participant must be p<N> (a person) or sim (the expert agent)")
    args.overrides = {name: getattr(args, name) for _, (name, _) in ARM_FLAGS.items() if hasattr(args, name)}
    return args


def device_for(args):
    return "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"


def make_renderer(**kw):
    from cursor_sa.renderer import Renderer  # pygame is only needed with a window
    return Renderer(**kw)


# ---------------------------------------------------------------- a protocol run


def run_protocol(args):
    protocol = config.load_protocol(args.protocol)
    sim = args.participant == "sim"
    agent = make_agent("expert", **protocol["expert"]) if sim else make_agent("human", allow_keyboard=args.allow_keyboard)
    run = Run(protocol, args.participant, args.session, agent, device_for(args), args.results, args.resume)
    run.start()
    print(f"{args.participant}, {run.run_name}: blocks {run.order}, {len(run.layouts)} trials + "
          f"{len(run.practice)} practice each, layout seed {run.seed} -> {run.dir}", flush=True)
    renderer = None
    if not sim or args.watch:
        renderer = make_renderer(title=f"{protocol['name']}: {args.participant}",
                                 blind=protocol["display"]["blind"] and not args.no_blind,
                                 ready_hold_s=protocol["trial"]["ready_hold_s"],
                                 feedback_s=protocol["trial"]["feedback_s"], auto_continue_s=1.0 if sim else None)
    tic = time.time()
    try:
        played = run.run(renderer)
        print(f"{played} trials in {time.time() - tic:.0f}s")
    except Aborted:
        print("aborted by the participant; re-run with --resume to continue")
    finally:
        if renderer is not None:
            renderer.close()
        if hasattr(agent, "close"):
            agent.close()
    print_summary(run.dir)


def print_summary(run_dir):
    manifest, records = load_run(run_dir)
    trials = [r for r in records if not r["practice"] and r["outcome"] != "aborted"]
    print(f"{run_dir}: {len(trials)} trials" + (" (complete)" if manifest.get("finished") else ""))
    for arm in manifest["block_order"]:
        rows = [r for r in trials if r["arm"] == arm]
        if rows:
            s = summarize(rows)
            print(f"  {arm:16s} {s['hits']:3d}/{s['n']:<3d} hits ({100 * s['success']:5.1f}%)  "
                  f"median {s['time_s'] if s['time_s'] is not None else float('nan'):.2f} s")


# ---------------------------------------------------------------- free play


def setting_text(arm):
    depth = (f"tau {arm['tau']}/{config.CHUNK_STEPS}, gain {arm['ic_gain']:g}, lambda {arm['lambda']}" if arm["mode"] == "flow_control" else
             f"n_reverse_steps {arm['n_reverse_steps'] or arm['flow_steps']}/{arm['flow_steps']}" if arm["mode"] == "frs"
             else "")
    names = arm["learned_adapter"]
    parts = ([] if arm["mode"] != "frs" else ["learned user adapter"] if isinstance(names, str) else
             (["learned user adapter"] if isinstance(names, dict) and names["user"] else
              ["fixed user adapter"] if arm["adapter_matrix"] is not None else []))
    if isinstance(names, dict) and names["task"]:
        parts.append("task adapter")
    adapter = "".join(f"   {p}" for p in parts)
    return f"{arm['method']}   {depth}{adapter}   speed gain {arm['speed_gain']:.1f}" + (" by time" if arm["gain_mode"] == "time" else "")


def apply_request(arm, what, value):
    """A free-play key: change the arm in place and re-derive it."""
    if what == "mode":
        arm["mode"] = value
    elif what == "depth":
        if arm["mode"] == "flow_control":
            arm["tau"] = int(np.clip(arm["tau"] + value, 0, config.CHUNK_STEPS))
        elif arm["mode"] == "frs":
            n = int(np.clip((arm["n_reverse_steps"] or arm["flow_steps"]) + value, 1, arm["flow_steps"]))
            arm["n_reverse_steps"] = None if n == arm["flow_steps"] else n
    elif what == "speed_gain":
        arm["speed_gain"] = round(float(np.clip(arm["speed_gain"] + value, 0.1, 2.0)), 1)
    config.derive(arm)


def free_play(args):
    from cursor_sa.renderer import Skipped
    arm = config.load_arm(args.arm, args.overrides)
    sim = args.participant == "sim"
    agent = make_agent("expert") if sim else make_agent("human", allow_keyboard=True)
    device = device_for(args)
    policies, adapters = load_checkpoints({arm["name"]: arm}, device)
    policy = policies[arm["model"]]
    renderer = make_renderer(title=f"free play: {arm['name']}", blind=False, free_play=True,
                             auto_continue_s=1.0 if sim else None)
    log_dir = Path(args.results) / args.participant / "free"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    rng = np.random.default_rng()
    channel = channel_for(arm, np.random.default_rng())
    tally, n = {}, 0
    try:
        while True:
            layouts = sample_layouts(rng, 1, tuple(arm["targets"]), arm_task(arm))
            hits, played = tally.get(arm["method"], (0, 0))
            hud = dict(title=setting_text(arm), line4=f"{arm['method']}: {hits}/{played} hits    trials {n}",
                       line3=f"corruption {arm['corruption'] or 'off'}    input noise {arm['input_noise']}    "
                             f"{'curl %g/s   ' % arm['curl'] if arm['curl'] else ''}"
                             f"{'obstacle %s   ' % arm['obstacle']['kind'] if arm['obstacle'] else ''}config {arm['name']}",
                       line2=("expert agent" if sim else f"SpaceMouse: {agent.status}   (arrow keys work too)"))
            renderer.status_line = "1-4 mode   [ ] depth   - = speed gain   space skip   esc quit"
            try:
                rec = run_trials(arm, layouts, agent, channel, policy, adapters, device, renderer, hud)[0]
            except Skipped:
                continue
            n += 1
            tally[arm["method"]] = (hits + (rec["outcome"] == "hit"), played + 1)
            append_record(log, trial_record(rec, arm, args.participant, "free", None, 0, arm["name"], n, False, None,
                                            agent.name))
            while renderer.requests:
                what, value = renderer.requests.pop(0)
                if what != "skip":
                    apply_request(arm, what, value)
                    if any(n not in adapters for n in config.adapter_names(arm)):
                        adapters.update(load_checkpoints({arm["name"]: arm}, device)[1])
            channel = channel_for(arm, channel.rng)
    except Aborted:
        pass
    finally:
        renderer.close()
        if hasattr(agent, "close"):
            agent.close()
    print(f"{n} trials logged to {log}" if n else "no finished trials")
    for method, (hits, played) in tally.items():
        print(f"  {method}: {hits}/{played} hits")


def main(argv=None):
    args = parse_args(argv)
    run_protocol(args) if args.protocol else free_play(args)


if __name__ == "__main__":
    main()
