"""The portal: run an experiment protocol for a participant.

    python run_experiment.py --protocol study_short --user U1 --participant p1            # a person on the SpaceMouse
    python run_experiment.py --protocol study_short --user U1 --participant p1 --resume   # continue an interrupted visit
    python run_experiment.py --protocol study_short --user U3 --participant p2            # another participant, another user
    python run_experiment.py --protocol study_short --user U1 --participant sim           # the expert agent, headless
    python run_experiment.py --protocol study_short --user U1 --participant sim --watch   # the expert, rendered
    python run_experiment.py --protocol study_long --user U1 --participant p1 --prepare   # a visit's scene tokens, ahead
    python run_experiment.py --protocol study_long --user U3 --participant p1             # the long study with another user
    python run_experiment.py --protocol study_short --user U1 --participant p1 --calibrate  # MARS on the person's own module

A participant is p<N> (a person) or sim (the expert agent). Both run the same
protocol through the same mechanism (cursor_sa/session.py) and write the same
records. A study has one protocol file per user (study_short_<user>.yaml,
study_long_<user>.yaml); --user picks it (study_short and U3 play
study_short_U3.yaml, which --protocol can also name whole). A protocol gives its
cells, one user x the tasks played, as a sweep of its own (run_sweep.py tunes
it): one task section per task, each a run of its own in
results/<study>/<participant>/<protocol>_s<session>_<task>/ (study_short or
study_long), with the arms for the user's cell at their tuned settings.

--calibrate trains the user module in the session, on the participant's own
commands: before the first block the cursor plays demonstration reaches by
itself while the person pushes as if steering it (protocol.calibration: 50
straight reaches after 5 practice ones, at half speed with the path drawn ahead,
about 2.5 minutes), a user module is trained on what the readout served (the
shared modules' recipe, 500 steps), and it takes
the shared module's place in MARS for the visit. The reaches are a run of their
own (<protocol>_s<session>_calibration/), the module is
checkpoints/adapters/participants/<study>/<participant>/<protocol>_s<session>/.

If a window cannot open (e.g. over SSH), set DISPLAY=:1 first.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cursor_sa import config
from cursor_sa.agents import make_agent
from cursor_sa.paths import CHECKPOINTS, RESULTS
from cursor_sa.records import load_run, now_iso, summarize
from cursor_sa.session import Aborted, Calibration, Visit, arm_task, load_checkpoints
from lerobot.policies.steering import AdapterModules


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", required=True, help="the study, with --user (study_short, study_long), or a protocol config "
                   "named whole, e.g. study_short_U1.yaml (bare names are looked up in configs/)")
    p.add_argument("--user", help="the user profile played, e.g. U3: the study's protocol file for that user")
    p.add_argument("--participant", required=True, help="p<N> for a person, sim for the expert agent")
    p.add_argument("--session", type=int, default=1, help="visit number; a new one gets new layouts")
    p.add_argument("--resume", action="store_true", help="continue an interrupted visit from its next trial")
    p.add_argument("--calibrate", action="store_true", help="train the user module in the session: calibration reaches "
                   "before the first block, then MARS plays the participant's own module in place of the shared one")
    p.add_argument("--watch", action="store_true", help="render a sim run in real time")
    p.add_argument("--no-blind", action="store_true", help="show the condition to the participant")
    p.add_argument("--allow-keyboard", action="store_true", help="arrow keys may drive the cursor (study runs)")
    p.add_argument("--prepare", action="store_true", help="make and cache the visit's scene tokens (the task modules that "
                   "read a picture of the scene), print its order, write nothing under results and exit")
    p.add_argument("--scene-cache", help="where a visit's scene tokens are kept (default checkpoints/scene/<protocol>)")
    p.add_argument("--results", default=str(RESULTS), help="results folder; a run goes to <results>/<study>/<participant>/")
    p.add_argument("--cpu", action="store_true", help="run the policy on the CPU")
    args = p.parse_args(argv)
    if args.participant != "sim" and not (args.participant.startswith("p") and args.participant[1:].isdigit()):
        p.error("--participant must be p<N> (a person) or sim (the expert agent)")
    try:
        args.protocol = protocol_file(args.protocol, args.user)
    except (FileNotFoundError, ValueError) as e:
        p.error(str(e))
    return args


def protocol_file(protocol, user=None):
    """The protocol config a visit plays: `protocol` as named, or with `user` the study's file for that user
    (study_short and U3 -> study_short_U3.yaml). A study named without its user, or with one it has no file for,
    is an error that lists the users it has."""
    study = protocol.removesuffix(".yaml")
    name = f"{study}_{user}.yaml" if user else f"{study}.yaml"
    try:
        config.resolve(name)
    except FileNotFoundError:
        prefix = f"{Path(study).name}_"
        users = sorted(p.stem.removeprefix(prefix) for p in config.CONFIG_DIR.glob(f"{prefix}*.yaml"))
        if not users:
            raise
        raise ValueError(f"{study} has a protocol per user: pass --user {', '.join(users[:-1])} or {users[-1]}") from None
    return name


def device_for(args):
    return "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"


def make_renderer(**kw):
    from cursor_sa.renderer import Renderer  # pygame is only needed with a window
    return Renderer(**kw)


# ---------------------------------------------------------------- a protocol run


def protocol_sections(protocol):
    """A protocol's task sections, {task: {method: arm}} in the order of its tasks, and the user played. The protocol's
    cells are a sweep of their own (`run_sweep.protocol_sweep`: one user x the tasks played, at one noise level), and
    every arm is the one that sweep builds for the user's cell on that task (`run_sweep.method_arm`: the user's
    corruption and module, the task's obstacle, time limit, policy and task module), at the cell's tuned setting from
    `cells.settings` ('<cell>|<noise>' -> method -> setting, run_sweep.py's settings.yaml for the protocol) and at the
    sweep's noise, which a sweep sets per trial and an arm of its own does not carry."""
    from run_sweep import cell_key, is_tuned, method_arm, protocol_sweep, with_setting  # run_sweep imports the session
    sweep = protocol_sweep(protocol)
    grid = sweep["grid"]
    if grid["family"] != "grid" or len(grid["noises"]) != 1:
        raise ValueError(f"protocol {protocol['name']}: its cells must be a users x tasks grid at one noise level")
    if len(grid["users"]) != 1:
        raise ValueError(f"protocol {protocol['name']}: a protocol plays one user, got {grid['users']} (one protocol "
                         f"per user: study_long_<user>.yaml)")
    user, noise, settings = grid["users"][0], grid["noises"][0], protocol["cells"]["settings"]
    sections = {}
    for task in grid["tasks"]:
        level = f"{user}x{task}"
        tuned, arms = settings.get(cell_key(level, noise)) or {}, {}
        for method in sweep["methods"]:
            arm = method_arm(sweep, method, level)
            if is_tuned(arm):
                if tuned.get(method) is None:
                    raise ValueError(f"protocol {protocol['name']}: cells.settings has no tuned setting for "
                                     f"{cell_key(level, noise)}, {method}")
                arm = with_setting(arm, tuned[method])
            arms[method] = {**arm, "input_noise": noise}
        sections[task] = arms
    return sections, user


def attach_scenes(runs, cache_dir, encoder_factory=None):
    """Give every run's layouts (its trials and its practice trials) the scene arrays its task modules read
    (`run_sweep.scene_sources`: a picture of the scene or its vision tokens), from the cache or made and cached
    (`run_sweep.attach_scene`, keyed by participant, visit and task). The runs' checkpoints must be loaded;
    `encoder_factory()` makes the pi0.5 scene encoder, called only for vision tokens the cache lacks."""
    from run_sweep import attach_scene, scene_sources
    for run in runs:
        sources = scene_sources(run.adapters)
        task = run.section["task"] if run.section else "run"
        for layouts, split in ((run.layouts, "trials"), (run.practice, "practice")):
            attach_scene(layouts, sources, cache_dir, f"{run.participant}_s{run.session}_{task}_{split}", encoder_factory)


def scene_encoders(device):
    """A factory of the one pi0.5 scene encoder (made at its first call) and the list it lands in, to close."""
    made = []

    def factory():
        if not made:
            from lerobot.policies.steering import SceneEncoder
            from run_sweep import SCENE_VLM_PATH
            made.append(SceneEncoder(SCENE_VLM_PATH, device=device))
        return made[0]

    return factory, made


def train_user_module(calibration, device, renderer=None):
    """The participant's own user module from their calibration (`session.Calibration`, every reach played): the
    shared modules' recipe on what the readout served during the reaches (`train_adapter.train_calibrated_user`, the
    protocol's `calibration.steps`), saved under the calibration's module name with the calibration it came from. A
    window shows a waiting screen and keeps answering its events meanwhile."""
    from train.train_adapter import train_calibrated_user  # the trainer imports the session

    plan, arm = calibration.protocol["calibration"], calibration.arms["calibration"]
    layouts, actions, lengths, served = calibration.data()
    if renderer is not None:
        renderer.wait_screen("One moment", ["The next part is being prepared."])
    tic = time.time()
    policies = calibration.policies or load_checkpoints(calibration.arms, device)[0]  # a resumed, played run loads none
    user, log = train_calibrated_user(policies[arm["model"]].model, layouts, actions, lengths, served, device,
                                      steps=plan["steps"], tick=None if renderer is None else renderer.pump)
    meta = dict(participant=calibration.participant, session=calibration.session, protocol=calibration.protocol["name"],
                run=calibration.run_name, user=calibration.labels.get("user"), reaches=len(layouts), steps=plan["steps"],
                slow=calibration.slow, seconds=round(float(lengths.sum()) * calibration.slow * arm_task(arm).dt, 1),
                corruption=arm["corruption"],
                input_noise=arm["input_noise"], model=arm["model"], log=log, trained=now_iso())
    path = AdapterModules.save_module(user, calibration.module_path, "user", meta={"calibration": meta})
    check = "  ".join(f"{k} {v:.3f}" for k, v in log[-1].items() if k.startswith("heading_cos"))
    print(f"user module from {len(layouts)} calibration reaches ({meta['seconds']:.0f} s of them, {time.time() - tic:.0f}s of "
          f"training): {check} -> {path}", flush=True)


def run_protocol(args):
    protocol = config.load_protocol(args.protocol)
    sim = args.participant == "sim"
    agent = make_agent("expert", **protocol["expert"]) if sim else make_agent("human", allow_keyboard=args.allow_keyboard)
    results = Path(args.results) / protocol["study"]  # results/study_short/p1/, results/study_long/p1/
    device, common = device_for(args), (args.participant, args.session, agent, device_for(args), results, args.resume)
    sections, user = protocol_sections(protocol)  # a task section per task, its arms at their tuned settings
    calibration = None
    if args.calibrate and not args.prepare:  # a visit's scene tokens do not depend on whose user module plays
        calibration = Calibration(protocol, sections, *common, labels={"user": user})
        sections = calibration.with_module(sections)  # MARS plays the participant's own module
        print(f"{args.participant}, {calibration.run_name}: {len(calibration.layouts)} reaches + "
              f"{len(calibration.practice)} practice, layout seed {calibration.seed} -> {calibration.dir}; the module is "
              f"{calibration.module}" + (" (trained)" if calibration.ready else ""), flush=True)
    visit = Visit(protocol, sections, *common, order=protocol["cells"]["task_order"], labels={"user": user})
    runs = visit.runs
    for run in runs:
        print(f"{args.participant}, {run.run_name}: blocks {run.order}, {len(run.layouts)} trials + "
              f"{len(run.practice)} practice each, layout seed {run.seed} -> {run.dir}", flush=True)
    cache = Path(args.scene_cache) if args.scene_cache else CHECKPOINTS / "scene" / protocol["name"]
    factory, encoders = scene_encoders(device)
    if args.prepare:  # the scene tokens alone: checkpoints loaded, nothing written under results
        try:
            for run in runs:
                run.policies, run.adapters = load_checkpoints(run.arms, device)
            attach_scenes(runs, cache, factory)
        finally:
            for encoder in encoders:
                encoder.close()
        print(f"scene tokens ready in {cache}")
        return
    renderer = None

    def window():
        """The session's window, opened when the first thing is shown (a headless sim run has none)."""
        nonlocal renderer
        if renderer is None and (not sim or args.watch):
            renderer = make_renderer(title=f"{protocol['name']}: {args.participant}",
                                     blind=protocol["display"]["blind"] and not args.no_blind, cues=protocol["display"]["cues"],
                                     ready_hold_s=protocol["trial"]["ready_hold_s"],
                                     feedback_s=protocol["trial"]["feedback_s"], auto_continue_s=1.0 if sim else None)
        return renderer

    tic, played = time.time(), 0
    try:
        if calibration is not None and not calibration.ready:
            visit.check()  # a section that has trials stops the visit before its calibration is played
            calibration.start()
            played += calibration.run(window())
            train_user_module(calibration, device, window())
        visit.start()
        try:
            attach_scenes([run for run in runs if run.adapters is not None], cache, factory)
        finally:
            for encoder in encoders:  # pi0.5 leaves the GPU before the first trial
                encoder.close()
        played += visit.run(window())
        print(f"{played} trials in {time.time() - tic:.0f}s")
    except Aborted:
        print("aborted by the participant; re-run with --resume to continue")
    finally:
        if renderer is not None:
            renderer.close()
        if hasattr(agent, "close"):
            agent.close()
    for run in runs:
        if (run.dir / "manifest.json").exists():
            print_summary(run.dir)


def print_summary(run_dir):
    manifest, records = load_run(run_dir)
    trials = [r for r in records if not r["practice"] and r["outcome"] != "aborted"]
    print(f"{run_dir}: {len(trials)} trials" + (" (complete)" if manifest.get("finished") else ""))
    for arm in manifest["block_order"]:
        rows = [r for r in trials if r["arm"] == arm]
        if rows:
            s = summarize(rows)
            frames = [dt for r in rows for dt in r["frame_dt_ms"] or []]  # a rendered run: how long its frames took
            pace = (f"  frames {np.mean(frames):.1f} ms mean, {np.median(frames):.1f} ms median, "
                    f"{np.percentile(frames, 95):.1f} ms at the 95th percentile" if frames else "")
            print(f"  {arm:16s} {s['hits']:3d}/{s['n']:<3d} hits ({100 * s['success']:5.1f}%)  "
                  f"median {s['time_s'] if s['time_s'] is not None else float('nan'):.2f} s{pace}")


def main(argv=None):
    args = parse_args(argv)
    run_protocol(args)


if __name__ == "__main__":
    main()
