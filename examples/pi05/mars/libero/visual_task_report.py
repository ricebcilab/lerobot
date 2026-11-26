"""Paired held-out success rates and cluster bootstrap intervals for the visual-T pilot."""

import json
from collections import Counter, defaultdict

import numpy as np

BOWL = "akita_black_bowl_1"


def failure_category(trial):
    if trial["success"]:
        return "placed with violation" if trial["violation"] else "joint success"
    if BOWL in trial["picked"]:
        return "bowl on other goal" if trial["bowl_final_on"] else "bowl lifted, missed goal"
    if trial["picked"]:
        return "other object only"
    return "no verified pickup"


def diagnostic_table(conditions):
    header = [
        "Condition",
        "Trials",
        "Joint success",
        "Placed with violation",
        "Bowl on other goal",
        "Bowl lifted, missed goal",
        "Other object only",
        "No verified pickup",
    ]
    table = ["| " + " | ".join(header) + " |", "|---" + "|---:" * 7 + "|"]
    for label, trials in conditions.items():
        counts = Counter(failure_category(t) for t in trials)
        cells = [label, str(len(trials)), *(str(counts[k.lower()]) for k in header[2:])]
        table.append("| " + " | ".join(cells) + " |")
    return table


def audit_report(root, reference=None):
    manifest = json.loads((root / "manifest.json").read_text())
    expected = 2 * min(20, manifest["states"])
    arms = {
        arm: {p.name: json.loads(p.read_text()) for p in (root / "test" / arm).glob("*.json")}
        for arm in ("frs", "base")
    }
    verify = ("pair_id", "initial_state_hash", "center", "policy_seed", "operator_seed", "layout_seed")
    frs, base = arms.values()
    if (
        len(frs) != expected
        or frs.keys() != base.keys()
        or any(frs[n][k] != base[n][k] for n in frs for k in verify)
    ):
        raise ValueError("Incomplete or unmatched baseline trials")
    lines = [
        "# Disk pickup audit",
        "",
        f"{expected} paired resets per arm. Prompt: `{manifest.get('prompt', 'do something')}`.",
        "",
        "A verified pickup requires a lift of at least 2.5 cm and contact with both gripper finger pads at the same control step.",
        "",
        *diagnostic_table({arm: rr.values() for arm, rr in arms.items()}),
    ]
    if reference is not None:
        original = {p.name: json.loads(p.read_text()) for p in (reference / "test/frs").glob("*.json")}
        if original.keys() != frs.keys() or any(original[n][k] != frs[n][k] for n in frs for k in verify):
            raise ValueError("Replay and original disk FRS resets differ")
        agree = sum(all(original[n][k] == frs[n][k] for k in ("success", "violation")) for n in frs)
        lines += [
            "",
            f"Original FRS reset and layout match: {len(frs)}/{expected}. Placement and disk-violation agreement: {agree}/{expected}.",
        ]
    for task, goal in [(1, "stove"), (8, "plate")]:
        lines += [
            "",
            f"## {goal.capitalize()}",
            "",
            *diagnostic_table(
                {arm: [r for r in rr.values() if r["task_id"] == task] for arm, rr in arms.items()}
            ),
        ]
    path = root / "audit.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


def report(root, seeds):
    manifest = json.loads((root / "manifest.json").read_text())
    holder = manifest.get("obstacle") == "holder"
    baselines = ["frs", "base"] if holder else ["frs"]
    arms = [*baselines, *(f"{arm}_s{s}" for s in seeds for arm in ("masked", "visual", "mismatch"))]
    expected = 2 * min(20, manifest["states"])
    runs = {}
    rows = [
        "| Arm | Task | Layout | Trials | Placement | Violation | Joint success | Replan ms | Δ ms vs FRS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        trials = [json.loads(p.read_text()) for p in sorted((root / "test" / arm).glob("*.json"))]
        run = {
            (t["pair_id"], t["layout_seed"], t["initial_state_hash"], tuple(t["center"])): t for t in trials
        }
        if len(run) != expected or len(run) != len(trials) or (runs and run.keys() != runs["frs"].keys()):
            raise ValueError(f"Missing, duplicate, or unmatched trials for {arm}")
        runs[arm] = run
        for task in (1, 8):
            for blocked in (False, True):
                group = [r for r in trials if r["task_id"] == task and r["blocked"] == blocked]
                if group:
                    values = [
                        np.mean([t[k] for t in group]) for k in ("success", "violation", "joint", "replan_ms")
                    ]
                    baseline = [
                        r["replan_ms"]
                        for r in runs["frs"].values()
                        if r["task_id"] == task and r["blocked"] == blocked
                    ]
                    rows.append(
                        f"| {arm} | {task} | {'blocked' if blocked else 'clear'} | {len(group)} | "
                        f"{values[0]:.1%} | {values[1]:.1%} | {values[2]:.1%} | {values[3]:.1f} | {values[3] - np.mean(baseline):+.1f} |"
                    )
    summary = [
        "# Clean-operator visual task pilot" + (" with a solid utensil holder" if holder else ""),
        "",
        f"Held-out evaluation on {expected} paired resets. Adapter rates average the same resets "
        f"over {len(seeds)} training seeds. Plain FRS runs once per reset.",
        "",
        f"| Arm | Correct placement | {'Holder contact' if holder else 'Disk violation'} | Placement without violation |",
        "|---|---:|---:|---:|",
    ]
    for name in (*baselines, "masked", "visual", "mismatch"):
        selected = [run for arm, run in runs.items() if arm.split("_")[0] == name]
        rates = [
            np.mean([r[key] for run in selected for r in run.values()])
            for key in ("success", "violation", "joint")
        ]
        summary.append(f"| {name} | {rates[0]:.1%} | {rates[1]:.1%} | {rates[2]:.1%} |")
    if holder:
        summary += [
            "",
            f"Executing prompt: `{manifest['prompt']}`. A violation is any bowl or robot contact with the holder during physics integration.",
            "",
            "## Held-out baseline outcomes",
            "",
            "A verified pickup requires a 2.5 cm lift and simultaneous contact on both finger pads. Wrong-object-only counts exclude every verified bowl pickup.",
            "",
            *diagnostic_table({arm: runs[arm].values() for arm in baselines}),
        ]
        prompt_rows = {}
        for arm in baselines:
            for label, prompt in (("generic", "do something"), ("bowl", "place the black bowl somewhere")):
                name = f"{arm}_{label}"
                if arm == "frs" and prompt == manifest["prompt"]:
                    name = "frs_d4_p10"
                trials = [json.loads(p.read_text()) for p in (root / "dev" / name).glob("*.json")]
                if trials:
                    prompt_rows[f"{arm} / {prompt}"] = trials
        if len(prompt_rows) == 4:
            paired = [
                {(r["pair_id"], r["initial_state_hash"], tuple(r["center"])) for r in trials}
                for trials in prompt_rows.values()
            ]
            if len(paired[0]) != 2 * min(10, manifest["states"]) or any(p != paired[0] for p in paired):
                raise ValueError("Unmatched development prompt trials")
            summary += [
                "",
                "## Development prompt comparison",
                "",
                "FRS uses depth 4 and pinned prefix 10, fixed from the earlier disk pilot.",
                "",
                "| Condition | Trials | Verified bowl pickup | Correct placement | Holder contact | Joint success |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for label, trials in prompt_rows.items():
                values = [
                    len(trials),
                    sum(BOWL in r["picked"] for r in trials),
                    *(sum(r[k] for r in trials) for k in ("success", "violation", "joint")),
                ]
                summary.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
    videos = sorted((root / "test" / f"visual_s{seeds[0]}").glob("*.mp4"))
    if videos:
        links = []
        for video in videos:
            trial = json.loads(video.with_suffix(".json").read_text())
            destination = "stove" if trial["task_id"] == 1 else "plate"
            layout = "blocked" if trial["blocked"] else "clear"
            links.append(f"[{destination}, {layout}]({video.relative_to(root)})")
        summary.extend(
            [
                "",
                "Representative visual-T rollouts with external and wrist views. " + " · ".join(links),
            ]
        )
    rows = (
        summary
        + [
            "",
            "## Results by destination and layout",
            "",
            "Replan latency times steering after the operator command is ready. For the mismatch control, "
            "the counterfactual render and its encoding also happen before this timer.",
            "",
        ]
        + rows
    )
    rows.extend(
        [
            "",
            "Paired differences in joint success, percentage points (95% bootstrap interval). "
            "Within each destination, resample initial states with all their layouts and training seeds.",
            "",
        ]
    )
    rng = np.random.default_rng(0)
    for other in ("frs", "masked", "mismatch"):
        clusters = defaultdict(lambda: defaultdict(list))
        for key, r in runs["frs"].items():
            for seed in seeds:
                control = "frs" if other == "frs" else f"{other}_s{seed}"
                diff = int(runs[f"visual_s{seed}"][key]["joint"]) - int(runs[control][key]["joint"])
                clusters[r["task_id"]][r["init_state_id"]].append(diff)
        means, draws = [], []
        for states in clusters.values():
            values = np.array([np.mean(v) for v in states.values()])
            means.append(values.mean())
            draws.append(rng.choice(values, (5000, len(values)), replace=True).mean(axis=1))
        lo, hi = np.percentile(np.mean(draws, axis=0) * 100, [2.5, 97.5])
        rows.append(f"- Visual minus {other}: {np.mean(means) * 100:+.1f} ({lo:+.1f}, {hi:+.1f})")
    ckpt = root / "checkpoints" / f"visual_s{seeds[0]}" / "task.pt"
    from lerobot.policies.steering import AdapterModules

    meta = AdapterModules.read_checkpoint(ckpt, mmap=True)["meta"]
    rows.extend(
        [
            "",
            f"Visual task head: {meta['parameters']:,} parameters. Frozen checkpoint: {meta['policy']} at {meta['revision']}.",
        ]
    )
    report_path = root / "report.md"
    report_path.write_text("\n".join(rows) + "\n")
    print(report_path)


def camera_report(root, seeds):
    """Compare camera representations and counterfactual views on matched test resets."""
    conditions = {"FRS": [root / "test/frs"]}
    for stage in ("pre", "post"):
        for view in ("visual", "mismatch"):
            conditions[f"{stage} {view}"] = [root / "test" / f"{stage}_s{s}_{view}" for s in seeds]
    contrasts = [
        ("post visual", "pre visual"),
        ("pre visual", "pre mismatch"),
        ("post visual", "post mismatch"),
        ("pre visual", "FRS"),
        ("post visual", "FRS"),
    ]
    paired_report(
        root,
        seeds,
        conditions,
        contrasts,
        "Utensil holder camera features",
        "The mismatch condition renders the holder on the other route for the adapter alone.",
    )


def factorial_report(root, baseline, seeds, arms):
    conditions = {"baseline": [baseline / "test" / f"post_s{s}_visual" for s in seeds]}
    conditions.update({arm: [root / "test" / f"{arm}_s{s}" for s in seeds] for arm in arms})
    contrasts = [(arm, "baseline") for arm in arms]
    if "combined" in arms:
        contrasts.extend([("combined", "geometry"), ("combined", "mixed")])
    collection = json.loads((root / "collection.json").read_text())
    selected = json.loads((root / "selected.json").read_text())
    detail = [
        "All heads use the real views. Baseline rows come from the archived post-fusion checkpoints.",
        "",
        "## Teacher recoveries",
        "",
        "| Task and layout | Source trials | Contact trials | Accepted recoveries |",
        "|---|---:|---:|---:|",
        *(
            f"| {k} | {v['attempted']} | {v['contact']} | {v['accepted']} |"
            for k, v in sorted(collection.items())
        ),
        "",
        f"Mixed training used {len(selected)} selected recoveries, balanced between blocked plate and other layouts."
        if selected
        else "The mixed and combined conditions were skipped because one pool had fewer than eight accepted recoveries.",
    ]
    paired_report(
        root,
        seeds,
        conditions,
        contrasts,
        "Utensil holder geometry and recovery demonstrations",
        "\n".join(detail),
    )


def paired_report(root, seeds, conditions, contrasts, title, description):
    manifest = json.loads((root / "manifest.json").read_text())
    expected = 2 * min(manifest["states"], 20)
    reference = next(iter(conditions))
    fields = (
        "task_id",
        "init_state_id",
        "initial_state_hash",
        "layout_seed",
        "route",
        "center",
        "blocked",
        "env_seed",
        "policy_seed",
        "operator_seed",
        "input_noise_seed",
    )
    runs = {}
    summaries, layouts, contacts = [], [], []
    for label, arms in conditions.items():
        runs[label] = []
        for arm in arms:
            rows = [json.loads(p.read_text()) for p in arm.glob("*.json")]
            run = {r["pair_id"]: r for r in rows}
            if len(rows) != expected or len(run) != expected:
                raise ValueError(f"Incomplete or duplicate trials for {arm}")
            expected_arm = "frs" if label == "FRS" else "mismatch" if label.endswith("mismatch") else "visual"
            if any(r["prompt"] != manifest["prompt"] or r["arm"] != expected_arm for r in rows):
                raise ValueError(f"Incorrect prompt or arm for {arm}")
            if runs[reference]:
                ref = runs[reference][0]
                if ref.keys() != run.keys() or any(run[k][f] != ref[k][f] for k in run for f in fields):
                    raise ValueError(f"Unmatched test resets for {arm}")
            runs[label].append(run)
        rows = [r for run in runs[label] for r in run.values()]

        def counts(trials):
            return " | ".join(
                str(x)
                for x in [
                    len(trials),
                    *(sum(r[k] for r in trials) for k in ("success", "violation", "joint")),
                ]
            )

        first = Counter(
            r["first_contact"]["with"]
            for r in rows
            if r["task_id"] == 8 and r["blocked"] and r.get("first_contact")
        )
        contacts.append(f"| {label} | {first['robot']} | {first['bowl']} |")
        summaries.append(f"| {label} | {counts(rows)} |")
        for task, goal in ((1, "stove"), (8, "plate")):
            for blocked in (False, True):
                group = [r for r in rows if r["task_id"] == task and r["blocked"] == blocked]
                layouts.append(
                    f"| {label} | {goal} | {'blocked' if blocked else 'clear'} | {counts(group)} |"
                )

    differences = []
    for left, right in contrasts:
        rng = np.random.default_rng(0)
        resamples, deltas = [], []
        for task in (1, 8):
            ids = sorted(k for k, r in runs[reference][0].items() if r["task_id"] == task)
            delta = np.array(
                [
                    np.mean([r[k]["joint"] for r in runs[left]])
                    - np.mean([r[k]["joint"] for r in runs[right]])
                    for k in ids
                ]
            )
            deltas.extend(delta)
            resamples.append(rng.choice(delta, (10000, len(ids))).sum(axis=1))
        lo, hi = np.percentile(100 * np.sum(resamples, axis=0) / len(deltas), [2.5, 97.5])
        differences.append(
            f"| {left} minus {right} | {100 * np.mean(deltas):+.1f} | [{lo:+.1f}, {hi:+.1f}] |"
        )
    report = [
        f"# {title}",
        "",
        f"Training and execution used `{manifest['prompt']}`. The goal-aware operator supplied the destination.",
        f"The {expected} held-out resets are shared across all conditions and {len(seeds)} training seeds. "
        "Joint success requires correct placement without bowl or robot contact with the holder.",
        description,
        "",
        "| Condition | Trials | Correct placement | Holder contact | Joint success |",
        "|---|---:|---:|---:|---:|",
        *summaries,
        "",
        "## Paired differences in joint success",
        "",
        "95% percentile intervals resample resets within destination, keeping the trained heads paired.",
        "",
        "| Comparison | Difference (pp) | 95% interval (pp) |",
        "|---|---:|---|",
        *differences,
        "",
        "## By destination and holder layout",
        "",
        "| Condition | Goal | Layout | Trials | Correct placement | Holder contact | Joint success |",
        "|---|---|---|---:|---:|---:|---:|",
        *layouts,
        "",
        "## First holder contact on blocked plate",
        "",
        "| Condition | Robot | Bowl |",
        "|---|---:|---:|",
        *contacts,
    ]
    (root / "report.md").write_text("\n".join(report) + "\n")
