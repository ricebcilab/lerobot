"""The experiment mechanism: design, records, resume, and the real-time path (with a fake renderer)."""

import sys

import numpy as np
import pytest

from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.channel import InputChannel
from cursor_sa.records import OUTCOMES, load_run, read_records
from cursor_sa.session import (Aborted, Run, balanced_latin_square, block_order, block_permutation, layout_seed,
                               load_checkpoints, make_layout_sets, run_trials)
from cursor_sa.env import TaskConfig

PROTOCOL = dict(name="tiny", arms={"teleop": "teleop.yaml", "flow_control": "flow_control.yaml"},
                blocks={"trials": 3, "practice": 1, "order": "latin_square"},
                trial={"ready_hold_s": 0.0, "feedback_s": 0.0}, expert={"lag": 0, "slow_radius": 0.15},
                display={"blind": True})


def test_latin_square_is_balanced():
    for n in (2, 3, 4, 5):
        rows = balanced_latin_square(n)
        for row in rows:
            assert sorted(row) == list(range(n))
        for pos in range(n):  # every arm in every position equally often
            assert sorted(row[pos] for row in rows) == sorted(list(range(n)) * (len(rows) // n))


def test_block_order_by_participant_and_listed():
    arms = ["a", "b", "c", "d"]
    orders = {tuple(block_order(arms, f"p{i}")) for i in range(1, 5)}
    assert len(orders) == 4 and block_order(arms, "sim") == block_order(arms, "p4")  # 4 rows, wrap at 4
    assert block_order(arms, "p3", "listed") == arms


def test_block_order_advances_with_the_session(tmp_path):
    arms = ["a", "b", "c", "d"]
    assert block_order(arms, "p1", session=1) == block_order(arms, "p1")  # a first visit is what it always was
    orders = [block_order(arms, "p1", session=k) for k in range(1, 5)]
    assert len({tuple(o) for o in orders}) == 4
    for pos in range(4):  # over four visits every arm sits in every position once
        assert sorted(o[pos] for o in orders) == arms
    assert block_order(arms, "p2", session=1) == block_order(arms, "p1", session=2)
    assert block_order(arms, "p1", session=3, order="listed") == arms
    first, second = (Run(PROTOCOL, "p1", k, ExpertAgent(), "cpu", tmp_path).order for k in (1, 2))
    assert first == block_order(list(PROTOCOL["arms"]), "p1", session=1) and second != first


def test_layout_sets_depend_on_participant_and_session_only():
    task = TaskConfig()
    a, pa = make_layout_sets(layout_seed("p1", 1), 5, 2, (2, 8), task)
    b, _ = make_layout_sets(layout_seed("p1", 1), 5, 2, (2, 8), task)
    c, _ = make_layout_sets(layout_seed("p1", 2), 5, 2, (2, 8), task)
    d, _ = make_layout_sets(layout_seed("p2", 1), 5, 2, (2, 8), task)
    np.testing.assert_array_equal(a.targets, b.targets)
    assert not np.array_equal(a.targets, c.targets) and not np.array_equal(a.targets, d.targets)
    assert len(pa) == 2 and not np.array_equal(pa.start, a.start[:2])


def test_headless_run_writes_records_and_resumes(tmp_path):
    run = Run(PROTOCOL, "sim", 1, ExpertAgent(), "cpu", tmp_path)
    run.start()
    assert run.run() == 2 * (1 + 3)
    manifest, records = load_run(run.dir)
    assert manifest["block_order"] == run.order and set(manifest["checkpoints"]) == {"policies/flow_goaldrop"}
    assert len(records) == 8 and [r["practice"] for r in records[:4]] == [True, False, False, False]
    for r in records:
        assert r["outcome"] in OUTCOMES and r["agent"] == "expert" and r["frame_dt_ms"] is None
        assert len(r["path"]) == round(r["duration_s"] / 0.02) + 1 == len(r["raw_commands"]) + 1
        assert r["targets_range"] == [2, 8] and r["method"] in ("teleop", "flow_control") and r["obstacles"] == []
    by_block = {b: {r["layout_id"] for r in records if r["block"] == b and not r["practice"]} for b in (1, 2)}
    assert by_block[1] == by_block[2] == {0, 1, 2}  # the same layouts in every arm
    assert not np.array_equal(block_permutation(run.seed, 1, 40), block_permutation(run.seed, 2, 40))  # new order per block
    assert manifest["finished"] is not None

    # a second visit is refused without --resume; with it, only the missing trials are played
    with pytest.raises(FileExistsError):
        Run(PROTOCOL, "sim", 1, ExpertAgent(), "cpu", tmp_path).start()
    (run.dir / f"block2_{run.order[1]}.jsonl").unlink()
    again = Run(PROTOCOL, "sim", 1, ExpertAgent(), "cpu", tmp_path, resume=True)
    again.start()
    assert again.run() == 4
    assert len(load_run(run.dir)[1]) == 8


class FakeRenderer:
    """Counts the real-time hooks; no window."""

    def __init__(self):
        self.calls = []

    def block_screen(self, info, agent):
        self.calls.append("block")

    def break_screen(self, info, agent):
        self.calls.append("break")

    def ready_phase(self, ctx):
        self.calls.append("ready")

    def frame(self, ctx):
        self.calls.append("frame")

    def feedback_phase(self, ctx):
        self.calls.append("feedback")


def test_realtime_path_runs_one_trial_at_a_time(tmp_path):
    renderer = FakeRenderer()
    run = Run(PROTOCOL, "p1", 1, ExpertAgent(), "cpu", tmp_path)
    run.start()
    assert run.run(renderer) == 8
    assert renderer.calls.count("block") == 2 and renderer.calls.count("ready") == 8 and renderer.calls.count("break") == 2
    records = load_run(run.dir)[1]
    assert all(len(r["frame_dt_ms"]) == len(r["raw_commands"]) for r in records)


def test_abort_logs_the_partial_trial(tmp_path):
    class AbortingRenderer(FakeRenderer):
        def frame(self, ctx):
            if len(ctx["env"].pos) and ctx["env"].t >= 5:
                raise Aborted()

    run = Run(PROTOCOL, "p2", 1, ExpertAgent(), "cpu", tmp_path)
    run.start()
    with pytest.raises(Aborted):
        run.run(AbortingRenderer())
    records = read_records(run.dir / f"block1_{run.order[0]}.jsonl")
    assert len(records) == 1 and records[0]["outcome"] == "aborted" and len(records[0]["path"]) == 6


def test_run_samples_layouts_from_the_arms_task(tmp_path):
    base = config.CONFIG_DIR / "teleop.yaml"
    (tmp_path / "near.yaml").write_text(f"extends: {base}\ntask: {{targets: [3, 3], reach: [0.5, 0.6]}}\n")
    (tmp_path / "near_fc.yaml").write_text(f"extends: {tmp_path / 'near.yaml'}\ncontrol: {{mode: flow_control}}\n")
    protocol = dict(PROTOCOL, arms={"a": str(tmp_path / "near.yaml"), "b": str(tmp_path / "near_fc.yaml")})
    run = Run(protocol, "p1", 1, ExpertAgent(), "cpu", tmp_path)
    assert run.layouts.mask.sum(1).tolist() == [3, 3, 3]
    reach = np.linalg.norm(run.layouts.goal_pos - run.layouts.start, axis=1)
    assert (0.5 <= reach).all() and (reach <= 0.6).all()
    # arms of one protocol must agree on the layouts, since every arm plays the same ones
    (tmp_path / "far.yaml").write_text(f"extends: {base}\ntask: {{targets: [3, 3], reach: [0.7, null]}}\n")
    with pytest.raises(ValueError, match="same task layouts"):
        Run(dict(protocol, arms={"a": str(tmp_path / "near.yaml"), "b": str(tmp_path / "far.yaml")}), "p1", 1,
            ExpertAgent(), "cpu", tmp_path)


def test_run_trials_batched_matches_single(tmp_path):
    arm = config.load_arm("study_frs.yaml")
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    layouts, _ = make_layout_sets(7, 4, 0, (2, 4), TaskConfig())
    channel = InputChannel(arm["corruption_matrix"], 0.0)
    recs = run_trials(arm, layouts, ExpertAgent(), channel, policies[arm["model"]], adapters, "cpu")
    assert len(recs) == 4 and all(r["served_commands"] != r["raw_commands"] for r in recs)  # the gain corrupts them
    for r in recs:
        assert r["targets"][r["goal"]] is not None and 2 <= r["n_targets"] <= 4


def test_logged_copies_what_is_printed_into_the_log_and_restores_the_console(tmp_path, capsys):
    from cursor_sa.records import logged
    stdout = sys.stdout
    with logged(tmp_path / "log.txt"):
        print("inside")
        print("an error", file=sys.stderr)
    print("outside")
    assert sys.stdout is stdout
    assert (tmp_path / "log.txt").read_text() == "inside\nan error\n"
    with logged(tmp_path / "log.txt", append=True):
        print("resumed")
    assert (tmp_path / "log.txt").read_text().endswith("resumed\n")
    assert "inside" in capsys.readouterr().out
