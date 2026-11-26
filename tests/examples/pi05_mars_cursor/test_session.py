"""The experiment mechanism: design, records, resume, and the real-time path (with a fake renderer)."""

import sys

import numpy as np
import pytest
import torch

from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.policy import FlowPolicy, as_policy
from cursor_sa.records import OUTCOMES, load_run, read_records
from cursor_sa.session import (Aborted, Run, Visit, balanced_latin_square, block_order, block_permutation, layout_seed,
                               make_layout_sets, task_order)
from cursor_sa.steering import SharedController, steered_chunk
from lerobot.envs.cursor import TaskConfig, sample_layouts
from lerobot.policies.steering import SteeringPolicy

PROTOCOL = dict(name="tiny", blocks={"trials": 3, "practice": 1, "order": "latin_square"},
                trial={"ready_s": 0.0, "feedback_s": 0.0}, expert={"lag": 0, "slow_radius": 0.15},
                display={"blind": True})


def _arms(**task):
    """Two arms on one task (the default one, or `task`'s overrides of it)."""
    return {"teleop": config.load_arm("base.yaml", {"mode": "teleop", **task}), "flow_control": config.load_arm("base.yaml", task)}


def test_block_order_is_balanced_by_participant_and_session(tmp_path):
    for n in (2, 3, 4, 5):
        rows = balanced_latin_square(n)
        for row in rows:
            assert sorted(row) == list(range(n))
        for pos in range(n):  # every arm in every position equally often
            assert sorted(row[pos] for row in rows) == sorted(list(range(n)) * (len(rows) // n))
    arms = ["a", "b", "c", "d"]
    assert len({tuple(block_order(arms, f"p{i}")) for i in range(1, 5)}) == 4
    assert block_order(arms, "sim") == block_order(arms, "p4") and block_order(arms, "p3", "listed") == arms  # wrap at 4
    assert block_order(arms, "p1", session=1) == block_order(arms, "p1")  # a first visit is what it always was
    orders = [block_order(arms, "p1", session=k) for k in range(1, 5)]
    for pos in range(4):  # over four visits every arm sits in every position once
        assert sorted(o[pos] for o in orders) == arms
    assert block_order(arms, "p2", session=1) == block_order(arms, "p1", session=2)
    assert block_order(arms, "p1", session=3, order="listed") == arms
    first, second = (Run(PROTOCOL, "p1", k, ExpertAgent(), "cpu", tmp_path, arms=_arms()).order for k in (1, 2))
    assert first == block_order(list(_arms()), "p1", session=1) and second != first


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
    run = Run(PROTOCOL, "sim", 1, ExpertAgent(), "cpu", tmp_path, arms=_arms())
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
        Run(PROTOCOL, "sim", 1, ExpertAgent(), "cpu", tmp_path, arms=_arms()).start()
    (run.dir / f"block2_{run.order[1]}.jsonl").unlink()
    again = Run(PROTOCOL, "sim", 1, ExpertAgent(), "cpu", tmp_path, resume=True, arms=_arms())
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


def test_realtime_path_runs_one_trial_at_a_time_and_logs_an_abort(tmp_path):
    renderer = FakeRenderer()
    run = Run(PROTOCOL, "p1", 1, ExpertAgent(), "cpu", tmp_path, arms=_arms())
    run.start()
    assert run.run(renderer) == 8
    assert renderer.calls.count("block") == 2 and renderer.calls.count("ready") == 8 and renderer.calls.count("break") == 2
    assert all(len(r["frame_dt_ms"]) == len(r["raw_commands"]) for r in load_run(run.dir)[1])

    class AbortingRenderer(FakeRenderer):
        def frame(self, ctx):
            if len(ctx["env"].pos) and ctx["env"].t >= 5:
                raise Aborted()

    run = Run(PROTOCOL, "p2", 1, ExpertAgent(), "cpu", tmp_path, arms=_arms())
    run.start()
    with pytest.raises(Aborted):
        run.run(AbortingRenderer())
    records = read_records(run.dir / f"block1_{run.order[0]}.jsonl")
    assert len(records) == 1 and records[0]["outcome"] == "aborted" and len(records[0]["path"]) == 6


def test_task_order_keeps_the_first_task_and_rotates_block_rows():
    """A visit over several tasks: the first task (the open field) always first, the others in a balanced Latin
    square row by participant and visit; each task section takes the next row of the block order's square."""
    tasks = ["T1", "T2", "T3", "T4", "T5"]
    orders = [task_order(tasks, f"p{i}") for i in range(1, 5)]
    assert all(o[0] == "T1" and sorted(o) == tasks for o in orders)
    for pos in range(1, 5):  # over four participants every later task sits in every position once
        assert sorted(o[pos] for o in orders) == tasks[1:]
    assert task_order(tasks, "p3", "listed") == tasks and task_order(tasks, "p1", session=2) == task_order(tasks, "p2")
    arms = ["a", "b", "c", "d"]
    rows = [block_order(arms, "p1", shift=j) for j in range(4)]
    assert rows[0] == block_order(arms, "p1") and len({tuple(r) for r in rows}) == 4
    seeds = [layout_seed("p1", 1), layout_seed("p1", 1, "T1"), layout_seed("p1", 1, "T2")]
    assert len(set(seeds)) == 3 and seeds[0] == layout_seed("p1", 1, None)  # a single-task run's seed is what it was


class VisitRenderer(FakeRenderer):
    """Keeps what the block, break and section screens are told."""

    def __init__(self):
        super().__init__()
        self.info = {"block": [], "break": [], "section": []}

    def block_screen(self, info, agent):
        self.info["block"].append(info)

    def break_screen(self, info, agent):
        self.info["break"].append(info)

    def section_screen(self, info, agent):
        self.info["section"].append(info)


def test_a_visit_plays_every_section_and_resumes_across_them(tmp_path):
    """A visit of two task sections: a run folder per section (its own layouts, block order and manifest), records
    that name their task and user, block numbers that count over the visit, one "done" at its end; a resume plays
    only what is missing and refuses arms that changed."""
    def arms(obstacle, tau=4):
        return {"teleop": config.load_arm("base.yaml", {"mode": "teleop", "obstacle": obstacle}),
                "flow_control": config.load_arm("base.yaml", {"obstacle": obstacle, "tau": tau})}

    disc = {"kind": "disc", "radius": 0.03, "along": 0.5}
    sections = {"T1": arms(None), "T2": arms(disc)}
    def make(sections=sections, **kw):
        return Visit(PROTOCOL, sections, "p1", 1, ExpertAgent(), "cpu", tmp_path, labels={"user": "U1"}, **kw)
    visit, renderer = make(), VisitRenderer()
    visit.start()
    assert visit.run(renderer) == 2 * 2 * (1 + 3)
    assert [r.dir.name for r in visit.runs] == ["tiny_s1_T1", "tiny_s1_T2"]
    assert visit.runs[0].seed != visit.runs[1].seed and visit.runs[0].order != visit.runs[1].order  # the next row
    for run, task in zip(visit.runs, ("T1", "T2"), strict=True):
        manifest, records = load_run(run.dir)
        assert manifest["section"]["task"] == task and manifest["section"]["user"] == "U1" and manifest["finished"]
        assert len(records) == 8 and {(r["task"], r["user"]) for r in records} == {(task, "U1")}
        assert (records[0]["obstacle"] is None) == (task == "T1")
    assert [i["task"] for i in renderer.info["section"]] == ["T1", "T2"]
    assert [(i["block"], i["n_blocks"]) for i in renderer.info["block"]] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [i["last"] for i in renderer.info["break"]] == [False, False, False, True]
    assert [i["section_end"] for i in renderer.info["break"]] == [False, True, False, False]
    alone, quiet = Visit(PROTOCOL, {"T1": arms(None)}, "p2", 1, ExpertAgent(), "cpu", tmp_path), VisitRenderer()
    alone.start()
    assert alone.run(quiet) == 8 and quiet.info["section"] == [] and len(quiet.info["block"]) == 2  # one task: no task screen

    finished = load_run(visit.runs[0].dir)[0]["finished"]
    with pytest.raises(FileExistsError):
        make().start()
    (visit.runs[1].dir / f"block2_{visit.runs[1].order[1]}.jsonl").unlink()
    again = make(resume=True)
    again.start()
    assert again.run() == 4 and load_run(visit.runs[0].dir)[0]["finished"] == finished  # the complete section untouched
    with pytest.raises(ValueError, match="different arms"):
        make({"T1": arms(None, tau=8), "T2": arms(disc)}, resume=True).start()


def test_run_samples_layouts_from_the_arms_task(tmp_path):
    near = {"targets": [3, 3], "reach": [0.5, 0.6]}
    run = Run(PROTOCOL, "p1", 1, ExpertAgent(), "cpu", tmp_path, arms=_arms(**near))
    assert run.layouts.mask.sum(1).tolist() == [3, 3, 3]
    reach = np.linalg.norm(run.layouts.goal_pos - run.layouts.start, axis=1)
    assert (0.5 <= reach).all() and (reach <= 0.6).all()
    # the arms of a run must agree on the layouts, since every arm plays the same ones
    far = config.load_arm("base.yaml", {"targets": [3, 3], "reach": [0.7, None]})
    with pytest.raises(ValueError, match="same task layouts"):
        Run(PROTOCOL, "p1", 1, ExpertAgent(), "cpu", tmp_path, arms={**_arms(**near), "flow_control": far})


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


# ---------------------------------------------------------------- the controller's steering policy


@pytest.fixture
def small_policy():
    torch.manual_seed(0)
    return as_policy(FlowPolicy(d_model=32, layers=1, heads=2, d_ff=64).eval())


@pytest.fixture
def layouts():
    return sample_layouts(np.random.default_rng(1), 3, (2, 4), TaskConfig())


def test_the_controller_runs_a_steering_policy(small_policy, layouts):
    ctl = SharedController(small_policy, layouts, "frs", n_reverse_steps=4, device="cpu")
    assert isinstance(ctl.steering, SteeringPolicy) and ctl.steering.config.mode == "frs"
    assert ctl.steering.base_policy is as_policy(small_policy) or ctl.steering.base_policy.model is small_policy.model


def test_policy_mode_is_mode_none(small_policy, layouts):
    ctl = SharedController(small_policy, layouts, "policy", device="cpu")
    assert ctl.steering.config.mode == "none"


@pytest.mark.parametrize("mode", ["frs"])
def test_steered_chunk_matches_the_controller_bit_for_bit(small_policy, layouts, mode):
    """The controller's chunk (screen units per step) is the example's steered_chunk for the same observation and
    command under one seed; one row idle."""
    ctl = SharedController(small_policy, layouts, mode, n_reverse_steps=4, device="cpu")
    command = np.array([[0.8, 0.1], [0.0, 0.01], [-0.5, 0.6]])
    torch.manual_seed(0)
    ctl.act(command)
    torch.manual_seed(0)
    want = steered_chunk(small_policy, ctl.observation(), torch.as_tensor(command, dtype=torch.float32), mode,
                         n_reverse_steps=4)
    assert np.array_equal(ctl.chunk, want.numpy() * 1)
