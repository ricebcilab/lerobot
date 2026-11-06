"""Rigid motions of layouts and actions, the rigid copies of the demonstrations (`augment_demos`), the sampler's
jitter, and the layout diagnostic (tools/diag_layouts.py) built on them."""

import json

import numpy as np
import pytest
import torch
from cursor_sa import demos
from cursor_sa.demos import DemoSampler, augment_demos, demo_positions, make_demos, replay
from cursor_sa.env import (
    Layouts,
    TaskConfig,
    layouts_on_screen,
    move_actions,
    move_layouts,
    obstacle_spec,
    sample_layouts,
)
from cursor_sa.policy import FlowPolicy
from cursor_sa.readout import corruption_spec
from diag_layouts import main as diag_main, moved_layouts, rebuild_layouts
from test_sweep import _image_adapter
from train import train_adapter

STUDY = {"min_reach": 0.35, "max_reach": 0.5}
TASK = TaskConfig(
    **STUDY, obstacle=obstacle_spec({"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.05})
)
FREE = TaskConfig(**STUDY)
DISC = "{kind: disc, radius: 0.08, along: 0.5}"
FIELDS = ("start", "targets", "mask", "goal", "obstacles", "obstacle_mask")


def _demos(task=TASK, n=6, seed=0):
    rng = np.random.default_rng(seed)
    lay = sample_layouts(rng, n, 4, task)
    return (lay, *make_demos(rng, lay, task))


def _one(start, goal, discs=()):
    """A one-trial layout from a start, a goal and optional discs (x, y, radius)."""
    discs = np.array(discs, float).reshape(1, -1, 3)
    mask = np.ones(discs.shape[:2], bool)
    start, goal = np.array([start], float), np.array([[goal]], float)
    return Layouts(start, goal, np.ones((1, 1), bool), np.array([0]), discs, mask)


def _lay():
    rng = np.random.default_rng(0)
    targets = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]], bool)
    discs = np.concatenate([rng.uniform(0.2, 0.8, (3, 2, 2)), rng.uniform(0.02, 0.08, (3, 2, 1))], 2)
    start, points = rng.uniform(0.2, 0.8, (3, 2)), rng.uniform(0.2, 0.8, (3, 4, 2))
    return Layouts(
        start, points, targets, np.array([0, 1, 2]), discs, np.array([[1, 1], [1, 0], [0, 0]], bool)
    )


def _dist(points):
    return np.linalg.norm(points[:, :, None] - points[:, None], axis=-1)


def reach(a, b):
    return np.linalg.norm(a - b, axis=-1)


def test_a_motion_is_rigid_and_moves_the_actions_with_the_start():
    lay, rng = _lay(), np.random.default_rng(1)
    lay.scene = {"x": np.zeros((3, 2))}
    zero = move_layouts(lay, np.zeros(3), np.zeros((3, 2)), np.zeros(3, bool))
    assert np.allclose(zero.start, lay.start) and np.allclose(zero.obstacles, lay.obstacles)
    assert (zero.mask == lay.mask).all() and (zero.obstacle_mask == lay.obstacle_mask).all()
    assert zero.scene is None
    one = _one((0.7, 0.5), (0.5, 0.8))
    turned = move_layouts(one, np.array([np.pi / 2]), np.zeros((1, 2)), np.array([False]))
    assert np.allclose(turned.start[0], (0.5, 0.7)) and np.allclose(turned.goal_pos[0], (0.2, 0.5))
    mirrored = move_layouts(one, np.zeros(1), np.array([[0.1, -0.1]]), np.array([True]))
    assert np.allclose(mirrored.start[0], (0.4, 0.4))  # about the centre, then shifted
    angle, shift = rng.uniform(-np.pi, np.pi, 3), rng.uniform(-0.1, 0.1, (3, 2))
    mirror = np.array([1, 0, 1], bool)
    moved = move_layouts(lay, angle, shift, mirror)
    before, after = ([x.start[:, None], x.targets, x.obstacles[..., :2]] for x in (lay, moved))
    assert np.allclose(_dist(np.concatenate(before, 1)), _dist(np.concatenate(after, 1)))
    assert (moved.obstacles[..., 2] == lay.obstacles[..., 2]).all()
    actions = rng.uniform(-0.05, 0.05, (3, 6, 2))
    path = lay.start[:, None] + np.cumsum(actions, 1)
    walk = Layouts(lay.start, path, np.ones((3, 6), bool), np.zeros(3, int))  # the path as targets
    walked = move_layouts(walk, angle, shift, mirror)
    steps = np.cumsum(move_actions(actions, angle, mirror), 1)
    assert np.allclose(walked.start[:, None] + steps, walked.targets)
    free = move_layouts(Layouts(lay.start, lay.targets, lay.mask, lay.goal), angle, shift, mirror)  # no discs
    assert free.obstacles.shape == (3, 0, 3) and np.array_equal(free.start, moved.start)
    none = move_layouts(lay[:0], np.zeros(0), np.zeros((0, 2)), np.zeros(0, bool))
    assert none.start.shape == (0, 2) and none.obstacles.shape == (0, 2, 3)


def test_on_screen_needs_the_discs_and_targets_inside_the_edge():
    edge_disc = _one((0.5, 0.5), (0.8, 0.5), [[0.05, 0.5, 0.08]])
    inward = move_layouts(edge_disc, np.zeros(1), np.array([[0.1, 0.0]]), np.zeros(1, bool))
    assert not layouts_on_screen(edge_disc)[0] and layouts_on_screen(inward)[0]
    assert not layouts_on_screen(_one((0.5, 0.5), (0.995, 0.5)))[0]
    padded = _lay()
    padded.targets[1, 2:], padded.obstacles[2] = 5.0, 5.0  # padding targets and padding discs do not count
    assert layouts_on_screen(padded).all()
    near = _one((0.03, 0.5), (0.5, 0.5), [(0.5, 0.7, 0.08)])  # clears the edge, not the sampler's 0.05 margin
    assert layouts_on_screen(near).all() and not layouts_on_screen(near, margin=0.05).all()
    disc = _one((0.5, 0.5), (0.5, 0.2), [(0.5, 0.75, 0.08)])
    assert layouts_on_screen(disc, margin=0.05).all()  # the margin is not the discs'
    disc.obstacles[0, 0, 1] = 0.92  # a disc edge at 0.99
    assert not layouts_on_screen(disc).all()


def test_copies_are_rigid_images_of_their_originals_and_the_originals_come_first():
    for task in (FREE, TASK):
        lay, actions, lengths = _demos(task)
        push = np.random.default_rng(2).normal(size=actions.shape)
        rng = np.random.default_rng(1)
        out, acts, lens, kept, moved = augment_demos(rng, lay, actions, lengths, 2, task, push=push)
        assert kept.dtype.kind == "i" and kept.sum() > 0 and (kept <= 2).all()
        assert len(out) == len(acts) == len(lens) == len(moved) == 6 + kept.sum()
        assert all(np.array_equal(getattr(out, name)[:6], getattr(lay, name)) for name in FIELDS)
        assert np.array_equal(acts[:6], actions) and np.array_equal(lens[:6], lengths)
        assert np.array_equal(moved[:6], push)
        env = replay(acts[6:], out[6:], task)
        assert env.success.all() and np.array_equal(lens[6:], env.steps)
        path = demo_positions(out[6:], acts[6:])  # the start plus the running sum of the actions
        assert np.array_equal(path[:, 1:], out[6:].start[:, None] + np.cumsum(acts[6:], 1))
        assert np.array_equal(path[:, 0], out[6:].start) and ((path >= 0) & (path <= 1)).all()
        assert layouts_on_screen(out[6:], margin=task.margin).all()
        origin = np.repeat(np.arange(6), kept)  # the copies come grouped by original, in original order
        copy, base = out[6:], lay[origin]
        assert np.allclose(reach(copy.start, copy.goal_pos), reach(base.start, base.goal_pos))
        assert np.allclose(np.linalg.norm(acts[6:], axis=-1), np.linalg.norm(actions[origin], axis=-1))
        # a rigid motion keeps dot products: per step, moved push . moved action == the original's
        dots = (push[origin] * actions[origin]).sum(-1)
        assert np.allclose((moved[6:] * acts[6:]).sum(-1), dots, atol=1e-9)
    for a, b in ((copy.start, base.start), (copy.goal_pos, base.goal_pos)):  # the disc where it was
        assert np.allclose(reach(a, copy.obstacles[:, 0, :2]), reach(b, base.obstacles[:, 0, :2]))
    assert np.array_equal(copy.obstacles[..., 2], base.obstacles[..., 2])
    assert augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, task)[4] is None
    with pytest.raises(ValueError, match="push"):
        augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, task, push=push[:3])


def test_motions_are_drawn_angle_then_shift_then_mirror_whatever_the_cursor_noise():
    lay, actions, lengths = _demos(FREE, n=8)
    draw = np.random.default_rng(3)
    angle, shift, mirror = (
        draw.uniform(0, 2 * np.pi, 8),
        draw.uniform(-0.1, 0.1, (8, 2)),
        draw.random(8) < 0.5,
    )
    for task in (FREE, TaskConfig(**STUDY, cursor_noise=0.05)):
        rng = np.random.default_rng(3)
        out, acts, lens, kept, _ = augment_demos(rng, lay, actions, lengths, 1, task, shift=0.1, tries=1)
        want = move_layouts(lay, angle, shift, mirror)[kept == 1]
        assert kept.sum() > 0 and np.array_equal(out.start[8:], want.start)
        assert np.array_equal(out.targets[8:], want.targets) and np.array_equal(acts[:8], actions)


def test_replays_run_500_rows_at_a_time(monkeypatch):
    lay, actions, lengths = _demos(FREE)
    rows = []
    monkeypatch.setattr(demos, "replay", lambda moved, *a: rows.append(len(moved)) or replay(moved, *a))
    many = (lay.tile(700), np.tile(actions, (700, 1, 1)), np.tile(lengths, 700))
    out = augment_demos(np.random.default_rng(1), *many, 1, FREE, tries=1)[0]
    assert len(rows) >= 2 and max(rows) == 500 and len(out) > 4200


@pytest.mark.parametrize("case", ["k=0", "all failed", "one failed"])
def test_nothing_to_copy_returns_the_originals(case):
    lay, actions, lengths = _demos(TASK)
    lengths = np.zeros_like(lengths) if case == "all failed" else lengths.copy()
    lengths[2] = 0  # a failed demonstration gets no copies
    push, rng = np.ones_like(actions), np.random.default_rng(1)
    state = rng.bit_generator.state
    k = 0 if case == "k=0" else 2
    out, acts, lens, kept, moved = augment_demos(rng, lay, actions, lengths, k, TASK, push=push)
    if case == "one failed":
        assert kept[2] == 0 and kept.sum() > 0 and 2 not in np.repeat(np.arange(6), kept)
        assert (lens[6:] > 0).all()
        return
    assert np.array_equal(out.start, lay.start) and np.array_equal(acts, actions)
    assert np.array_equal(lens, lengths)
    assert np.array_equal(moved, push) and kept.tolist() == [0] * 6 and rng.bit_generator.state == state


def test_a_copy_is_refused_off_the_screen_or_without_success():
    # a reach across the whole screen: few motions keep it on screen, so the copies run out short of k
    start, goal = np.array([0.02, 0.5]), np.array([0.98, 0.5])
    targets = np.array([[goal, (0.3, 0.2), (0.7, 0.2), (0.5, 0.8)]])
    lay = Layouts(start[None], targets, np.ones((1, 4), bool), np.array([0]))
    actions = np.concatenate([np.tile((goal - start) / 60, (1, 60, 1)), np.zeros((1, 30, 2))], 1)
    lengths = replay(actions, lay, FREE).steps
    out, acts, _, kept, _ = augment_demos(np.random.default_rng(0), lay, actions, lengths, 8, FREE, tries=1)
    assert lengths[0] < FREE.max_steps and kept[0] < 8 and len(out) == 1 + kept[0]
    # out to x = 0.96 and back to a goal beside the start: the layout has room to move, the path barely has
    start, far, goal = np.array([0.45, 0.5]), np.array([0.96, 0.5]), np.array([0.55, 0.5])
    leg = np.linspace(0, 1, 26)[None, :, None]
    legs = np.diff(leg * (far - start), axis=1), np.diff(leg * (goal - far), axis=1), np.zeros((1, 30, 2))
    actions = np.concatenate(legs, 1)
    lay = Layouts(start[None], goal[None, None], np.ones((1, 1), bool), np.array([0]))
    lengths = replay(actions, lay, FREE).steps
    out, acts, _, kept, _ = augment_demos(np.random.default_rng(0), lay, actions, lengths, 200, FREE, tries=1)
    path = demo_positions(out[1:], acts[1:])
    assert 0 < kept[0] < 200 and ((path >= 0) & (path <= 1)).all()
    lay, actions, lengths = _demos(FREE)  # layouts and path fit, but the cursor never moves: no copy succeeds
    still = augment_demos(np.random.default_rng(1), lay, np.zeros_like(actions), lengths, 2, FREE, tries=1)
    assert still[3].sum() == 0


def test_jitter_zero_draws_nothing():
    lay, actions, lengths = _demos(FREE)
    sampler = DemoSampler(lay, actions, lengths, 8, 2, "cpu", jitter=0.0)
    # the three documented draws, by hand: the demos, the steps k and the goal dropout
    torch.manual_seed(0)
    i = sampler.valid[torch.randint(len(sampler.valid), (8,))]
    k = (torch.rand(8) * sampler.lengths[i]).long()
    torch.rand(8, 1)
    three = torch.get_rng_state()
    delta = 0.02 * torch.randn(8, 2)
    four = torch.get_rng_state()

    torch.manual_seed(0)
    obs, chunk, got_i, got_k = sampler.sample(8, indexed=True)
    assert torch.equal(torch.get_rng_state(), three)  # jitter 0: not one draw past the three
    assert torch.equal(got_i, i) and torch.equal(got_k, k) and torch.equal(obs["pos"], sampler.pos[i, k])

    torch.manual_seed(0)
    jittered = DemoSampler(lay, actions, lengths, 8, 2, "cpu", jitter=0.02)
    obs, _ = jittered.sample(8)
    assert not torch.equal(torch.get_rng_state(), three) and torch.equal(torch.get_rng_state(), four)
    assert torch.equal(obs["pos"], sampler.pos[i, k] + delta)


def test_jitter_displaces_the_position_and_the_chunk_returns_to_the_path():
    lay, actions, lengths = _demos(FREE)
    for horizon in (8, 2):  # a chunk shorter than jitter_steps corrects over what there is
        s = DemoSampler(lay, actions, lengths, horizon, 2, "cpu", jitter=0.02, jitter_steps=4)
        torch.manual_seed(0)
        obs, chunk, i, k = s.sample(4096, indexed=True)
        ref = DemoSampler(lay, actions, lengths, horizon, 2, "cpu")
        delta, window = obs["pos"] - ref.pos[i, k], ref.actions[i[:, None], k[:, None] + ref.window]
        base, n = window[:, 2:], min(horizon, 4)
        assert abs(delta.std() - 0.02) < 0.002 and torch.equal(obs["hist"], window[:, :2])  # hist untouched
        assert torch.allclose(chunk[:, :n], base[:, :n] - delta[:, None] / n, atol=1e-6)  # spread evenly
        assert torch.equal(chunk[:, n:], base[:, n:])
    # at a hold step the label is all zeros and the correction alone brings the cursor back onto the goal
    goal = torch.as_tensor(lay.goal_pos, dtype=torch.float32)[i]
    held = (base == 0).all(2).all(1) & ((ref.pos[i, k] - goal).norm(dim=1) < 1e-6)
    assert held.sum() > 0 and torch.allclose(obs["pos"][held] + chunk[held].sum(1), goal[held], atol=1e-5)
    for bad in ({"jitter": -0.01}, {"jitter_steps": 0}):
        with pytest.raises(ValueError, match="jitter"):
            DemoSampler(lay, actions, lengths, 8, 2, "cpu", **bad)


def test_the_diagnostics_layouts_are_the_trainers_and_move_rigidly():
    task = ["--kind", "task", "--obstacle", DISC, "--name", "t", "--user-module", "u", "--val-demos", "4"]
    args = train_adapter.parse_args([*task, "--corruption", "{rotation_deg: 60}", "--n-demos", "60"])
    expert = (FlowPolicy().eval(), "cpu", np.random.default_rng(0), *corruption_spec(args.corruption), 0.25)
    train = train_adapter.expert_splits(args, args.obstacle, *expert)["train"].sampler  # the trainer's draw
    lay = rebuild_layouts(0, 60, 0.25, obstacle_spec(args.obstacle))
    assert len(lay) == 60 and lay.obstacle_mask.any(1).sum() == 45  # 15 without the obstacle, at the end
    assert np.allclose(lay.start, train.pos[:, 0].numpy()) and np.allclose(
        lay.obstacles, train.obstacles.numpy()
    )
    lay = lay[:40]
    moved = moved_layouts(lay, np.random.default_rng(1))
    assert moved.scene is None and layouts_on_screen(moved).all() and np.array_equal(moved.goal, lay.goal)
    assert np.array_equal(moved.obstacle_mask, lay.obstacle_mask)
    assert np.array_equal(moved.obstacles[..., 2], lay.obstacles[..., 2])
    for b in range(40):  # every pairwise distance kept
        before, after = (
            [x.start[b], *x.targets[b, x.mask[b]], *x.real_obstacles(b)[:, :2]] for x in (lay, moved)
        )
        assert np.allclose(_dist(np.array([before]))[0], _dist(np.array([after]))[0])
    assert (np.linalg.norm(moved.start - lay.start, axis=1) > 1e-6).mean() > 0.9  # the rows did move
    kept = moved_layouts(lay, np.random.default_rng(1), tries=0)  # no round: every row keeps its original
    assert np.array_equal(kept.start, lay.start) and np.array_equal(kept.obstacles, lay.obstacles)


def _diag_sweep(tmp_path, rel="nowhere"):
    """A users x tasks sweep with a disc task T2 and an open-field task T0 (with `rel` a missing module, a refusal
    comes first)."""
    (tmp_path / "diag.yaml").write_text(
        "name: tiny_diag\nagent: expert_chunk\narm: condition_grid.yaml\nmethods:\n"
        "  oracle: {mode: teleop, readout: off}\n"
        "  oracle_task: {mode: teleop, readout: off, reversal_adapter: {task: cell}}\n"
        f"  oracle_task_rel: {{mode: teleop, readout: off, reversal_adapter: {{task: {rel}}}}}\n"
        "grid:\n  users: {U0: {corruption: null, adapter: null}}\n"
        f"  tasks: {{T2: {{obstacle: {DISC}, adapter: null}}, T0: {{obstacle: null, adapter: null}}}}\n"
        "  noises: [0.0]\ntrials: {tune: 2, test: 2}\nseed: 3\n"
    )
    return tmp_path / "diag.yaml"


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--context", "T2", "--trials", "0"], "--trials must be at least 1"),
        (["--context", "T0", "--trials", "3"], "no obstacle"),
        (["--context", "T2", "--trials", "3", "--p-free", "1.0"], "the train set is empty"),
        (["--context", "T2", "--trials", "3", "--n-demos", "0"], "the train set would be empty"),
        (["--context", "T9", "--trials", "3"], "needs exactly one cell"),
    ],
)
def test_the_diagnostic_refuses_what_it_cannot_diagnose(tmp_path, capsys, extra, message):
    sweep, out = _diag_sweep(tmp_path), tmp_path / "d"
    with pytest.raises(SystemExit) as stop:
        diag_main(["--sweep", str(sweep), "--method", "oracle_task_rel", "--out", str(out), *extra, "--cpu"])
    assert stop.value.code == 2 and message in capsys.readouterr().err and not out.exists()


def test_the_diagnostic_runs_end_to_end_on_a_tiny_sweep(tmp_path, monkeypatch):
    sweep = _diag_sweep(tmp_path, _image_adapter(tmp_path, monkeypatch, geometry="relative"))
    argv = [
        "--sweep",
        str(sweep),
        "--context",
        "T2",
        "--method",
        "oracle_task_rel",
        "--n-demos",
        "6",
        "--trials",
        "3",
    ]
    diag_main([*argv, "--out", str(tmp_path / "diag"), "--scene-cache", str(tmp_path / "scene"), "--cpu"])
    out = json.loads((tmp_path / "diag" / "T2_oracle_task_rel.json").read_text())
    assert set(out) == {"train", "fresh", "moved"}
    assert all(set(s) == {"oracle", "oracle_task", "oracle_task_rel"} for s in out.values())
    rates = [v for s in out.values() for v in s.values()]
    assert all(v["n"] == 3 and 0 <= v["hits"] <= 1 and 0 <= v["collision"] <= 1 for v in rates)
    assert {p.name.split("_cnn_")[0] for p in (tmp_path / "scene").glob("*.npy")} == {
        "T2_train",
        "T2_fresh",
        "T2_moved",
    }
