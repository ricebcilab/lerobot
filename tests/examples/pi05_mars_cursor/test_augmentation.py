"""Rigid copies of the demonstrations (`augment_demos`) and the sampler's jitter."""

import numpy as np
import pytest
import torch
from cursor_sa import demos
from cursor_sa.demos import DemoSampler, augment_demos, make_demos, replay
from cursor_sa.env import Layouts, TaskConfig, layouts_on_screen, move_layouts, obstacle_spec, sample_layouts

STUDY = {"min_reach": 0.35, "max_reach": 0.5}
DISC = {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.05}
TASK = TaskConfig(**STUDY, obstacle=obstacle_spec(DISC))
FREE = TaskConfig(**STUDY)


def _demos(task=TASK, n=6, seed=0):
    rng = np.random.default_rng(seed)
    lay = sample_layouts(rng, n, 4, task)
    return (lay, *make_demos(rng, lay, task))


def _path(layouts, actions):
    """Every cursor position of the action sequences, [B, L + 1, 2]."""
    steps = np.concatenate([np.zeros_like(actions[:, :1]), np.cumsum(actions, 1)], 1)
    return layouts.start[:, None] + steps


def reach(a, b):
    return np.linalg.norm(a - b, axis=-1)


def _straight(start, goal, steps, targets=(), hold=30):
    """One demonstration at constant speed from `start` to `goal`, then a hold; its layout has `goal` first."""
    start, goal = np.array(start, float), np.array(goal, float)
    lay = Layouts(
        start[None], np.array([goal, *targets])[None], np.ones((1, 1 + len(targets)), bool), np.array([0])
    )
    actions = np.concatenate([np.tile((goal - start) / steps, (1, steps, 1)), np.zeros((1, hold, 2))], 1)
    return lay, actions, replay(actions, lay, FREE).steps


def test_copies_are_rigid_images_of_their_originals_and_the_originals_come_first():
    lay, actions, lengths = _demos()
    out, acts, lens, kept, _ = augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, TASK)
    assert kept.dtype.kind == "i" and kept.sum() > 0
    assert len(out) == 6 + kept.sum() and (kept <= 2).all() and len(acts) == len(lens) == len(out)
    for name in ("start", "targets", "mask", "goal", "obstacles", "obstacle_mask"):
        assert np.array_equal(getattr(out, name)[:6], getattr(lay, name))
    assert np.array_equal(acts[:6], actions) and np.array_equal(lens[:6], lengths)

    env = replay(acts[6:], out[6:], TASK)
    assert env.success.all() and np.array_equal(lens[6:], env.steps)
    assert layouts_on_screen(out[6:]).all()
    assert ((_path(out[6:], acts[6:]) >= 0) & (_path(out[6:], acts[6:]) <= 1)).all()

    origin = np.repeat(np.arange(6), kept)  # the copies come grouped by original, in original order
    copy, base = out[6:], lay[origin]
    assert np.allclose(reach(copy.start, copy.goal_pos), reach(base.start, base.goal_pos))
    assert np.allclose(
        reach(copy.start, copy.obstacles[:, 0, :2]), reach(base.start, base.obstacles[:, 0, :2])
    )
    assert np.allclose(
        reach(copy.goal_pos, copy.obstacles[:, 0, :2]), reach(base.goal_pos, base.obstacles[:, 0, :2])
    )
    assert np.array_equal(copy.obstacles[..., 2], base.obstacles[..., 2])
    assert np.allclose(np.linalg.norm(acts[6:], axis=-1), np.linalg.norm(actions[origin], axis=-1))


def test_motions_are_drawn_angle_then_shift_then_mirror_for_the_originals_short_of_k():
    lay, actions, lengths = _demos(FREE, n=8)
    out, _, _, kept, _ = augment_demos(
        np.random.default_rng(3), lay, actions, lengths, 1, FREE, shift=0.1, tries=1
    )
    draw = np.random.default_rng(3)
    angle = draw.uniform(0, 2 * np.pi, 8)
    shift = draw.uniform(-0.1, 0.1, (8, 2))
    mirror = draw.random(8) < 0.5
    want = move_layouts(lay, angle, shift, mirror)[kept == 1]
    assert kept.sum() > 0
    assert np.array_equal(out.start[8:], want.start) and np.array_equal(out.targets[8:], want.targets)


def test_a_seed_gives_the_same_copies_whatever_the_cursor_noise_of_the_task():
    lay, actions, lengths = _demos(FREE)
    noisy = TaskConfig(**STUDY, cursor_noise=0.05)
    a = augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, FREE)
    b = augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, noisy)
    assert (
        np.array_equal(a[0].start, b[0].start) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    )


def test_replays_run_500_rows_at_a_time(monkeypatch):
    lay, actions, lengths = _demos(FREE)
    lay, actions, lengths = lay.tile(350), np.tile(actions, (350, 1, 1)), np.tile(lengths, 350)
    rows = []

    def counted(moved, layouts, task):
        rows.append(len(moved))
        return replay(moved, layouts, task)

    monkeypatch.setattr(demos, "replay", counted)
    out = augment_demos(np.random.default_rng(1), lay, actions, lengths, 1, FREE, tries=1)[0]
    assert len(rows) >= 2 and max(rows) == 500 and len(out) > 2100


def test_a_failed_demonstration_gets_no_copies():
    lay, actions, lengths = _demos()
    lengths = lengths.copy()
    lengths[2] = 0
    out, _, lens, kept, _ = augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, TASK)
    assert kept[2] == 0 and kept.sum() > 0 and lens[2] == 0 and len(out) == 6 + kept.sum()
    origin = np.repeat(np.arange(6), kept)
    assert 2 not in origin and (lens[6:] > 0).all()


@pytest.mark.parametrize("k, failed", [(0, False), (2, True)])
def test_nothing_to_copy_returns_the_originals(k, failed):
    lay, actions, lengths = _demos(FREE)
    lengths = np.zeros_like(lengths) if failed else lengths
    push = np.ones_like(actions)
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    out, acts, lens, kept, moved = augment_demos(rng, lay, actions, lengths, k, FREE, push=push)
    assert (
        np.array_equal(out.start, lay.start)
        and np.array_equal(acts, actions)
        and np.array_equal(lens, lengths)
    )
    assert np.array_equal(moved, push) and kept.tolist() == [0] * 6
    assert rng.bit_generator.state == state


def test_copies_run_out_on_a_screen_spanning_reach():
    distractors = ((0.3, 0.2), (0.7, 0.2), (0.5, 0.8))
    lay, actions, lengths = _straight((0.02, 0.5), (0.98, 0.5), 60, distractors)
    assert lengths[0] < FREE.max_steps and layouts_on_screen(lay).all()
    out, acts, lens, kept, _ = augment_demos(
        np.random.default_rng(0), lay, actions, lengths, 8, FREE, tries=1
    )
    assert kept[0] < 8 and len(out) == 1 + kept[0]
    assert np.array_equal(out.start[:1], lay.start) and np.array_equal(acts[:1], actions)
    assert np.array_equal(lens[:1], lengths)


def test_a_copy_whose_path_leaves_the_screen_is_refused_though_its_layout_fits():
    # out to x = 0.96 and back to a goal beside the start: the layout has room to move, the path barely has
    start, far, goal = np.array([0.45, 0.5]), np.array([0.96, 0.5]), np.array([0.55, 0.5])
    lay = Layouts(start[None], goal[None, None], np.ones((1, 1), bool), np.array([0]))
    out_leg, back_leg = (
        np.linspace(0, 1, 26)[None, :, None] * (far - start),
        np.linspace(0, 1, 26)[None, :, None] * (goal - far),
    )
    actions = np.concatenate([np.diff(out_leg, axis=1), np.diff(back_leg, axis=1), np.zeros((1, 30, 2))], 1)
    lengths = replay(actions, lay, FREE).steps
    assert lengths[0] < FREE.max_steps
    out, acts, _, kept, _ = augment_demos(np.random.default_rng(0), lay, actions, lengths, 200, FREE, tries=1)
    assert 0 < kept[0] < 200
    path = _path(out[1:], acts[1:])
    assert ((path >= 0) & (path <= 1)).all()


def test_a_copy_that_does_not_replay_to_success_is_refused():
    lay, actions, lengths = _demos(FREE)
    still = np.zeros_like(
        actions
    )  # the layouts fit the screen and so does the path, but the cursor never moves
    out, _, _, kept, _ = augment_demos(np.random.default_rng(1), lay, still, lengths, 2, FREE, tries=1)
    assert kept.sum() == 0 and len(out) == 6


def test_free_layouts_are_copied_too():
    lay, actions, lengths = _demos(FREE)
    assert lay.obstacles.shape[1] == 0
    out, acts, _, kept, _ = augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, FREE)
    assert kept.sum() > 0 and replay(acts[6:], out[6:], FREE).success.all()


def test_pushes_move_with_their_copies():
    lay, actions, lengths = _demos()
    push = np.random.default_rng(2).normal(size=actions.shape)
    out, acts, lens, kept, moved = augment_demos(
        np.random.default_rng(1), lay, actions, lengths, 2, TASK, push=push
    )
    assert np.array_equal(moved[:6], push) and moved.shape == acts.shape
    # a rigid motion preserves dot products: per step, moved push . moved action == push . action of the original
    origin = np.repeat(np.arange(6), kept)
    assert np.allclose((moved[6:] * acts[6:]).sum(-1), (push[origin] * actions[origin]).sum(-1), atol=1e-9)
    assert augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, TASK)[4] is None
    with pytest.raises(ValueError, match="push"):
        augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, TASK, push=push[:3])


def test_jitter_zero_draws_nothing():
    lay, actions, lengths = _demos(FREE)
    sampler = DemoSampler(lay, actions, lengths, 8, 2, "cpu")
    torch.manual_seed(0)
    a = sampler.sample(8)
    after = torch.rand(1)
    torch.manual_seed(0)
    b = DemoSampler(lay, actions, lengths, 8, 2, "cpu", jitter=0.0).sample(8)
    assert (
        torch.equal(a[0]["pos"], b[0]["pos"])
        and torch.equal(a[1], b[1])
        and torch.equal(after, torch.rand(1))
    )


def test_jitter_displaces_the_position_and_the_chunk_returns_to_the_path():
    lay, actions, lengths = _demos(FREE)
    s = DemoSampler(lay, actions, lengths, horizon=8, hist=2, device="cpu", jitter=0.02, jitter_steps=4)
    assert s.jitter == 0.02 and s.jitter_steps == 4
    torch.manual_seed(0)
    obs, chunk, i, k = s.sample(4096, indexed=True)
    ref = DemoSampler(lay, actions, lengths, 8, 2, "cpu")
    delta = obs["pos"] - ref.pos[i, k]
    assert abs(delta.std() - 0.02) < 0.002
    base = ref.actions[i[:, None], k[:, None] + ref.window][:, 2:]
    assert torch.allclose(chunk[:, :4].sum(1), base[:, :4].sum(1) - delta, atol=1e-6)
    assert torch.allclose(chunk[:, :4], base[:, :4] - delta[:, None] / 4, atol=1e-6)  # spread evenly
    assert torch.equal(chunk[:, 4:], base[:, 4:])
    assert torch.equal(obs["hist"], ref.actions[i[:, None], k[:, None] + ref.window][:, :2])  # hist untouched


def test_jitter_with_a_short_chunk_corrects_over_what_there_is():
    lay, actions, lengths = _demos(FREE)
    s = DemoSampler(lay, actions, lengths, horizon=2, hist=2, device="cpu", jitter=0.02, jitter_steps=4)
    torch.manual_seed(0)
    obs, chunk, i, k = s.sample(4096, indexed=True)
    ref = DemoSampler(lay, actions, lengths, 2, 2, "cpu")
    delta = obs["pos"] - ref.pos[i, k]
    base = ref.actions[i[:, None], k[:, None] + ref.window][:, 2:]
    assert torch.allclose(chunk.sum(1), base.sum(1) - delta, atol=1e-6)
    # at a hold step the label is all zeros and the correction alone brings the cursor back onto the goal
    goal = torch.as_tensor(lay.goal_pos, dtype=torch.float32)[i]
    held = (base == 0).all(2).all(1) & ((ref.pos[i, k] - goal).norm(dim=1) < 1e-6)
    assert held.sum() > 0
    assert torch.allclose(obs["pos"][held] + chunk[held].sum(1), goal[held], atol=1e-5)


@pytest.mark.parametrize("kwargs", [{"jitter": -0.01}, {"jitter_steps": 0}])
def test_jitter_arguments_are_checked(kwargs):
    lay, actions, lengths = _demos(FREE, n=2)
    with pytest.raises(ValueError, match="jitter"):
        DemoSampler(lay, actions, lengths, 8, 2, "cpu", **kwargs)
