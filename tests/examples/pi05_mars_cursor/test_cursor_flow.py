"""The cursor task, its demonstrations and sampler, the policy, the readout, the controller and the configs. The
steering methods themselves are the library's (tests/policies/steering) and pinned against the cursor code in the
equivalence tests."""

import numpy as np
import pytest
import torch

from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.demos import DemoSampler, make_demos, replay
from lerobot.envs.cursor import CursorEnv, Layouts, TaskConfig, _layout, sample_layouts
from cursor_sa.policy import CursorFlowPolicy, FlowPolicy, load_policy
from cursor_sa.records import summarize
from cursor_sa.session import load_checkpoints
from cursor_sa.steering import (MAX_SPEED, as_adapter, SharedController, bci_chunk, hide_goal, observation, to_action,
                                to_deflection)

from lerobot.policies.steering import SteeringConfig
from lerobot.teleoperators import readout as shared_readout
from lerobot.teleoperators.readout import Readout, matrix_spec

CFG = TaskConfig()


def test_layouts_respect_reach_spacing_and_fail_loudly_when_impossible():
    for cfg, n, count in ((CFG, 200, (1, 16)), (TaskConfig(min_target_sep=0.2, min_reach=0.4, max_reach=0.5), 100, 4)):
        layouts = sample_layouts(np.random.default_rng(0), n, count, cfg)
        for b in range(len(layouts)):
            pts = layouts.targets[b, layouts.mask[b]]
            assert layouts.goal[b] < len(pts) and (np.linalg.norm(pts[:, None] - pts[None], axis=-1) + 9 * np.eye(len(pts))).min() >= cfg.min_target_sep
            d = np.linalg.norm(pts - layouts.start[b], axis=1)
            assert cfg.min_reach <= d[layouts.goal[b]] <= (cfg.max_reach or 2) and d.min() >= 2 * cfg.target_radius
    with pytest.raises(ValueError, match="cannot place 30 targets"):
        sample_layouts(np.random.default_rng(0), 1, 30, TaskConfig(min_target_sep=0.3))
    # a reach of 0.95 fits only goals near a corner: most target sets admit no start, and are redrawn
    far = TaskConfig(min_reach=0.95)
    lay = sample_layouts(np.random.default_rng(0), 20, 2, far)
    assert (np.linalg.norm(lay.goal_pos - lay.start, axis=1) >= 0.95).all()
    with pytest.raises(ValueError, match="cannot place a start"):
        _layout(np.random.default_rng(0), 2, 0, far, max_tries=20, rounds=1)
    assert _layout(np.random.default_rng(0), 2, 0, far, max_tries=20, rounds=500)[0].shape == (2,)


def test_layouts_carry_their_arrays_through_indexing_tile_and_concat():
    a = sample_layouts(np.random.default_rng(0), 3, (1, 2), CFG)
    b = Layouts(np.zeros((2, 2)), np.full((2, 5, 2), 0.5), np.ones((2, 5), bool), np.zeros(2, int),
                np.ones((2, 3, 3)), np.array([[True, True, False], [True, False, False]]))
    assert a.goal_mask().shape == a.mask.shape and (a.goal_mask().sum(1) == 1).all()
    np.testing.assert_array_equal(a.goal_mask().argmax(1), a.goal)
    both = Layouts.concat([a, b])
    assert len(both) == 5 and both.mask.shape == (5, 5) and both.mask[:3, 2:].sum() == 0
    np.testing.assert_array_equal(both.goal_pos[:3], a.goal_pos)
    assert both.obstacles.shape == (5, 3, 3) and both.obstacle_mask[:3].sum() == 0 and both.obstacle_mask[3:].tolist() == b.obstacle_mask.tolist()
    tiled = a.tile(2)
    assert len(tiled) == 6 and np.array_equal(tiled.targets[3:], a.targets) and np.array_equal(tiled[[1]].start, tiled[[4]].start)
    assert b[[1]].obstacles.shape == (1, 3, 3) and b.tile(3).obstacle_mask.shape == (6, 3)
    a.scene = {"cnn": np.arange(3 * 2 * 4 * 4, dtype=np.uint8).reshape(3, 2, 4, 4), "vlm_image": np.ones((3, 5, 6), np.float16)}
    perm = np.array([2, 0, 1])
    assert np.array_equal(a[[1]].scene["cnn"][0], a.scene["cnn"][1]) and np.array_equal(a[perm].scene["vlm_image"], a.scene["vlm_image"][perm])
    assert np.array_equal(a.tile(2).scene["cnn"][3:], a.scene["cnn"])
    assert Layouts.concat([a, a[[0]]]).scene["vlm_image"].shape == (4, 5, 6)
    assert b.scene is None and Layouts.concat([b, b]).scene is None and b.tile(2).scene is None
    with pytest.raises(ValueError, match="scene"):
        Layouts.concat([a, b])


def test_demos_select_goal_and_end_at_rest_on_it():
    rng = np.random.default_rng(1)
    layouts = sample_layouts(rng, 300, (1, 8), CFG)
    actions, lengths = make_demos(rng, layouts, CFG)
    env = replay(actions, layouts, CFG)
    assert env.success.all()
    np.testing.assert_array_equal(env.steps, lengths)
    end = layouts.start + actions.sum(1)
    np.testing.assert_allclose(end, layouts.goal_pos, atol=1e-9)
    speed = np.linalg.norm(actions, axis=-1)
    assert speed.max() < 0.1  # no teleporting: < 5 screens/s
    assert actions.shape[1] >= lengths.max()


def test_the_sampler_draws_within_every_demo_and_hides_the_goal_at_the_dropout():
    rng = np.random.default_rng(3)
    layouts = sample_layouts(rng, 100, (1, 8), CFG)
    actions, lengths = make_demos(rng, layouts, CFG)
    sampler = DemoSampler(layouts, actions, lengths, horizon=16, hist=4, device="cpu")
    flags = [sampler.sample(20000, p)[0]["goal"].sum(1) for p in (0.0, 0.25, 1.0)]
    assert (flags[0] == 1).all() and (flags[2] == 0).all() and set(flags[1].unique().tolist()) == {0, 1}
    assert abs((flags[1] == 0).float().mean().item() - 0.25) < 0.02
    sampler.lengths[:] = int(lengths.max())  # k up to the longest trial for every demo: the windows still fit
    obs, chunk = sampler.sample(20000)
    assert chunk.shape == (20000, 16, 2) and obs["hist"].shape == (20000, 4, 2)


def _obs(B=3, N=5, hist=4):
    g = torch.Generator().manual_seed(0)
    goal = torch.zeros(B, N, dtype=torch.bool)
    goal[torch.arange(B), torch.randint(N, (B,), generator=g)] = True
    return dict(pos=torch.rand(B, 2, generator=g), hist=0.01 * torch.randn(B, hist, 2, generator=g),
                targets=torch.rand(B, N, 2, generator=g), mask=torch.ones(B, N, dtype=torch.bool),
                goal=goal)


@pytest.mark.parametrize("mode", ["flow", "regress"])
def test_policy_ignores_target_order_and_padding(mode):
    torch.manual_seed(0)
    model = FlowPolicy(mode=mode).eval()
    obs = _obs()
    x, t = torch.randn(3, model.horizon, 2), torch.rand(3)
    ref = model(obs, x, t)
    perm = torch.randperm(5)
    shuffled = {**obs, **{k: obs[k][:, perm] for k in ("targets", "mask", "goal")}}
    torch.testing.assert_close(model(shuffled, x, t), ref, atol=1e-5, rtol=1e-5)
    padded = {**obs,
              "targets": torch.cat([obs["targets"], torch.rand(3, 4, 2)], 1),
              "mask": torch.cat([obs["mask"], torch.zeros(3, 4, dtype=torch.bool)], 1),
              "goal": torch.cat([obs["goal"], torch.zeros(3, 4, dtype=torch.bool)], 1)}
    torch.testing.assert_close(model(padded, x, t), ref, atol=1e-5, rtol=1e-5)
    assert model.sample(obs).shape == (3, model.horizon, 2)
    assert torch.isfinite(model.loss(obs, 0.01 * torch.randn(3, model.horizon, 2)))


@pytest.mark.parametrize("mode", ["flow", "regress"])
def test_obstacle_tokens_ignore_order_and_padding_and_are_read(mode, tmp_path):
    """A policy built with obstacle tokens (the grid's retrained policy) reads the discs in any order and
    padding; a policy without them ignores the obstacle keys and saves the config it always did."""
    torch.manual_seed(0)
    model = FlowPolicy(mode=mode, obstacle_tokens=True).eval()
    g = torch.Generator().manual_seed(1)
    obs = {**_obs(), "obstacle_mask": torch.ones(3, 3, dtype=torch.bool),
           "obstacles": torch.cat([torch.rand(3, 3, 2, generator=g), 0.03 + 0.05 * torch.rand(3, 3, 1, generator=g)], -1)}
    x, t = torch.randn(3, model.horizon, 2), torch.rand(3)
    ref = model(obs, x, t)
    perm = torch.tensor([2, 0, 1])
    shuffled = {**obs, "obstacles": obs["obstacles"][:, perm], "obstacle_mask": obs["obstacle_mask"][:, perm]}
    torch.testing.assert_close(model(shuffled, x, t), ref, atol=1e-5, rtol=1e-5)
    padded = {**obs, "obstacles": torch.cat([obs["obstacles"], torch.rand(3, 2, 3)], 1),
              "obstacle_mask": torch.cat([obs["obstacle_mask"], torch.zeros(3, 2, dtype=torch.bool)], 1)}
    torch.testing.assert_close(model(padded, x, t), ref, atol=1e-5, rtol=1e-5)
    bare = {k: v for k, v in obs.items() if not k.startswith("obstacle")}
    assert not torch.allclose(model(bare, x, t), ref, atol=1e-4)  # the discs are read
    none = {**obs, "obstacles": torch.zeros(3, 0, 3), "obstacle_mask": torch.zeros(3, 0, dtype=torch.bool)}
    torch.testing.assert_close(model(none, x, t), model(bare, x, t))  # no disc: as without the keys
    CursorFlowPolicy.wrapping(model).save_pretrained(tmp_path)
    torch.testing.assert_close(load_policy(tmp_path).model(obs, x, t), ref)
    plain = FlowPolicy(mode=mode).eval()
    assert "obstacle_tokens" not in plain.config and model.config["obstacle_tokens"] is True
    torch.testing.assert_close(plain(obs, x, t), plain(bare, x, t))  # without tokens the keys are ignored


def test_teleop_moves_by_the_command_and_a_noiseless_user_always_hits():
    rng = np.random.default_rng(4)
    layouts = sample_layouts(rng, 100, (1, 8), CFG)
    env, ctl, user = CursorEnv(layouts, CFG), SharedController(FlowPolicy(), layouts, "teleop"), ExpertAgent()
    np.testing.assert_allclose(ctl.act(np.array([[1.0, 0.0]])), [[MAX_SPEED * CFG.dt, 0.0]])
    user.reset(len(layouts))
    while not env.done.all():
        raw, served, reference = user.step(env.pos, layouts, ctl)
        np.testing.assert_array_equal(raw, served)  # no readout: served as meant
        assert reference is None
        prev = env.pos.copy()
        env.step(ctl.act(served))
        ctl.record(env.pos - prev)
    assert env.success.all()


def test_observation_units_and_adapter_helpers():
    policy = FlowPolicy(action_scale=0.004).eval()
    layouts = sample_layouts(np.random.default_rng(8), 3, (2, 4), CFG)
    hidden = observation(layouts, layouts.start, np.zeros((3, policy.hist, 2)))
    given = observation(layouts, layouts.start, np.zeros((3, policy.hist, 2)), goal_given=True)
    assert not hidden["goal"].any() and (given["goal"].sum(1) == 1).all()
    assert hidden["pos"].dtype == torch.float32 and hidden["mask"].shape == (3, layouts.mask.shape[1])
    discs = sample_layouts(np.random.default_rng(0), 2, 4, TaskConfig(min_reach=0.35, max_reach=0.5, obstacle={
        "kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.0, "side": "random"}))
    with_discs = observation(discs, discs.start, np.zeros((2, 4, 2)))
    assert with_discs["obstacles"].shape == (2, 1, 3) and with_discs["obstacle_mask"].tolist() == [[True], [True]]
    torch.testing.assert_close(hide_goal(given)["goal"], hidden["goal"])
    d = torch.tensor([[0.5, -1.0]])
    torch.testing.assert_close(to_deflection(to_action(d, policy), policy), d)
    torch.testing.assert_close(to_action(d, policy), d * MAX_SPEED * CFG.dt / 0.004)
    F = torch.diag(torch.tensor([1.0, 0.5]))
    x, v = torch.randn(3, policy.horizon, 2), torch.randn(3, policy.horizon, 2)
    out = as_adapter(F).adapt(hidden, x, torch.zeros(3), v, backward=True)  # the fixed F, in the backward pass only
    torch.testing.assert_close(out, v @ F.T)
    torch.testing.assert_close(as_adapter(F.expand(3, 2, 2)).adapt(hidden, x, torch.zeros(3), v, backward=True), out)
    torch.testing.assert_close(as_adapter(F).adapt(hidden, x, torch.zeros(3), v, backward=False), v)


def test_the_torch_readout_twins_follow_the_shared_readout():
    """The cursor's torch versions of the readout (training) give the shared numpy rule's numbers for the same draws."""
    policy = FlowPolicy().eval()
    m = matrix_spec({"scale": [1.0, 0.25]}, 2)
    deflection = torch.tensor([[[0.8, 0.8], [0.02, 0.01], [-0.5, 1.0]]])  # the middle step is idle
    served = Readout(torch.as_tensor(m, dtype=torch.float32)).chunk(deflection, torch.zeros(1))
    torch.testing.assert_close(served, torch.as_tensor(Readout(m)(deflection.numpy()), dtype=torch.float32))
    bci = bci_chunk(to_action(deflection, policy), torch.as_tensor(m, dtype=torch.float32), torch.zeros(1), policy)
    torch.testing.assert_close(to_deflection(bci, policy), served, atol=1e-6, rtol=1e-6)
    pushed = torch.tensor([[[0.0, 0.5], [0.02, 0.01]]]).expand(4000, -1, -1)  # noise far from the clip at +-1
    noisy = Readout(None).chunk(pushed, torch.full((4000,), 0.3))
    assert abs(noisy[:, 0, 0].std().item() - 0.3) < 0.02 and torch.equal(noisy[:, 1], pushed[:, 1])
    g = np.random.default_rng(5)
    raw = g.uniform(-1, 1, (6, 2))
    raw[1] = 0.02  # idle
    core, sigma, noise = shared_readout.corrupt(raw, m), g.uniform(0, 2, 6), g.normal(size=(6, 2))
    out = Readout.serve(torch.tensor(core), torch.tensor(raw), torch.tensor(sigma), torch.tensor(noise))
    np.testing.assert_allclose(out.numpy(), shared_readout.add_noise(core, raw, sigma, noise), atol=1e-12)
    chunk = torch.tensor(g.uniform(-1, 1, (6, 4, 2)))
    np.testing.assert_allclose(Readout(torch.tensor(m)).chunk(chunk, torch.zeros(6)).numpy(), shared_readout.corrupt(chunk.numpy(), m),
                               atol=1e-12)


def test_study_statistics_from_trial_records():
    from cursor_sa.analysis import aligned_path, cumulative_hits, paired_lead, path_efficiency
    # paired lead: trials are paired by (run, layout_id), whatever their order in the list
    focus = [dict(arm="a", run="s1", layout_id=i, outcome="hit") for i in range(40)]
    other = [dict(arm="b", run="s1", layout_id=i, outcome="hit" if i % 2 else "timeout") for i in range(40)]
    lead = paired_lead(focus + other[::-1], "a", "b", np.random.default_rng(0))
    assert np.isclose(lead[0], 0.5) and lead[1] <= 0.5 <= lead[2] and lead[1] > 0.3
    assert paired_lead(focus + focus[:1], "a", "a", np.random.default_rng(0))[0] == 0  # a method against itself
    # two participants whose runs share names pair within each participant, never across
    p1 = [dict(r, participant="p1") for r in focus + other]
    p2 = [dict(r, participant="p2", outcome="hit" if r["arm"] == "b" else "timeout") for r in focus + other]
    assert np.isclose(paired_lead(p1 + p2, "a", "b", np.random.default_rng(0))[0], (0.5 - 1.0) / 2)
    # cumulative hits: the fraction of all trials that have hit by time t
    trials = [dict(outcome="hit", duration_s=1.0), dict(outcome="hit", duration_s=3.0), dict(outcome="timeout", duration_s=5.0),
              dict(outcome="wrong_target", duration_s=2.0)]
    np.testing.assert_allclose(cumulative_hits(trials, [0.5, 1.0, 2.5, 5.0]), [0, 0.25, 0.25, 0.5])
    # aligned path: the start at the origin, the goal at (1, 0), everything scaled by the reach
    c, s_ = np.cos(0.7), np.sin(0.7)
    rot = lambda x, y: [0.5 + 0.4 * (c * x - s_ * y), 0.5 + 0.4 * (s_ * x + c * y)]
    rec = dict(start=rot(0, 0), targets=[rot(1, 0), rot(0.5, 0.6)], goal=0, target_radius=0.05,
               path=[rot(0, 0), rot(0.5, 0.1), rot(1, 0), rot(1, 0)], outcome="hit")
    path, targets, radius = aligned_path(rec)
    np.testing.assert_allclose(path, [[0, 0], [0.5, 0.1], [1, 0], [1, 0]], atol=1e-9)
    np.testing.assert_allclose(targets, [[1, 0], [0.5, 0.6]], atol=1e-9)
    assert np.isclose(radius, 0.05 / 0.4)
    # path efficiency: straight distance over the path length up to the first entry into the goal
    rec["path"] = [rot(0, 0), rot(0.5, 0.5), rot(1, 0), rot(1.2, 0.3)]
    assert np.isclose(path_efficiency(rec), 1 / np.sqrt(2))
    assert path_efficiency(dict(rec, outcome="timeout")) is None
    s = summarize(trials)
    assert s == dict(n=4, hits=2, success=0.5, wrong=0.25, collision=0.0, timeout=0.25, time_s=2.0)
    assert summarize(trials[2:])["time_s"] is None


def test_gain_scales_the_cursor_by_step_or_by_time():
    """Mode "step" scales every displacement but not the policy's history; mode "time" follows the planned chunk at
    the gain's speed, the history holding the planned steps, the next chunk after `execute` planned steps; at gain 1
    the two modes coincide."""
    layouts = sample_layouts(np.random.default_rng(5), 4, 3, CFG)
    z = np.zeros((4, 2))
    teleop = SharedController(FlowPolicy(), layouts, "teleop", speed_gain=0.5)
    np.testing.assert_allclose(teleop.act(np.array([[1.0, 0.0]])), [[0.5 * MAX_SPEED * CFG.dt, 0.0]])
    torch.manual_seed(0)
    full = SharedController(FlowPolicy().eval(), layouts, "policy", execute=2)
    steps = []
    for _ in range(2):
        steps.append(full.act(z))
        full.record(steps[-1])
    pos2, hist2 = full.pos.copy(), full.hist.copy()
    third = full.act(z)
    torch.manual_seed(0)
    ctl = SharedController(FlowPolicy().eval(), layouts, "policy", speed_gain=0.5)
    half = ctl.act(z)
    np.testing.assert_allclose(half, 0.5 * steps[0], rtol=1e-6)
    ctl.record(half)
    np.testing.assert_allclose(ctl.hist[:, -1], steps[0], rtol=1e-6)
    torch.manual_seed(0)
    slow = SharedController(FlowPolicy().eval(), layouts, "policy", execute=2, speed_gain=0.5, gain_mode="time")
    assert slow.cycle == 4
    moves = []
    for k in range(4):
        moves.append(slow.act(z))
        slow.record(moves[-1])
        if k == 0:  # the rest of the chunk, played at half speed, ends where the full-speed cursor ends
            plan = slow.planned()
            assert plan.shape == (4, 3, 2)
            np.testing.assert_allclose(plan[:, -1], layouts.start + steps[0] + steps[1], rtol=1e-6)
    for k, step in ((0, steps[0]), (1, steps[0]), (2, steps[1]), (3, steps[1])):
        np.testing.assert_allclose(moves[k], 0.5 * step, rtol=1e-6)
    np.testing.assert_allclose(slow.pos, pos2, rtol=1e-6)
    np.testing.assert_allclose(slow.hist, hist2, rtol=1e-6)  # one history entry per planned step, the whole step
    assert slow.chunks == 1 and full.chunks == 2 and slow.planned().shape == (4, 0, 2)
    np.testing.assert_allclose(slow.act(z), 0.5 * third, rtol=1e-6)  # the same next chunk, from the same state
    assert slow.chunks == 2
    same = SharedController(FlowPolicy(), layouts, "policy", execute=2, speed_gain=1.0, gain_mode="time")
    assert same.cycle == 2 and [same.steps_of(t) for t in range(3)] == [[(0, 1.0)], [(1, 1.0)], []]
    odd = SharedController(FlowPolicy(), layouts, "policy", execute=2, speed_gain=0.3, gain_mode="time")
    assert odd.cycle == 7
    played = {}
    for t in range(7):
        for i, share in odd.steps_of(t):
            played[i] = played.get(i, 0) + share
    assert played.keys() == {0, 1} and all(abs(s - 1) < 1e-9 for s in played.values())
    np.testing.assert_allclose([s for _, s in odd.steps_of(3)], [0.1, 0.2])  # a control step across two planned steps


def test_the_readout_corrupts_and_adds_noise_only_while_commanding():
    np.testing.assert_allclose(matrix_spec({"scale": [1.0, 0.5]}, 2), [[1, 0], [0, 0.5]])
    np.testing.assert_allclose(matrix_spec({"rotation_deg": 90}, 2) @ [1, 0], [0, 1], atol=1e-12)
    np.testing.assert_allclose(matrix_spec([[0, 1], [1, 0]], 2), [[0, 1], [1, 0]])
    assert matrix_spec(None, 2) is None
    m = matrix_spec({"scale": [2.0, 0.5]}, 2)
    adapter_matrix = lambda spec, corruption: SteeringConfig.from_spec(spec, corruption, n_dims=2, device="cpu").fixed_matrix
    np.testing.assert_allclose(adapter_matrix({"translation": "corruption"}, m), m)
    np.testing.assert_allclose(adapter_matrix({"F": {"scale": [3, 3]}}, None), 3 * np.eye(2))
    for bad in (lambda: matrix_spec({"scale": [1, 1, 1]}, 2), lambda: adapter_matrix({"translation": "corruption"}, None)):
        with pytest.raises(ValueError):
            bad()
    readout = Readout(matrix_spec({"scale": [1.0, 0.5]}, 2), input_noise=0.3, rng=np.random.default_rng(0))
    idle = np.array([[0.01, -0.02]] * 1000)
    np.testing.assert_array_equal(readout(idle), idle)
    pushed = readout(np.array([[0.4, 0.8]] * 20000))
    np.testing.assert_allclose(pushed.mean(0), [0.4, 0.4], atol=0.02)
    assert abs(pushed[:, 0].std() - 0.3) < 0.02
    np.testing.assert_allclose(Readout(matrix_spec({"scale": [2.0, 2.0]}, 2))(np.array([0.8, -0.1])), [1.0, -0.2])
    # per-trial matrices and noise levels, on single commands and on chunks
    ms = np.stack([matrix_spec({"scale": [1.0, 0.5]}, 2), matrix_spec({"rotation_deg": 90}, 2)])
    raw = np.array([[0.4, 0.8], [0.6, -0.3]])
    batched = Readout(ms)(raw)
    for i in range(2):
        np.testing.assert_allclose(batched[i], Readout(ms[i])(raw[i]))
    out = Readout(ms, np.array([0.0, 0.0]))(np.tile([[0.4, 0.8]], (2, 16, 1)))
    np.testing.assert_allclose(out[0], np.tile([[0.4, 0.4]], (16, 1)))
    np.testing.assert_allclose(out[1], np.tile([[-0.8, 0.4]], (16, 1)), atol=1e-12)
    noisy = Readout(None, np.array([0.0, 0.2]), np.random.default_rng(0))(np.tile([0.0, 0.5], (2, 2000, 1)))
    assert np.ptp(noisy[0], axis=0).max() == 0 and abs(noisy[1, :, 0].std() - 0.2) < 0.01


def test_shipped_configs_load_and_flags_override_them():
    arms = {p.name: config.load_arm(p) for p in config.CONFIG_DIR.glob("*.yaml") if p.name != "study.yaml"}
    assert {"base.yaml", "teleop.yaml", "flow_control.yaml", "frs.yaml", "condition_study.yaml", "condition_grid.yaml",
            "study_teleop.yaml", "study_flow_control.yaml", "study_frs.yaml", "study_mars.yaml"} == set(arms)
    assert {arms[f"study_{m}.yaml"]["mode"] for m in ("teleop", "flow_control", "frs", "mars")} == {"teleop", "flow_control", "frs"}
    assert all(arms[f"study_{m}.yaml"]["corruption"] == {"scale": [1.0, 0.25]} for m in ("teleop", "flow_control", "frs", "mars"))
    assert arms["frs.yaml"]["corruption"] is None
    cfg = config.load_arm("frs.yaml", {"speed_gain": 0.3, "corruption": {"scale": [1.0, 0.5]},
                                       "reversal_adapter": {"translation": "corruption"}})  # MARS with F = M
    assert cfg["speed_gain"] == 0.3 and cfg["mode"] == "frs" and cfg["method"] == "mars"
    np.testing.assert_allclose(cfg["adapter_matrix"], cfg["corruption_matrix"])
    assert config.load_arm("study_flow_control.yaml", {"corruption": None})["corruption_matrix"] is None
    study_mars = config.load_arm("study_mars.yaml")
    assert (study_mars["learned_adapter"], study_mars["n_reverse_steps"], study_mars["pinned_steps"]) == ("user_gain025y_expert_n100_s0", 5, 8)
    assert config.load_arm("teleop.yaml")["method"] == "teleop" and config.load_arm("frs.yaml")["method"] == "frs"
    study_fc = config.load_arm("study_flow_control.yaml")  # the command in action units, at the tuned tau and lambda
    assert study_fc["mode"] == "flow_control" and (study_fc["tau"], study_fc["lambda"]) == (14, 1)
    policy = load_checkpoints({"a": study_fc}, "cpu")[0][study_fc["model"]]
    assert study_fc["ic_gain"] == pytest.approx(float(to_action(1.0, policy)), abs=1e-4)  # a full push's
    base_fc = config.load_arm("flow_control.yaml")
    assert (base_fc["tau"], base_fc["ic_gain"], base_fc["lambda"]) == (8, 2.0, 1)
    assert config.load_arm("flow_control.yaml", {"mode": "shared_flow_control"})["mode"] == "flow_control"
    assert config.load_arm("flow_control.yaml", {"tau": 16})["tau"] == 16
    for bad in ({"tau": 17}, {"lambda": 11}, {"ic_gain": 0.0}):
        with pytest.raises(ValueError, match="control"):
            config.load_arm("flow_control.yaml", bad)
    from cursor_sa.records import arm_fields
    assert config.load_arm("base.yaml")["gain_mode"] == "step" and arm_fields(config.load_arm("condition_grid.yaml"))["gain_mode"] == "time"
    with pytest.raises(ValueError, match="gain_mode"):
        config.load_arm("teleop.yaml", {"gain_mode": "slow"})


def test_arm_task_parameters_reach_the_env_and_the_records():
    from cursor_sa.records import arm_fields
    from cursor_sa.session import arm_task
    base = config.load_arm("base.yaml")
    assert (base["target_radius"], base["min_target_sep"], base["reach"]) == (0.05, 0.12, [0.2, None])
    assert (base["dwell_s"], base["time_limit_s"]) == (0.3, 5.0)
    assert arm_task(base) == TaskConfig()  # the defaults are the task as it was
    arm = config.load_arm("base.yaml", {"dwell_s": 0.5, "time_limit_s": 8, "reach": [0.3, 0.6], "min_target_sep": 0.2,
                                        "target_radius": 0.04})
    task = arm_task(arm)
    assert (task.dwell_steps, task.max_steps, task.min_reach, task.max_reach) == (25, 400, 0.3, 0.6)
    assert (task.min_target_sep, task.target_radius) == (0.2, 0.04)
    fields = arm_fields(arm)
    assert fields["reach"] == [0.3, 0.6] and fields["dwell_s"] == 0.5 and fields["time_limit_s"] == 8
    assert fields["min_target_sep"] == 0.2 and fields["target_radius"] == 0.04
    with pytest.raises(ValueError, match="task.reach must be"):
        config.load_arm("base.yaml", {"reach": 0.3})
    with pytest.raises(ValueError, match="task.time_limit_s"):
        config.load_arm("base.yaml", {"time_limit_s": 0})
    from run_experiment import parse_args
    args = parse_args(["--arm", "teleop.yaml", "--reach", "[0.3, 0.6]", "--dwell-s", "0.5", "--min-target-sep", "0.2"])
    assert args.overrides == {"reach": [0.3, 0.6], "dwell_s": 0.5, "min_target_sep": 0.2}  # the free-play flags
    assert config.load_arm("teleop.yaml", args.overrides)["dwell_s"] == 0.5


def test_config_files_fill_defaults_and_reject_unknown_keys(tmp_path):
    protocol = config.load_protocol("study.yaml")
    assert list(protocol["arms"]) == ["teleop", "flow_control", "frs", "mars"]
    assert protocol["blocks"]["trials"] == 40 and protocol["display"]["blind"] is True
    (tmp_path / "p.yaml").write_text("arms: {a: teleop.yaml}\nblocks: {trial: 3}\n")
    with pytest.raises(ValueError, match="unknown keys in blocks"):
        config.load_protocol(tmp_path / "p.yaml")
    (tmp_path / "q.yaml").write_text("arms: {a: teleop.yaml}\n")
    assert config.load_protocol(tmp_path / "q.yaml")["blocks"]["practice"] == 5
    (tmp_path / "bad.yaml").write_text("extends: %s\ncontrol:\n  tua: 3\n" % (config.CONFIG_DIR / "base.yaml"))
    with pytest.raises(ValueError, match="unknown key control.tua"):
        config.load_arm(tmp_path / "bad.yaml")


def test_cursor_inertia_and_curl_field():
    layouts = sample_layouts(np.random.default_rng(6), 1, 1, CFG)
    command = np.array([[0.01, 0.0]])  # 0.5 screens/s to the right
    for curl, angle in [(0.0, 0.0), (10.0, -45.0)]:
        cfg = TaskConfig(velocity_tau=0.1, curl=curl, max_steps=10_000)
        env = CursorEnv(layouts, cfg)
        for _ in range(100):  # 2 s, many time constants
            env.step(command)
        v = env.vel[0]
        np.testing.assert_allclose(np.degrees(np.arctan2(v[1], v[0])), angle, atol=1e-6)  # clockwise turn
        np.testing.assert_allclose(np.linalg.norm(v), 0.5 / np.sqrt(1 + (0.1 * curl) ** 2), rtol=1e-6)
    with pytest.raises(ValueError):
        CursorEnv(layouts, TaskConfig(curl=5.0))
