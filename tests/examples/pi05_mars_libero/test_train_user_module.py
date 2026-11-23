import numpy as np
import pytest
import torch
import train_user_module as tum
from teleop import Curl

from lerobot.policies.steering import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    Flow,
    LearnedUserModule,
)

CHUNK, MAX_DIM = 4, 8


def test_operator_command_follows_the_policy_operator_rule():
    chunk = np.zeros((6, 7))
    chunk[:2, :3] = [[3.0, 0.0, -0.5], [1.0, 0.0, -0.5]]
    chunk[:2, 6] = [0.2, 0.1]
    translation, gripper = tum.operator_command(chunk, n_action_steps=2)
    np.testing.assert_allclose(translation, [1.0, 0.0, -0.5])  # mean (2, 0, -0.5) clipped
    assert gripper == GRIPPER_CLOSE
    assert tum.operator_command(-chunk, n_action_steps=2)[1] == GRIPPER_OPEN


def test_readout_cores_are_what_the_rollouts_reader_serves_along_an_episode():
    from config import ControlSettings
    from teleop import ReadoutReader, TranslationGain

    class Push:
        gripper = GRIPPER_OPEN

    pushes = np.array(
        [[0.6, -0.2, 0.4], [0.6, -0.2, 0.4], [0.1, 0.9, 0.0], [0.6, -0.2, 0.4], [0.9, 0.9, 0.9]]
    )
    episodes, steps = [0, 0, 0, 1, 1], [0, 10, 20, 0, 10]
    lag = ControlSettings(corruption_curl=Curl(2.0, 0.5), resolved_translation_gain=1.0)
    cores = tum.readout_cores(pushes, episodes, steps, lag)
    # a rollout of episode 0: the field stepped at every control step, read at each chunk boundary with the new push
    source = Push()
    reader = ReadoutReader(TranslationGain(source, 1.0), curl=Curl(2.0, 0.5))
    served = []
    for step in range(21):
        if step % 10 == 0:
            source.translation = pushes[step // 10]
        reader.advance(step)
        if step % 10 == 0:
            served.append(reader.translation)
    np.testing.assert_array_equal(cores[:3], served)
    assert not np.allclose(
        cores[0], cores[1]
    )  # the lag carries across chunks: the same push is served differently
    np.testing.assert_array_equal(cores[3], cores[0])  # a new episode starts from rest
    assert np.linalg.norm(
        tum.readout_cores(pushes[4:], [0], [0], ControlSettings(resolved_translation_gain=1.0))[0]
    ) == pytest.approx(1.0)
    keypad = ControlSettings(corruption_quantize=True, resolved_translation_gain=1.0)
    np.testing.assert_allclose(tum.readout_cores(pushes[:1], [0], [0], keypad), [[np.sqrt(0.56), 0.0, 0.0]])
    stick = ControlSettings(corruption_matrix=np.diag([1.0, 1.0, 0.0]), resolved_translation_gain=1.0)
    np.testing.assert_allclose(tum.readout_cores(pushes[:1], [0], [0], stick), [[0.6, -0.2, 0.0]])


def test_serve_noises_every_core_an_idle_one_included():
    core = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])  # the second: the operator rests
    served = tum.serve(core, [0.1, 0.1], np.random.default_rng(0))
    draws = np.random.default_rng(0).normal(size=core.shape)
    np.testing.assert_allclose(served, np.clip(core + 0.1 * draws, -1.0, 1.0))  # the rollouts' rule
    np.testing.assert_allclose(tum.serve(core, 0.0, np.random.default_rng(0)), core)


def test_the_session_is_built_without_the_arms_adapter():
    from config import ControlSettings

    control = ControlSettings(
        mode="shared_flow_reversal_steering",
        corruption={"rotation_z_deg": 20},
        corruption_matrix=np.eye(3),
        reversal_adapter={"learned": "outputs/user_modules/rotz20/user.pt"},
        reversal_adapter_matrix=np.eye(7),
        reversal_adapter_user_path="outputs/user_modules/rotz20/user.pt",
        reversal_adapter_task_path="task.pt",
    )
    calibration = tum.calibration_control(control)
    assert calibration.reversal_adapter is None and calibration.reversal_adapter_matrix is None
    assert calibration.reversal_adapter_user_path is None and calibration.reversal_adapter_task_path is None
    assert (
        calibration.corruption == {"rotation_z_deg": 20}
        and calibration.corruption_matrix is control.corruption_matrix
    )
    assert control.reversal_adapter_user_path is not None  # the arm's own settings are left alone
    args = tum.parse_args(
        [
            "--config",
            "c.yaml",
            "--out",
            "o",
            "--set",
            "scene.suite=libero_spatial",
            "--set",
            "scene.task_ids=[0, 1]",
        ]
    )
    assert args.sets == ["scene.suite=libero_spatial", "scene.task_ids=[0, 1]"]


def test_field_loss_is_zero_for_the_exact_path_velocity_and_scales_otherwise():
    user = LearnedUserModule(horizon=CHUNK, n_dims=7, d=16, tdim=8)  # the identity at init
    a_bci, z_l = torch.randn(2, CHUNK, MAX_DIM), torch.randn(2, CHUNK, MAX_DIM)
    t = torch.rand(2)
    exact = lambda x_t, tt: z_l - a_bci  # noqa: E731
    assert LearnedUserModule.field_loss(user, exact, a_bci, z_l, t).item() == pytest.approx(0.0, abs=1e-6)
    doubled = lambda x_t, tt: 2 * (z_l - a_bci)  # noqa: E731
    expected = ((z_l - a_bci)[..., :7] ** 2).mean((1, 2))
    assert LearnedUserModule.field_loss(user, doubled, a_bci, z_l, t).item() == pytest.approx(
        expected.mean().item(), rel=1e-5
    )
    weighted = LearnedUserModule.field_loss(user, doubled, a_bci, z_l, t, weight=torch.tensor([1.0, 0.0]))
    assert weighted.item() == pytest.approx(expected[0].item(), rel=1e-5)
    weighted.backward()  # gradients reach the module
    assert user.net[-1].weight.grad is not None


def test_regenerate_round_trips_a_constant_field_and_heading_cosine():
    field = lambda x_t, t: torch.ones_like(x_t)  # noqa: E731
    a = torch.randn(3, CHUNK, MAX_DIM)
    torch.testing.assert_close(Flow(field).regenerate(a, 10), a, atol=1e-5, rtol=0)
    chunk = torch.zeros(1, CHUNK, MAX_DIM)
    chunk[0, :, 0] = 1.0
    other = torch.zeros(1, CHUNK, MAX_DIM)
    other[0, :, 1] = 1.0
    assert Flow.heading_cosine(chunk, chunk).item() == pytest.approx(1.0)
    assert Flow.heading_cosine(chunk, other).item() == pytest.approx(0.0, abs=1e-6)


def test_stack_batches_and_batched_reference():
    b1 = {"observation.state": torch.zeros(1, 2), "task": ["a"], "action": None, "next.reward": 0.0}
    b2 = {"observation.state": torch.ones(1, 2), "task": ["b"], "action": None, "next.reward": 0.0}
    stacked = tum.stack_batches([b1, b2])
    assert stacked["observation.state"].shape == (2, 2) and stacked["task"] == ["a", "b"]
    # the preprocessor's per-transition defaults are carried over, not stacked
    assert stacked["action"] is None and stacked["next.reward"] == 0.0
    ref = Flow.uniform_reference(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]),
        np.array([-1.0, 1.0]),
        np.zeros(7),
        np.ones(7),
        CHUNK,
        MAX_DIM,
    )
    assert ref.shape == (2, CHUNK, MAX_DIM)
    assert ref[0, 0, 0] == 1.0 and ref[1, 3, 1] == 0.5 and ref[1, 0, 6] == 1.0 and ref[0, 0, 6] == -1.0


def test_state_cache_round_trips_with_a_safe_load(tmp_path):
    state = {
        "observation": {"img": np.zeros((2, 2, 3), np.uint8), "state": np.ones(3)},
        "task": "open",
        "task_id": 1,
        "label": torch.zeros(CHUNK, 7),
        "latent": torch.ones(CHUNK, MAX_DIM),
        "noise": np.array([0.3, -0.1, 0.0]),  # its frozen draw
        "pushed": np.zeros(3),
        "resting": True,
        "raw": np.array([0.2, 0.0, 0.0]),
        "core": np.array([0.1, 0.0, 0.0]),
        "gripper_normalized": -1.0,
    }
    tum.save_states([state], tmp_path / "states.pt")
    back = tum.load_states(tmp_path / "states.pt")[0]
    assert back["task"] == "open" and back["task_id"] == 1 and back["gripper_normalized"] == -1.0
    np.testing.assert_array_equal(back["observation"]["img"], state["observation"]["img"])
    assert isinstance(back["raw"], np.ndarray) and isinstance(back["observation"]["state"], np.ndarray)
    assert torch.is_tensor(back["label"]) and torch.is_tensor(back["latent"])
    np.testing.assert_array_equal(back["noise"], state["noise"])
    assert isinstance(back["noise"], np.ndarray) and back["resting"] is True


def test_cross_step_dims_take_action_dim_names_or_indices():
    assert tum.parse_cross_step_dims(None) is None
    assert tum.parse_cross_step_dims("dx,dy,dz") == [0, 1, 2]
    assert tum.parse_cross_step_dims("[dx, dy, dz]") == [0, 1, 2]
    assert tum.parse_cross_step_dims(["dyaw", 6]) == [5, 6]
    assert tum.parse_cross_step_dims("0,2") == [0, 2]
    # caught at parse time, before collect_states
    for bad in ("dx,yaw", "9", "-1", "dx,dx", "0,0", "", [True]):
        with pytest.raises(ValueError, match="cross_step_dims"):
            tum.parse_cross_step_dims(bad)
    args = tum.parse_args(
        ["--config", "c.yaml", "--out", "o", "--cross-step-rank", "2", "--cross-step-dims", "dx,dy,dz"]
    )
    assert (args.step_rank, args.cross_step_rank, args.cross_step_dims) == (None, 2, [0, 1, 2])
    defaults = tum.parse_args(["--config", "c.yaml", "--out", "o"])
    assert (defaults.step_rank, defaults.cross_step_rank, defaults.cross_step_dims) == (None, 0, None)
    with pytest.raises(SystemExit):
        tum.parse_args(
            ["--config", "c.yaml", "--out", "o", "--cross-step-dims", "dx"]
        )  # no coupling to restrict


def test_states_are_collected_at_the_chunk_boundaries_from_the_first():
    from types import SimpleNamespace

    resets, prompts = [], []

    class FakeEnv:
        def reset(self, seed=None, options=None):
            resets.append(options["init_state_id"])
            return {"reset": len(resets)}, {}

    class FakeSession:
        suite, task_id, task_description, vec_env = "s", 0, "t", FakeEnv()

        def __init__(self):
            self.modes = []
            self.operator = SimpleNamespace(translation=np.zeros(3), resting=False)

        def set_mode(self, mode):
            self.modes.append(mode)

        def set_scene(self, suite, task_id):
            self.task_id = task_id

        def rollout(self, task, recorder=None, initial_observation=None, max_steps=None):
            assert initial_observation == {"reset": len(resets)}  # the rollout starts from the state just set
            prompts.append((task, max_steps))
            for k in range(
                max_steps or 25
            ):  # as Session.rollout reports: the step count after the step, the observation before it
                if k % 10 == 0:  # a chunk boundary: the operator's update and the readout's read
                    self.operator_chunk = np.full((50, 7), k)
                    self.operator.resting = k == 10
                    self.operator.translation = np.zeros(3) if k == 10 else np.array([0.3, 0.0, 0.0])
                recorder(step=k + 1, observation={"k": k})

    settings = SimpleNamespace(
        task_ids=[0],
        prompt="do something",
        session=SimpleNamespace(
            suite="s", n_action_steps=10, control=SimpleNamespace(mode="shared_flow_reversal_steering")
        ),
    )
    states = tum.collect_states(FakeSession(), settings, 2, np.random.default_rng(0))
    assert [(s["episode"], s["step"], s["observation"]["k"]) for s in states] == [
        (0, 0, 0), (0, 10, 10), (0, 20, 20), (1, 0, 0), (1, 10, 10), (1, 20, 20)
    ]  # fmt: skip

    # two tasks, four episodes: each pass over the tasks starts from a task's next initial state
    resets.clear()
    settings.task_ids = [3, 7]
    states = tum.collect_states(FakeSession(), settings, 4, np.random.default_rng(0))
    assert resets == [0, 0, 1, 1]
    assert [s["task_id"] for s in states[::3]] == [3, 7, 3, 7]

    # rollouts under the arm's own prompt follow, the robot unsteered and the operator watching it: each state keeps
    # the chunk the operator planned, the push it would deliver (zero where it rests) and whether it rests
    resets.clear()
    prompts.clear()
    session = FakeSession()
    states = tum.collect_states(
        session, settings, 1, np.random.default_rng(0), open_episodes=2, open_steps=15
    )
    assert prompts == [("t", None), ("do something", 15), ("do something", 15)]
    assert session.modes == ["policy"]  # nothing steers: the operator's command does not act
    assert resets == [0, 0, 0]  # each pass over the tasks from a task's next initial state
    watched = [s for s in states if s["rollout"] == "open"]
    assert [(s["episode"], s["step"], s["task"], s["task_id"]) for s in watched] == [
        (1, 0, "t", 3), (1, 10, "t", 3), (2, 0, "t", 7), (2, 10, "t", 7)
    ]  # fmt: skip
    for s in watched:  # what the fake operator had at that chunk boundary
        k = s["observation"]["k"]
        np.testing.assert_array_equal(s["label"], np.full((50, 7), k))
        np.testing.assert_array_equal(s["pushed"], [0.0, 0.0, 0.0] if k == 10 else [0.3, 0.0, 0.0])
        assert s["resting"] == (k == 10)
    assert not any(
        key in states[0] for key in ("label", "pushed", "resting")
    )  # task-prompted: labelled later


def test_frozen_noise_is_drawn_once_per_state_and_served_through_its_core():
    states = [{"core": np.array([0.5, 0.0, 0.0])}, {"core": np.zeros(3)}]  # the second: the operator rests
    tum.freeze_noise(states, (0.2, 0.2), np.random.default_rng(0))
    draws = np.random.default_rng(0).normal(size=(2, 3))
    for state, draw in zip(states, draws, strict=True):
        np.testing.assert_array_equal(state["noise"], draw)
        assert state["sigma"] == pytest.approx(0.2)
    kept = [dict(s) for s in states]
    tum.freeze_noise(
        states, (0.0, 1.0), np.random.default_rng(1)
    )  # a state keeps its draw (another arm's run)
    np.testing.assert_array_equal(states[0]["noise"], kept[0]["noise"])
    rows = [
        states[0],
        states[1],
        {"core": np.array([0.4, 0.0, 0.0])},
    ]  # the last: no frozen draw, served afresh
    served = tum.batch_commands(rows, np.full(3, 0.1), np.random.default_rng(2))
    np.testing.assert_allclose(
        served[:2], np.clip(np.stack([s["core"] for s in states]) + 0.2 * draws, -1.0, 1.0)
    )
    fresh = np.random.default_rng(2).normal(size=(3, 3))[2]
    np.testing.assert_allclose(served[2], np.clip(rows[2]["core"] + 0.1 * fresh, -1.0, 1.0))


def test_commands_are_relabelled_through_the_arms_readout_and_a_faithful_one():
    from config import ControlSettings

    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    raw = np.array([0.6, -0.2, 0.4])
    states = [{"raw": raw, "episode": 0, "step": 0, "core": np.zeros(3)}]  # cores from another user's file
    tum.label_commands(states, ControlSettings(corruption_matrix=turn, resolved_translation_gain=1.0))
    np.testing.assert_allclose(states[0]["core"], turn @ raw)
    np.testing.assert_allclose(states[0]["intent"], raw)  # what a readout without the corruption serves
    tum.label_commands(states, ControlSettings(corruption_quantize=True, resolved_translation_gain=1.0))
    np.testing.assert_allclose(states[0]["core"], [np.sqrt(0.56), 0.0, 0.0])
    np.testing.assert_allclose(states[0]["intent"], raw)


def test_executed_summary_splits_by_whether_the_plan_agrees_with_the_intent():
    intent = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.01]])  # the last: idle
    plan = np.array([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    arms = {
        "none": np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "user": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    out = tum.executed_summary(intent, plan, arms)
    assert (out["off_n"], out["on_n"]) == (1, 2)
    assert out["off_plan"] == pytest.approx(-1.0) and out["on_plan"] == pytest.approx(1.0)
    assert out["off_none"] == pytest.approx(0.0) and out["on_none"] == pytest.approx(0.5)
    assert out["off_user"] == pytest.approx(1.0) and out["on_user"] == pytest.approx(1.0)
    assert out["on_len_user"] == pytest.approx(1.0) and out["off_len_plan"] == pytest.approx(1.0)  # |moved|
    empty = tum.executed_summary(intent[:1], plan[:1], {"none": arms["none"][:1]})
    assert empty["off_n"] == 0 and empty["off_none"] is None and empty["rest_n"] == 0
    # a resting operator pushes nothing; its states are scored against what it means
    meant = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    resting = np.array([False, False, False, True])
    out = tum.executed_summary(intent, plan, arms, meant=meant, resting=resting)
    assert (out["off_n"], out["on_n"], out["rest_n"]) == (1, 2, 1)
    assert out["rest_plan"] == pytest.approx(1.0) and out["rest_none"] == pytest.approx(1.0)


def test_a_translation_module_corrects_the_translation_alone():
    from lerobot.policies.steering import AdaptedField

    args = tum.parse_args(
        ["--config", "c.yaml", "--out", "o", "--dims", "3", "--step-rank", "1", "--d", "64"]
    )
    user = tum.build_user_module(args, 50)
    assert (
        user.n_dims == 3 and sum(p.numel() for p in user.parameters()) < 30_000
    )  # the default: about 800 000
    torch.manual_seed(0)
    with torch.no_grad():  # away from the identity it starts at
        for p in user.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x, v = torch.randn(2, 50, 32), torch.randn(2, 50, 32)
    out = AdaptedField(None, user).adapt(x, 0.3, v, backward=True)
    torch.testing.assert_close(out[..., 3:], v[..., 3:])  # rotation, gripper and padding pass through
    assert not torch.allclose(out[..., :3], v[..., :3])
    assert tum.parse_args(["--config", "c.yaml", "--out", "o"]).dims == 7  # every action dim unless asked
    args = tum.parse_args(["--config", "c.yaml", "--out", "o", "--dims", "3", "--additive"])
    assert tum.build_user_module(args, 50).additive and not user.additive  # U v + b only when asked
    with pytest.raises(SystemExit):  # a coupling on a dim the module does not have
        tum.parse_args(["--config", "c.yaml", "--out", "o", "--dims", "3", "--cross-step-rank", "1",
                        "--cross-step-dims", "dx,droll"])  # fmt: skip


def test_the_depth_loss_counts_the_executed_translation_of_a_partial_reversal():
    """At depth d the module is fitted along the straight path from the user's reference (t = 0) to the meant chunk
    reversed d steps (t = d / N), on the pinned steps' translation: what steering keeps of a partial reversal."""
    torch.manual_seed(0)
    user = LearnedUserModule(horizon=CHUNK, n_dims=3, d=16, tdim=8)  # the identity at init
    start, end = torch.randn(4, CHUNK, MAX_DIM), torch.randn(4, CHUNK, MAX_DIM)
    t_end, t = 0.8, torch.tensor([0.0, 0.2, 0.5, 0.79])
    exact = lambda x, tt: (end - start) / t_end  # noqa: E731  (a field already on the path)
    assert tum.depth_loss(user, exact, start, end, t, t_end, steps=2).item() == pytest.approx(0.0, abs=1e-10)
    off = end.clone()
    off[:, 2:] += 5.0  # past the pinned steps
    off[..., 3:] += 5.0  # rotation, gripper, padding
    moved = lambda x, tt: (off - start) / t_end  # noqa: E731
    assert tum.depth_loss(user, moved, start, off, t, t_end, steps=2).item() == pytest.approx(0.0, abs=1e-10)
    wrong = lambda x, tt: torch.zeros_like(x)  # noqa: E731
    loss = tum.depth_loss(user, wrong, start, end, t, t_end, steps=2)
    target = ((end - start) / t_end)[:, :2, :3]
    assert loss.item() == pytest.approx((target**2).mean().item(), rel=1e-5)
    args = tum.parse_args(["--config", "c.yaml", "--out", "o", "--depth", "8"])
    assert args.depth == 8 and args.val_depths[0] == 8  # its own depth checked first
    assert (
        tum.parse_args(["--config", "c.yaml", "--out", "o"]).depth is None
    )  # the full reversal unless asked
    with pytest.raises(SystemExit):
        tum.parse_args(["--config", "c.yaml", "--out", "o", "--depth", "0"])
