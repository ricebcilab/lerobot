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


def test_serve_noises_only_an_active_core():
    rng = np.random.default_rng(0)
    core = np.array(
        [[0.5, 0.0, 0.0], [0.02, 0.0, 0.0]]
    )  # the second: an active push the readout shrank to idle
    served = tum.serve(core, [0.1, 0.1], rng)
    assert not np.allclose(served[0], core[0]) and np.all(np.abs(served[0]) <= 1.0)
    np.testing.assert_array_equal(
        served[1], core[1]
    )  # the rollouts' rule: an idle command is served as it is
    np.testing.assert_allclose(tum.serve(core, 0.0, rng)[0], core[0])


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
        "path": torch.ones(3, CHUNK, MAX_DIM),
        "cue_path": torch.ones(2, 3, CHUNK, MAX_DIM),
        "cue_served": np.zeros((2, 3)),
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
    assert torch.is_tensor(back["path"]) and back["path"].shape == (3, CHUNK, MAX_DIM)
    assert torch.is_tensor(back["cue_path"]) and isinstance(back["cue_served"], np.ndarray)


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

        def set_mode(self, mode):
            pass

        def set_scene(self, suite, task_id):
            self.task_id = task_id

        def rollout(self, task, recorder=None, initial_observation=None, max_steps=None):
            assert initial_observation == {"reset": len(resets)}  # the rollout starts from the state just set
            prompts.append((task, max_steps))
            for k in range(
                max_steps or 25
            ):  # as Session.rollout reports: the step count after the step, the observation before it
                recorder(step=k + 1, observation={"k": k})

    settings = SimpleNamespace(
        task_ids=[0], prompt="do something", session=SimpleNamespace(suite="s", n_action_steps=10)
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

    # rollouts under the arm's own prompt follow, from the same start states, kept up to a step, with the scene's task
    resets.clear()
    prompts.clear()
    states = tum.collect_states(FakeSession(), settings, 2, np.random.default_rng(0), 2, 15)
    assert prompts == [("t", None), ("t", None), ("do something", 15), ("do something", 15)]
    assert resets == [0, 0, 0, 0]
    generic = [s for s in states if s["rollout"] == "generic"]
    assert [(s["episode"], s["step"], s["task"], s["task_id"]) for s in generic] == [
        (2, 0, "t", 3), (2, 10, "t", 3), (3, 0, "t", 7), (3, 10, "t", 7)
    ]  # fmt: skip
    assert {s["rollout"] for s in states[:6]} == {"task"}


def test_command_targets_bring_the_users_reference_onto_the_intended_reversal():
    n = 10
    field = lambda x, t: 0.5 * x + 1.0  # noqa: E731  (a field that depends on where the chunk is)
    intended = torch.randn(2, CHUNK, MAX_DIM)
    path = tum.reversal_path(Flow(field), intended, n)
    assert path.shape == (2, n + 1, CHUNK, MAX_DIM)
    torch.testing.assert_close(path[:, 0], intended)
    torch.testing.assert_close(path[:, -1], Flow(field).reverse(intended, n), atol=1e-5, rtol=0)
    served = intended.clone()
    served[..., :3] += torch.randn(2, CHUNK, 3)  # the user's reference: another translation
    for correct_by in (n, 4):  # over the whole reversal, or done by its fourth step
        x = served
        for k in range(n):
            at, target = tum.command_target(path, served, torch.full((2,), k), correct_by)
            # the regression's points are the adapted reversal's own
            torch.testing.assert_close(at, x, atol=1e-4, rtol=0)
            x = x + target / n
            if k + 1 >= correct_by:  # on the intended reversal from there on
                torch.testing.assert_close(x, path[:, k + 1], atol=1e-4, rtol=0)
    # a faithful command needs no correction: the plain field's own steps
    rows, k = torch.arange(2), torch.tensor([3, 7])
    at, target = tum.command_target(path, intended, k, 4)
    torch.testing.assert_close(at, path[rows, k])
    torch.testing.assert_close(target, (path[rows, k + 1] - path[rows, k]) * n)
    args = tum.parse_args(["--config", "c.yaml", "--out", "o"])
    assert (args.target, args.correct_by) == ("label", None)  # the cursor's recipe unless asked
    args = tum.parse_args(["--config", "c.yaml", "--out", "o", "--target", "command", "--correct-by", "4"])
    assert (args.target, args.correct_by) == ("command", 4)


def test_translation_loss_trains_the_translation_rows_only():
    torch.manual_seed(0)
    user = LearnedUserModule(horizon=CHUNK, n_dims=7, d=16, tdim=8)  # the identity at init
    x, v = torch.randn(4, CHUNK, MAX_DIM), torch.randn(4, CHUNK, MAX_DIM)
    t = torch.tensor([0.0, 0.1, 0.5, 0.9])
    field = lambda x_t, tt: v  # noqa: E731
    target = v.clone()
    target[..., :3] = 2 * v[..., :3]
    target[..., 3:7] = 5.0  # the other dims' targets do not count
    assert tum.translation_loss(user, field, x, t, v).item() == pytest.approx(0.0, abs=1e-6)
    start = tum.translation_loss(user, field, x, t, target)
    assert start.item() == pytest.approx((v[..., :3] ** 2).mean().item(), rel=1e-5)
    one = tum.translation_loss(user, field, x, t, target, weight=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert one.item() == pytest.approx((v[0, :, :3] ** 2).mean().item(), rel=1e-5)
    opt = torch.optim.SGD(user.parameters(), lr=0.05)
    for _ in range(20):
        opt.zero_grad()
        tum.translation_loss(user, field, x, t, target).backward()
        opt.step()
    assert tum.translation_loss(user, field, x, t, target).item() < 0.9 * start.item()
    with torch.no_grad():  # rotation and gripper rows stay the identity: their velocity passes unchanged
        torch.testing.assert_close(user(x, t)[..., 3:, :], torch.eye(7)[3:].expand(4, CHUNK, 4, 7))


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
    empty = tum.executed_summary(intent[:1], plan[:1], {"none": arms["none"][:1]})
    assert empty["off_n"] == 0 and empty["off_none"] is None


def test_cued_pushes_keep_each_length_and_hold_an_evenly_drawn_direction():
    rng = np.random.default_rng(0)
    lengths = np.full(6000, 0.5)
    lengths[:6] = [0.2, 0.4, 0.6, 0.8, 0.5, 0.5]
    episodes = np.repeat(np.arange(1000), 6)  # six chunk boundaries an episode
    pushes = tum.cue_pushes(lengths, episodes, rng, hold=3)
    np.testing.assert_allclose(np.linalg.norm(pushes, axis=1), lengths)  # the state's own push length
    blocks = (pushes / lengths[:, None]).reshape(-1, 3, 3)  # (cue, chunk of the hold, xyz)
    np.testing.assert_allclose(blocks[:, 0], blocks[:, 1])
    np.testing.assert_allclose(blocks[:, 0], blocks[:, 2])  # one direction for the three chunks of a cue
    assert not np.allclose(blocks[0, 0], blocks[1, 0])  # then another, within the episode
    assert np.abs(blocks[:, 0].mean(0)).max() < 0.05  # no preferred direction over the session
    short = tum.cue_pushes(np.ones(4), [0, 0, 1, 1], np.random.default_rng(1), hold=3)
    assert np.allclose(short[0], short[1]) and not np.allclose(short[1], short[2])  # a new episode, a new cue
    long = tum.cue_pushes(np.full(50, 1.7), np.zeros(50), rng, hold=1)
    assert np.abs(long).max() <= 1.0  # clipped as the operator's pushes are
    args = tum.parse_args(["--config", "c.yaml", "--out", "o"])
    assert (args.cues, args.cue_hold, args.cue_noise) == (0, 3, "recorded")  # off unless asked
    args = tum.parse_args(["--config", "c.yaml", "--out", "o", "--target", "command", "--cues", "4"])
    assert args.cues == 4
    with pytest.raises(SystemExit):  # a cue's label is its own reference: the command target
        tum.parse_args(["--config", "c.yaml", "--out", "o", "--cues", "4"])
