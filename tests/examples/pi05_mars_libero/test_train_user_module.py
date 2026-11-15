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
    from teleop import InputGain, ReadoutReader

    class Push:
        gripper = GRIPPER_OPEN

    pushes = np.array(
        [[0.6, -0.2, 0.4], [0.6, -0.2, 0.4], [0.1, 0.9, 0.0], [0.6, -0.2, 0.4], [0.9, 0.9, 0.9]]
    )
    episodes, steps = [0, 0, 0, 1, 1], [0, 10, 20, 0, 10]
    lag = ControlSettings(corruption_curl=Curl(2.0, 0.5), resolved_input_gain=1.0)
    cores = tum.readout_cores(pushes, episodes, steps, lag)
    # a rollout of episode 0: the field stepped at every control step, read at each chunk boundary with the new push
    source = Push()
    reader = ReadoutReader(InputGain(source, 1.0), curl=Curl(2.0, 0.5))
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
        tum.readout_cores(pushes[4:], [0], [0], ControlSettings(resolved_input_gain=1.0))[0]
    ) == pytest.approx(1.0)
    keypad = ControlSettings(corruption_quantize=True, resolved_input_gain=1.0)
    np.testing.assert_array_equal(tum.readout_cores(pushes[:1], [0], [0], keypad), [[1.0, 0.0, 0.0]])
    stick = ControlSettings(corruption_matrix=np.diag([1.0, 1.0, 0.0]), resolved_input_gain=1.0)
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

    class FakeSession:
        suite, task_id, task_description = "s", 0, "t"

        def set_mode(self, mode):
            pass

        def rollout(self, task, recorder=None):
            for k in range(
                25
            ):  # as Session.rollout reports: the step count after the step, the observation before it
                recorder(step=k + 1, observation={"k": k})

    settings = SimpleNamespace(task_ids=[0], session=SimpleNamespace(suite="s", n_action_steps=10))
    states = tum.collect_states(FakeSession(), settings, 2, np.random.default_rng(0))
    assert [(s["episode"], s["step"], s["observation"]["k"]) for s in states] == [
        (0, 0, 0), (0, 10, 10), (0, 20, 20), (1, 0, 0), (1, 10, 10), (1, 20, 20)
    ]  # fmt: skip
