import numpy as np
import pytest
import torch
import train_user_module as tum
from teleop import Curl

from lerobot.policies.pi05.steering import (
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LearnedUserModule,
    uniform_reference,
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


def test_channel_core_matrix_curl_and_idle():
    m = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(tum.channel_core([1.0, 0.0, 0.0], matrix=m), [0.0, 1.0, 0.0])
    curl = Curl(20.0, 0.1, dt=0.05)
    decay, drive = curl.discretization()
    v = np.zeros(3)
    for _ in range(10):
        v = decay @ v + drive @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(tum.channel_core([1.0, 0.0, 0.0], curl=curl, n_steps=10), np.clip(v, -1, 1))
    idle = np.array([DEADBAND / 2, 0.0, 0.0])
    np.testing.assert_allclose(tum.channel_core(idle, matrix=m), idle)
    np.testing.assert_allclose(tum.channel_core([0.5, 0.0, 0.0]), [0.5, 0.0, 0.0])


def test_serve_adds_noise_only_where_active():
    rng = np.random.default_rng(0)
    core = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    raw = np.array([[1.0, 0.0, 0.0], [DEADBAND / 2, 0.0, 0.0]])
    served = tum.serve(core, raw, [0.1, 0.1], rng)
    assert not np.allclose(served[0], core[0]) and np.all(np.abs(served[0]) <= 1.0)
    np.testing.assert_allclose(served[1], raw[1])
    np.testing.assert_allclose(tum.serve(core, raw, 0.0, rng)[0], core[0])


def test_field_loss_is_zero_for_the_exact_path_velocity_and_scales_otherwise():
    user = LearnedUserModule(horizon=CHUNK, n_dims=7, d=16, tdim=8)  # the identity at init
    a_bci, z_l = torch.randn(2, CHUNK, MAX_DIM), torch.randn(2, CHUNK, MAX_DIM)
    t = torch.rand(2)
    exact = lambda x_t, tt: z_l - a_bci  # noqa: E731
    assert tum.field_loss(user, exact, z_l, a_bci, t).item() == pytest.approx(0.0, abs=1e-6)
    doubled = lambda x_t, tt: 2 * (z_l - a_bci)  # noqa: E731
    expected = ((z_l - a_bci)[..., :7] ** 2).mean((1, 2))
    assert tum.field_loss(user, doubled, z_l, a_bci, t).item() == pytest.approx(
        expected.mean().item(), rel=1e-5
    )
    weighted = tum.field_loss(user, doubled, z_l, a_bci, t, weight=torch.tensor([1.0, 0.0]))
    assert weighted.item() == pytest.approx(expected[0].item(), rel=1e-5)
    weighted.backward()  # gradients reach the module
    assert user.net[-1].weight.grad is not None


def test_regenerate_round_trips_a_constant_field_and_heading_cosine():
    field = lambda x_t, t: torch.ones_like(x_t)  # noqa: E731
    a = torch.randn(3, CHUNK, MAX_DIM)
    torch.testing.assert_close(tum.regenerate(field, a, 10), a, atol=1e-5, rtol=0)
    chunk = torch.zeros(1, CHUNK, MAX_DIM)
    chunk[0, :, 0] = 1.0
    other = torch.zeros(1, CHUNK, MAX_DIM)
    other[0, :, 1] = 1.0
    assert tum.heading_cosine(chunk, chunk).item() == pytest.approx(1.0)
    assert tum.heading_cosine(chunk, other).item() == pytest.approx(0.0, abs=1e-6)


def test_stack_batches_and_batched_reference():
    b1 = {"observation.state": torch.zeros(1, 2), "task": ["a"]}
    b2 = {"observation.state": torch.ones(1, 2), "task": ["b"]}
    stacked = tum.stack_batches([b1, b2])
    assert stacked["observation.state"].shape == (2, 2) and stacked["task"] == ["a", "b"]
    ref = uniform_reference(
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
