"""CPU tests for the shared-autonomy steering wrappers (no checkpoint needed)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.pi05.steering import (
    DEADBAND,
    GRIPPER_DIM,
    N_ACTION_DIMS,
    FlowControlPolicy,
    FlowReversalSteeringPolicy,
    ReversalAdapter,
    build_reversal_adapter,
    get_action_mean_std,
    reverse_flow,
    rotation_about_z,
    translation_matrix,
    validate_matrix,
)

NUM_STEPS = 10
CHUNK = 4
MAX_DIM = 32


class FakePolicy:
    """Records the kwargs of predict_action_chunk and runs the hooks like the real sampler."""

    def __init__(self):
        self.config = SimpleNamespace(
            num_inference_steps=NUM_STEPS, chunk_size=CHUNK, max_action_dim=MAX_DIM, n_action_steps=2
        )
        self.calls: list[dict] = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(kwargs)
        x = torch.zeros(1, CHUNK, MAX_DIM)
        noise_fn = kwargs.get("noise_fn")
        if noise_fn is not None:
            x = noise_fn(lambda x_t, t: torch.ones_like(x_t), x)
        hook = kwargs.get("x_t_hook")
        if hook is not None:
            for step in range(NUM_STEPS):
                x = hook(step, 1.0 - step / NUM_STEPS, x)
        return x


class Source:
    def __init__(self, translation=(0.0, 0.0, 0.0), gripper=-1.0):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.gripper = gripper


def postprocessor(mean=None, std=None):
    mean = np.zeros(N_ACTION_DIMS) if mean is None else np.asarray(mean)
    std = np.ones(N_ACTION_DIMS) if std is None else np.asarray(std)
    return SimpleNamespace(steps=[SimpleNamespace(stats={"action": {"mean": mean, "std": std}})])


# ---------------------------------------------------------------- matrices


def test_validate_matrix_rejects_shape_and_nan():
    with pytest.raises(ValueError, match="3x3"):
        validate_matrix([[1, 0], [0, 1]], 3, "m")
    with pytest.raises(ValueError, match="finite"):
        validate_matrix([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], 3, "m")


def test_rotation_about_z_is_exact():
    r = rotation_about_z(90)
    np.testing.assert_allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation_about_z(0), np.eye(3))


def test_translation_matrix_forms():
    np.testing.assert_allclose(translation_matrix({"rotation_z_deg": 20}), rotation_about_z(20))
    np.testing.assert_allclose(translation_matrix({"scale": [1, 2, 3]}), np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(translation_matrix({"M": np.eye(3).tolist()}), np.eye(3))
    np.testing.assert_allclose(translation_matrix(np.eye(3).tolist()), np.eye(3))
    with pytest.raises(ValueError, match="rotation_z_deg"):
        translation_matrix({"rotate": 20})
    with pytest.raises(ValueError, match="scale"):
        translation_matrix({"scale": [1, 2]})


def test_build_reversal_adapter_blocks():
    m = rotation_about_z(20)
    f = build_reversal_adapter({"translation": "corruption", "orientation": "zero", "gripper": "zero"}, m)
    assert f.shape == (7, 7)
    np.testing.assert_allclose(f[:3, :3], m)
    np.testing.assert_allclose(f[3:, :], 0)
    np.testing.assert_allclose(f[:3, 3:], 0)

    f = build_reversal_adapter({"translation": {"scale": [2, 2, 2]}})
    np.testing.assert_allclose(f[:3, :3], 2 * np.eye(3))
    np.testing.assert_allclose(f[3:, 3:], np.eye(4))

    assert build_reversal_adapter(None) is None
    np.testing.assert_allclose(build_reversal_adapter({"F": np.eye(7).tolist()}), np.eye(7))


def test_build_reversal_adapter_errors():
    with pytest.raises(ValueError, match="corruption"):
        build_reversal_adapter({"translation": "corruption"}, None)
    with pytest.raises(ValueError, match="orientation"):
        build_reversal_adapter({"orientation": "half"})
    with pytest.raises(ValueError, match="unknown"):
        build_reversal_adapter({"rotation": "zero"})
    with pytest.raises(ValueError, match="F"):
        build_reversal_adapter({"F": np.eye(7).tolist(), "gripper": "zero"})


def test_reversal_adapter_holder():
    holder = ReversalAdapter()
    assert holder.matrix is None
    assert "off" in holder.describe()
    holder.matrix = np.eye(7)
    holder.label = "identity"
    assert "identity" in holder.describe()
    with pytest.raises(ValueError):
        holder.matrix = np.eye(3)


# ---------------------------------------------------------------- reverse_flow


def test_reverse_flow_constant_field_closed_form():
    x = torch.zeros(1, CHUNK, MAX_DIM)
    out = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS)
    torch.testing.assert_close(out, torch.ones_like(x))  # 10 steps of h=0.1, v=1
    out = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS, n_reversal_steps=3)
    torch.testing.assert_close(out, torch.full_like(x, 0.3))


def test_reverse_flow_visits_times_zero_to_one():
    seen = []

    def velocity(x_t, t):
        seen.append(round(t, 6))
        return torch.zeros_like(x_t)

    reverse_flow(torch.zeros(1, CHUNK, MAX_DIM), velocity, NUM_STEPS)
    assert seen == [round(k / NUM_STEPS, 6) for k in range(NUM_STEPS)]


def test_reverse_flow_adapter_touches_only_action_dims():
    x = torch.zeros(1, CHUNK, MAX_DIM)
    adapter = torch.zeros(7, 7)
    adapter[0, 0] = 2.0  # double dx, kill the other six action dims
    out = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS, adapter=adapter)
    torch.testing.assert_close(out[..., 0], torch.full((1, CHUNK), 2.0))
    torch.testing.assert_close(out[..., 1:7], torch.zeros(1, CHUNK, 6))
    torch.testing.assert_close(out[..., 7:], torch.ones(1, CHUNK, MAX_DIM - 7))  # padding untouched
    identity = reverse_flow(x, lambda x_t, t: torch.ones_like(x_t), NUM_STEPS, adapter=torch.eye(7))
    torch.testing.assert_close(identity, torch.ones_like(x))


# ---------------------------------------------------------------- FlowControlPolicy


def test_get_action_mean_std_reads_first_action_stats():
    mean, std = get_action_mean_std(postprocessor(mean=np.arange(7), std=np.zeros(7)))
    np.testing.assert_allclose(mean, np.arange(7))
    assert np.all(std >= 1e-6)
    with pytest.raises(RuntimeError):
        get_action_mean_std(SimpleNamespace(steps=[]))


def test_flow_control_writes_normalized_translation_for_tau_steps():
    policy = FakePolicy()
    source = Source(translation=(0.5, 0.0, -0.5))
    wrapper = FlowControlPolicy(policy, source, 3, postprocessor(mean=[0.1] * 7, std=[0.5] * 7))
    wrapper.reset()
    action = wrapper.select_action({})
    assert action.shape == (1, MAX_DIM)
    assert wrapper.guided_steps == 3
    expected = (np.array([0.5, 0.0, -0.5]) - 0.1) / 0.5
    np.testing.assert_allclose(action[0, :3].numpy(), expected, atol=1e-6)
    assert policy.resets == 1


def test_flow_control_skips_idle_input_and_queues_actions():
    policy = FakePolicy()
    wrapper = FlowControlPolicy(policy, Source(translation=(DEADBAND / 2, 0, 0)), 5, postprocessor())
    first = wrapper.select_action({})
    second = wrapper.select_action({})
    assert wrapper.guided_steps == 0
    torch.testing.assert_close(first, torch.zeros(1, MAX_DIM))
    torch.testing.assert_close(second, torch.zeros(1, MAX_DIM))
    assert len(policy.calls) == 1  # n_action_steps=2 actions per chunk


# ---------------------------------------------------------------- FlowReversalSteeringPolicy


def test_reverse_flow_steering_passes_schedule_only_while_steering():
    policy = FakePolicy()
    source = Source(translation=(0.0, 0.0, 0.0))
    wrapper = FlowReversalSteeringPolicy(policy, source, postprocessor(), n_reversal_steps=4)
    wrapper.select_action({})
    assert "flow_start_time" not in policy.calls[-1]
    assert wrapper.steered_chunks == 0

    source.translation = np.array([1.0, 0.0, 0.0])
    wrapper.select_action({})
    wrapper.select_action({})  # served from the queue, no new call
    assert len(policy.calls) == 2
    assert policy.calls[-1]["flow_start_time"] == pytest.approx(0.4)
    assert policy.calls[-1]["num_forward_steps"] == 6
    assert wrapper.steered_chunks == 1
    assert len(wrapper.reconstruction_errors) == 1


def test_reverse_flow_steering_full_reversal_has_no_schedule_kwargs():
    policy = FakePolicy()
    wrapper = FlowReversalSteeringPolicy(policy, Source(translation=(1.0, 0, 0)), postprocessor())
    wrapper.select_action({})
    assert "flow_start_time" not in policy.calls[-1]
    assert wrapper.n_reversal_steps is None


def test_reverse_flow_steering_validates_n_reversal_steps():
    wrapper = FlowReversalSteeringPolicy(FakePolicy(), Source(), postprocessor())
    for bad in (0, 11, 2.5, "3"):
        with pytest.raises(ValueError):
            wrapper.n_reversal_steps = bad
    wrapper.n_reversal_steps = NUM_STEPS
    assert wrapper.n_reversal_steps is None  # N means full reversal


def test_reference_chunk_uses_last_gripper_and_normalization():
    policy = FakePolicy()
    post = postprocessor(mean=[0.0] * 6 + [0.5], std=[2.0] * 7)
    wrapper = FlowReversalSteeringPolicy(policy, Source(translation=(1.0, 0, 0)), post)
    ref = wrapper.reference_chunk(np.array([1.0, 0.0, 0.0]))
    assert ref.shape == (1, CHUNK, MAX_DIM)
    assert ref[0, 0, 0].item() == pytest.approx(0.5)  # (1 - 0) / 2
    assert ref[0, 0, GRIPPER_DIM].item() == pytest.approx((-1.0 - 0.5) / 2.0)  # open at start
