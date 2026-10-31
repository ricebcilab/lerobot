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
        self.last_latent: torch.Tensor | None = None
        self.last_chunk: torch.Tensor | None = None

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(kwargs)
        x = torch.zeros(1, CHUNK, MAX_DIM)
        noise_fn = kwargs.get("noise_fn")
        if noise_fn is not None:
            x = noise_fn(lambda x_t, t: torch.ones_like(x_t), x)
            self.last_latent = x.clone()
        hook = kwargs.get("x_t_hook")
        if hook is not None:
            for step in range(NUM_STEPS):
                x = hook(step, 1.0 - step / NUM_STEPS, x)
            self.last_chunk = x.clone()  # the steered chunk; the paired plain pass has no hook
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

    f = build_reversal_adapter({"translation": "corruption_inverse"}, m)
    np.testing.assert_allclose(f[:3, :3], np.linalg.inv(m))
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


def test_adapter_normalization_preserves_environment_velocity_transform():
    # Unequal scales and cross-group mixing expose copying F directly into model space.
    std = np.array([0.3, 0.4, 0.5, 0.04, 0.06, 0.08, 1.0])
    matrix = np.eye(7)
    matrix[:3, :3] = rotation_about_z(20)
    matrix[0, 3] = 0.2
    adapter = ReversalAdapter(matrix)
    normalized_velocity = np.arange(1.0, 8.0)
    actual = std * (adapter.normalized_matrix(std) @ normalized_velocity)
    expected = matrix @ (std * normalized_velocity)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(ReversalAdapter(np.eye(7)).normalized_matrix(std), np.eye(7))
    assert ReversalAdapter().normalized_matrix(std) is None


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


def test_get_action_mean_std_follows_the_normalization_mode():
    q01, q99 = np.full(7, -0.02), np.full(7, 0.06)
    step = SimpleNamespace(stats={"action": {"mean": np.ones(7), "std": np.ones(7), "q01": q01, "q99": q99}})
    step.norm_map = {"ACTION": "QUANTILES"}
    mean, std = get_action_mean_std(SimpleNamespace(steps=[step]))
    np.testing.assert_allclose(mean, 0.02)  # 2 (x - q01) / (q99 - q01) - 1 == (x - mean) / std
    np.testing.assert_allclose(std, 0.04)
    step.norm_map = {"ACTION": "MEAN_STD"}
    mean, std = get_action_mean_std(SimpleNamespace(steps=[step]))
    np.testing.assert_allclose(mean, 1.0)
    step.stats["action"].update(min=np.zeros(7), max=np.full(7, 4.0))
    step.norm_map = {"ACTION": "MIN_MAX"}
    mean, std = get_action_mean_std(SimpleNamespace(steps=[step]))
    np.testing.assert_allclose(mean, 2.0)
    np.testing.assert_allclose(std, 2.0)


def test_wrappers_take_the_deadband_in_the_source_units():
    source = Source(translation=(0.01, 0.0, 0.0))  # a full push in a 0.01-per-unit action space
    wrapper = FlowControlPolicy(FakePolicy(), source, postprocessor(), lam=3, deadband=0.001)
    wrapper.select_action({})
    assert wrapper.guided_steps == 3
    wrapper = FlowControlPolicy(FakePolicy(), source, postprocessor(), lam=3)  # DEADBAND of 0.05: idle
    wrapper.select_action({})
    assert wrapper.guided_steps == 0
    wrapper = FlowReversalSteeringPolicy(FakePolicy(), source, postprocessor(), deadband=0.001)
    wrapper.select_action({})
    assert wrapper.steered_chunks == 1
    wrapper = FlowReversalSteeringPolicy(FakePolicy(), source, postprocessor())
    wrapper.select_action({})
    assert wrapper.steered_chunks == 0


def test_flow_control_kao_writes_the_scaled_command_into_tau_chunk_steps_at_lambda_euler_steps():
    # Kao et al.: ic_gain * command (noise-prior units, no mean/std) into the first `tau` chunk
    # steps of x_t at the start of each of the first `lam` Euler steps; the rest of the chunk and
    # the other dims are untouched.
    policy = FakePolicy()
    source = Source(translation=(0.5, 0.0, -0.5))
    wrapper = FlowControlPolicy(
        policy, source, postprocessor(mean=[0.1] * 7, std=[0.5] * 7), tau=2, lam=3, ic_gain=2.0
    )
    wrapper.reset()
    action = wrapper.select_action({})
    assert action.shape == (1, MAX_DIM)
    assert wrapper.guided_steps == 3  # one write per Euler step, lam of them
    written = policy.last_chunk  # the chunk as the hooks left it
    np.testing.assert_allclose(written[0, :2, :3].numpy(), np.array([[1.0, 0.0, -1.0]] * 2), atol=1e-6)
    torch.testing.assert_close(written[0, 2:, :], torch.zeros(CHUNK - 2, MAX_DIM))
    torch.testing.assert_close(written[0, :2, 3:], torch.zeros(2, MAX_DIM - 3))
    assert policy.resets == 1


def test_flow_control_action_units_write_the_whole_chunk_once_like_the_recorded_study():
    # The lab's recorded arm = Kao's method with tau = the whole chunk, lam 1 and the command in
    # action units ((c - mean) / std), the default knobs.
    policy = FakePolicy()
    wrapper = FlowControlPolicy(
        policy, Source(translation=(0.5, 0.0, -0.5)), postprocessor(mean=[0.1] * 7, std=[0.5] * 7)
    )
    assert wrapper.tau == CHUNK and wrapper.lam == 1 and wrapper.ic_gain == "action"
    wrapper.select_action({})
    assert wrapper.guided_steps == 1
    expected = (np.array([0.5, 0.0, -0.5]) - 0.1) / 0.5
    np.testing.assert_allclose(policy.last_chunk[0, :, :3].numpy(), np.tile(expected, (CHUNK, 1)), atol=1e-6)


def test_flow_control_lambda_zero_is_the_policy_alone():
    policy = FakePolicy()
    wrapper = FlowControlPolicy(policy, Source(translation=(1.0, 0.0, 0.0)), postprocessor(), lam=0)
    wrapper.select_action({})
    assert wrapper.guided_steps == 0
    assert len(policy.calls) == 1 and "x_t_hook" not in policy.calls[0]  # no steering, no paired pass


def test_flow_control_validates_its_knobs():
    policy, src, post = FakePolicy(), Source(translation=(1.0, 0.0, 0.0)), postprocessor()
    with pytest.raises(ValueError, match="tau"):
        FlowControlPolicy(policy, src, post, tau=CHUNK + 1)
    with pytest.raises(ValueError, match="tau"):
        FlowControlPolicy(policy, src, post, tau=0)
    with pytest.raises(ValueError, match="lam"):
        FlowControlPolicy(policy, src, post, lam=NUM_STEPS + 1)
    with pytest.raises(ValueError, match="ic_gain"):
        FlowControlPolicy(policy, src, post, ic_gain=0.0)
    with pytest.raises(ValueError, match="ic_gain"):
        FlowControlPolicy(policy, src, post, ic_gain="noise")


def test_flow_control_skips_idle_input_and_queues_actions():
    policy = FakePolicy()
    wrapper = FlowControlPolicy(policy, Source(translation=(DEADBAND / 2, 0, 0)), postprocessor(), lam=5)
    first = wrapper.select_action({})
    second = wrapper.select_action({})
    assert wrapper.guided_steps == 0
    torch.testing.assert_close(first, torch.zeros(1, MAX_DIM))
    torch.testing.assert_close(second, torch.zeros(1, MAX_DIM))
    assert len(policy.calls) == 1  # n_action_steps=2 actions per chunk


def test_flow_control_consumes_one_noisy_command_per_chunk():
    class ChangingSource:
        reads = 0

        @property
        def translation(self):
            self.reads += 1
            return np.array([self.reads / 10, 0.0, 0.0])

    source = ChangingSource()
    wrapper = FlowControlPolicy(FakePolicy(), source, postprocessor(), lam=NUM_STEPS)
    first = wrapper.select_action({})
    second = wrapper.select_action({})
    assert source.reads == 1
    torch.testing.assert_close(first, second)
    assert first[0, 0].item() == pytest.approx(0.1)
    wrapper.select_action({})
    assert source.reads == 2


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
    steered, plain = policy.calls[1:]  # a steered chunk costs a second, plain pass for the policy's own plan
    assert len(policy.calls) == 3 and plain == {}
    assert steered["flow_start_time"] == pytest.approx(0.4)
    assert steered["num_forward_steps"] == 4
    assert wrapper.steered_chunks == 1
    assert len(wrapper.reconstruction_errors) == 1


def test_reverse_flow_steering_full_reversal_has_no_schedule_kwargs():
    policy = FakePolicy()
    wrapper = FlowReversalSteeringPolicy(policy, Source(translation=(1.0, 0, 0)), postprocessor())
    wrapper.select_action({})
    assert "noise_fn" in policy.calls[0] and "flow_start_time" not in policy.calls[0]
    assert wrapper.n_reversal_steps is None


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 10])
def test_partial_reversal_roundtrip_and_evaluation_budget(depth):
    class IntegratingPolicy(FakePolicy):
        call_times: list[list[float]] = []

        def predict_action_chunk(self, batch, **kwargs):
            self.times = []
            self.call_times.append(self.times)

            def velocity(x, time):
                self.times.append(time)
                return torch.ones_like(x)

            noise_fn = kwargs.get("noise_fn")
            x = torch.zeros(1, CHUNK, MAX_DIM)
            if noise_fn is not None:
                x = noise_fn(velocity, x)
            start = kwargs.get("flow_start_time", 1.0)
            steps = kwargs.get("num_forward_steps", NUM_STEPS)
            dt = -start / steps
            self.forward_dt = dt
            for i in range(steps):
                x = x + dt * velocity(x, start + i * dt)
            return x

    policy = IntegratingPolicy()
    wrapper = FlowReversalSteeringPolicy(
        policy, Source((0.5, 0.0, -0.5)), postprocessor(), n_reversal_steps=depth
    )
    actual = wrapper.select_action({})
    # Exact constant-field inversion preserves the commanded dimensions, while
    # delegated ones complete the ordinary noise -> action trajectory.
    torch.testing.assert_close(actual[0, :3], torch.tensor([0.5, 0.0, -0.5]))
    torch.testing.assert_close(actual[0, 3:], torch.full((MAX_DIM - 3,), -1.0))
    assert policy.forward_dt == pytest.approx(-1.0 / NUM_STEPS)
    steered, plain = policy.call_times[-2:]
    assert len(steered) == NUM_STEPS + depth  # n reverse, N - n unsteered, n forward
    assert len(plain) == NUM_STEPS  # the policy's own plan, for the record
    torch.testing.assert_close(wrapper.last_policy_action, torch.full((1, MAX_DIM), -1.0))


def test_wrapper_converts_adapter_before_reversal():
    policy = FakePolicy()
    std = np.array([2.0, 4.0, 6.0, 1.0, 1.0, 1.0, 1.0])
    matrix = np.eye(7)
    matrix[:3, :3] = rotation_about_z(90)
    wrapper = FlowReversalSteeringPolicy(
        policy,
        Source((1.0, 0.0, 0.0)),
        postprocessor(std=std),
        adapter=ReversalAdapter(matrix),
        n_reversal_steps=2,
    )
    wrapper.select_action({})
    physical_delta = matrix @ std  # FakePolicy uses normalized v = ones.
    expected = np.array([1.0, 0.0, 0.0]) / std[:3] + 0.2 * physical_delta[:3] / std[:3]
    np.testing.assert_allclose(policy.last_latent[0, 0, :3].numpy(), expected, atol=1e-6)


def test_a_replaced_adapter_matrix_is_used_even_when_it_reuses_the_old_one_s_memory():
    """Per-trial adapters: clearing the adapter and setting a new matrix must not serve the old one."""
    policy = FakePolicy()
    adapter = ReversalAdapter()
    wrapper = FlowReversalSteeringPolicy(
        policy, Source((1.0, 0.0, 0.0)), postprocessor(), adapter=adapter, n_reversal_steps=1
    )
    latents = []
    for gain in (2.0, 3.0, 4.0):
        adapter.matrix = None  # frees the previous matrix, so the next one can land on its address
        matrix = np.eye(7)
        matrix[0, 0] = gain
        adapter.matrix = matrix
        del matrix
        wrapper.reset()
        wrapper.select_action({})
        latents.append(float(policy.last_latent[0, 0, 0]))
    # FakePolicy's velocity is all ones, so one reversal step of 1/10 moves x by 0.1 * gain.
    np.testing.assert_allclose(latents, [1.2, 1.3, 1.4], atol=1e-6)


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
    wrapper = FlowReversalSteeringPolicy(policy, Source(translation=(1.0, 0, 0)), post, gripper_open=0.0)
    ref = wrapper.reference_chunk(np.array([1.0, 0.0, 0.0]))
    assert ref[0, 0, GRIPPER_DIM].item() == pytest.approx((0.0 - 0.5) / 2.0)  # a 0/1 gripper's open state
    wrapper.reset()
    assert wrapper._gripper_ref == pytest.approx((0.0 - 0.5) / 2.0)


# ---------------------------------------------------------------- delegation to the policy


def test_forward_flow_constant_field_closed_form():
    from lerobot.policies.pi05.steering import forward_flow

    x = torch.zeros(1, CHUNK, MAX_DIM)
    times = []

    def velocity(x_t, t):
        times.append(round(t, 6))
        return torch.ones_like(x_t)

    out = forward_flow(x, velocity, NUM_STEPS, stop_at=4)  # 6 steps of -h * 1 from t=1 down to t=0.4
    assert out[0, 0, 0].item() == pytest.approx(-0.6)
    assert times == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    assert forward_flow(x, velocity, NUM_STEPS)[0, 0, 0].item() == pytest.approx(-1.0)


def test_delegation_mask_marks_uncommanded_dims_and_tail():
    from lerobot.policies.pi05.steering import delegation_mask

    mask = delegation_mask(CHUNK, MAX_DIM, ("translation",), pinned_steps=2)
    assert mask.shape == (CHUNK, MAX_DIM)
    assert not mask[:2, :3].any()  # translation, first two steps: the operator's
    assert mask[:2, 3:].all()  # rotation, gripper, padding: the policy's
    assert mask[2:, :].all()  # tail of the chunk: the policy's
    full = delegation_mask(CHUNK, MAX_DIM, ("translation", "rotation", "gripper"), pinned_steps=None)
    assert not full[:, :N_ACTION_DIMS].any() and full[:, N_ACTION_DIMS:].all()
    with pytest.raises(ValueError, match="operator_dims"):
        delegation_mask(CHUNK, MAX_DIM, ("translation", "wrist"), pinned_steps=2)


def test_reverse_flow_steering_delegates_uncommanded_elements():
    # FakePolicy: reference reversed under v=1 gains +h per step; the initial noise is zeros.
    policy = FakePolicy()
    source = Source(translation=(1.0, 0.0, 0.0))
    wrapper = FlowReversalSteeringPolicy(policy, source, postprocessor())  # full reversal
    wrapper.select_action({})
    latent = policy.last_latent
    assert latent[0, 0, 0].item() == pytest.approx(1.0 + 1.0)  # reference + 10 * 0.1
    assert latent[0, 0, 3:].abs().max().item() == 0.0  # rotation, gripper, padding = fresh noise
    assert latent[0, 2:, :].abs().max().item() == 0.0  # tail beyond n_action_steps = fresh noise

    wrapper = FlowReversalSteeringPolicy(policy, source, postprocessor(), n_reversal_steps=4)
    wrapper.select_action({})
    latent = policy.last_latent
    assert latent[0, 0, 0].item() == pytest.approx(1.0 + 0.4)
    assert latent[0, 0, 3].item() == pytest.approx(-0.6)  # unsteered state at t = 0.4

    wrapper = FlowReversalSteeringPolicy(
        policy,
        source,
        postprocessor(),
        operator_dims=("translation", "rotation", "gripper"),
        pinned_steps=None,
    )
    wrapper.select_action({})
    latent = policy.last_latent
    assert latent[0, 3, 5].item() == pytest.approx(0.0 + 1.0)  # rotation at the last step: kept
    assert latent[0, 3, N_ACTION_DIMS].item() == 0.0  # padding: always the policy's


def test_operator_dims_setter_rebuilds_the_mask():
    wrapper = FlowReversalSteeringPolicy(FakePolicy(), Source(), postprocessor())
    assert wrapper.operator_dims == ("translation",)
    assert wrapper._delegated[0, 3].item() is True
    wrapper.operator_dims = ["translation", "rotation"]
    assert wrapper._delegated[0, 3].item() is False
    with pytest.raises(ValueError, match="operator_dims"):
        wrapper.operator_dims = ["translation", "wrist"]


class NoisyPolicy(FakePolicy):
    """Draws its noise from torch's generator, like the real sampler; hooks may reshape it."""

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(kwargs)
        x = torch.randn(1, CHUNK, MAX_DIM)
        hook = kwargs.get("x_t_hook")
        if hook is not None:
            for step in range(NUM_STEPS):
                x = hook(step, 1.0 - step / NUM_STEPS, x)
            self.last_chunk = x.clone()  # the steered chunk; the paired plain pass has no hook
        return x


def test_flow_control_records_the_policy_plan_from_the_same_noise_without_disturbing_the_stream():
    policy = NoisyPolicy()
    wrapper = FlowControlPolicy(policy, Source((1.0, 0.0, 0.0)), postprocessor(), lam=2)
    torch.manual_seed(3)
    steered = wrapper.select_action({})
    plan = wrapper.last_policy_action
    after = torch.randn(3)  # whatever the run draws next

    torch.manual_seed(3)
    expected_plain = NoisyPolicy().predict_action_chunk({})[:, 0]
    torch.testing.assert_close(plan, expected_plain)  # same Gaussian sample as the steered pass
    assert not torch.equal(steered[0, :3], plan[0, :3])  # dims 0-2 were steered ...
    assert torch.equal(steered[0, 3:], plan[0, 3:])  # ... and nothing else

    # A single steered pass (no second one) leaves the generator exactly where the wrapper did.
    torch.manual_seed(3)
    alone = FlowControlPolicy(NoisyPolicy(), Source((1.0, 0.0, 0.0)), postprocessor(), lam=2)
    alone._pending_command = np.array([1.0, 0.0, 0.0])
    NoisyPolicy().predict_action_chunk({}, x_t_hook=alone._x_t_hook)
    torch.testing.assert_close(torch.randn(3), after)

    idle = FlowControlPolicy(NoisyPolicy(), Source((0.0, 0.0, 0.0)), postprocessor(), lam=2)
    action = idle.select_action({})
    assert idle.last_policy_action is action  # not steering: the plan is the action, no second pass
    assert len(idle._policy.calls) == 1


# ---------------------------------------------------------------- MARS: modular reversal adapters

from lerobot.policies.pi05.steering import (  # noqa: E402
    AdapterSpec,
    LearnedTaskModule,
    LearnedUserModule,
    forward_flow,
    load_adapter_module,
    parse_adapter_spec,
    save_adapter_module,
    sinusoidal_time_embedding,
)


def _constant_field(value=1.0):
    return lambda x_t, t: torch.full_like(x_t, value)


def test_learned_user_module_starts_as_the_identity_and_acts_backward_only():
    module = LearnedUserModule(horizon=CHUNK, n_dims=N_ACTION_DIMS)
    x = torch.randn(2, CHUNK, MAX_DIM)
    u = module(x, 0.3)
    assert u.shape == (2, CHUNK, N_ACTION_DIMS, N_ACTION_DIMS)
    torch.testing.assert_close(u, torch.eye(N_ACTION_DIMS).expand(2, CHUNK, -1, -1))
    with torch.no_grad():
        module.net[-1].bias.fill_(0.1)  # no longer the identity
    adapter = ReversalAdapter(user_module=module)
    x0 = torch.zeros(1, CHUNK, MAX_DIM)
    with_user = reverse_flow(x0, _constant_field(), NUM_STEPS, adapter=adapter)
    without = reverse_flow(x0, _constant_field(), NUM_STEPS)
    assert not torch.allclose(with_user[..., :N_ACTION_DIMS], without[..., :N_ACTION_DIMS])
    torch.testing.assert_close(
        with_user[..., N_ACTION_DIMS:], without[..., N_ACTION_DIMS:]
    )  # padding dims untouched
    # the user module does not act on the forward flow
    torch.testing.assert_close(
        forward_flow(x0, _constant_field(), NUM_STEPS, adapter=adapter),
        forward_flow(x0, _constant_field(), NUM_STEPS),
    )


def test_task_slot_acts_in_both_directions_and_composes_before_the_user_module():
    n = N_ACTION_DIMS
    eye = torch.eye(n)
    task = lambda x_t, t, v: (2.0 * eye.expand(v.shape[0], v.shape[1], n, n), torch.zeros_like(v))  # noqa: E731
    user = lambda x_t, t: 3.0 * eye.expand(x_t.shape[0], x_t.shape[1], n, n)  # noqa: E731
    adapter = ReversalAdapter(user_module=user, task_module=task)
    assert adapter.kind == "composed"
    x0 = torch.zeros(1, CHUNK, MAX_DIM)
    back = reverse_flow(x0, _constant_field(), NUM_STEPS, adapter=adapter)  # backward: user(task(v)) = 6 v
    torch.testing.assert_close(back[..., :n], torch.full((1, CHUNK, n), 6.0))
    torch.testing.assert_close(back[..., n:], torch.full((1, CHUNK, MAX_DIM - n), 1.0))
    fwd = forward_flow(x0, _constant_field(), NUM_STEPS, adapter=adapter)  # forward: task(v) = 2 v
    torch.testing.assert_close(fwd[..., :n], torch.full((1, CHUNK, n), -2.0))
    torch.testing.assert_close(fwd[..., n:], torch.full((1, CHUNK, MAX_DIM - n), -1.0))


def test_learned_task_module_starts_as_the_identity():
    module = LearnedTaskModule(horizon=CHUNK, n_dims=N_ACTION_DIMS)
    x, v = torch.randn(3, CHUNK, MAX_DIM), torch.randn(3, CHUNK, N_ACTION_DIMS)
    a, b = module(x, torch.full((3,), 0.5), v)
    torch.testing.assert_close(a, torch.eye(N_ACTION_DIMS).expand(3, CHUNK, -1, -1))
    torch.testing.assert_close(b, torch.zeros(3, CHUNK, N_ACTION_DIMS))


def test_adapter_module_checkpoint_roundtrip(tmp_path):
    user = LearnedUserModule(horizon=CHUNK, n_dims=N_ACTION_DIMS, d=32, tdim=8)
    with torch.no_grad():
        user.net[-1].bias.normal_()
    path = tmp_path / "user.pt"
    save_adapter_module(user, path, kind="user", meta={"corruption": {"rotation_z_deg": 20}})
    loaded, kind, meta = load_adapter_module(path)
    assert kind == "user" and meta["corruption"] == {"rotation_z_deg": 20}
    assert isinstance(loaded, LearnedUserModule) and loaded.config == user.config
    x = torch.randn(1, CHUNK, MAX_DIM)
    torch.testing.assert_close(loaded(x, 0.2), user(x, 0.2))
    task = LearnedTaskModule(horizon=CHUNK, n_dims=N_ACTION_DIMS, d=16, tdim=8)
    save_adapter_module(task, tmp_path / "task.pt", kind="task")
    loaded_task, kind, _ = load_adapter_module(tmp_path / "task.pt")
    assert kind == "task" and isinstance(loaded_task, LearnedTaskModule)
    with pytest.raises(ValueError, match="kind"):
        save_adapter_module(user, tmp_path / "bad.pt", kind="policy")
    adapter = ReversalAdapter()
    adapter.load_modules(user_path=path, task_path=tmp_path / "task.pt")
    assert adapter.kind == "composed" and "MARS" in adapter.describe()


@pytest.mark.parametrize("name", ["mars", "pi05"])
def test_learned_user_module_time_embedding(tmp_path, name):
    from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding

    module = LearnedUserModule(horizon=CHUNK, n_dims=N_ACTION_DIMS, d=16, tdim=8, time_embedding=name)
    seen = []
    module.net[0].register_forward_hook(lambda layer, args, out: seen.append(args[0][:, -8:]))
    t = torch.tensor([0.1, 0.7])
    module(torch.randn(2, CHUNK, MAX_DIM), t)
    if name == "mars":
        expected = sinusoidal_time_embedding(t, 8)
    else:
        expected = create_sinusoidal_pos_embedding(t, 8, 0.004, 4.0, t.device).float()
    torch.testing.assert_close(seen[0], expected)
    path = save_adapter_module(module, tmp_path / "user.pt", kind="user")
    assert load_adapter_module(path)[0].config["time_embedding"] == name
    # checkpoints saved before the option existed keep the MARS embedding
    ckpt = torch.load(path, weights_only=True)
    del ckpt["config"]["time_embedding"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert load_adapter_module(tmp_path / "old.pt")[0].time_embedding == "mars"
    with pytest.raises(ValueError, match="time_embedding"):
        LearnedUserModule(horizon=CHUNK, time_embedding="rope")


def test_parse_adapter_spec_forms():
    m = rotation_about_z(20)
    assert parse_adapter_spec(None, m) == AdapterSpec()
    fixed = parse_adapter_spec({"translation": "corruption", "gripper": "zero"}, m)
    np.testing.assert_allclose(fixed.matrix[:3, :3], m)
    assert fixed.user_path is None and fixed.task_path is None
    assert parse_adapter_spec({"learned": "u.pt"}, m) == AdapterSpec(user_path="u.pt")
    composed = parse_adapter_spec({"user": {"translation": "corruption"}, "task": "t.pt"}, m)
    np.testing.assert_allclose(composed.matrix[:3, :3], m)
    assert composed.task_path == "t.pt" and composed.user_path is None
    assert parse_adapter_spec({"user": {"learned": "u.pt"}, "task": None}, m) == AdapterSpec(user_path="u.pt")
    assert parse_adapter_spec({"user": None, "task": "t.pt"}, m) == AdapterSpec(task_path="t.pt")
    np.testing.assert_allclose(parse_adapter_spec({"F": np.eye(7).tolist()}, m).matrix, np.eye(7))
    for bad in (
        {"user": {"learned": "u"}, "F": np.eye(7).tolist()},
        {"learned": 1},
        {"task": 3},
        {"model": "x"},
        {"user": {"learned": "u", "translation": "corruption"}},
    ):
        with pytest.raises(ValueError):
            parse_adapter_spec(bad, m)


def test_reversal_adapter_kind_and_description():
    adapter = ReversalAdapter()
    assert adapter.kind == "none" and "off" in adapter.describe()
    adapter.matrix, adapter.label = np.eye(7), "identity"
    assert adapter.kind == "user" and "MARS-U" in adapter.describe()
    adapter.task_module = lambda x_t, t, v: (torch.eye(7).expand(1, CHUNK, 7, 7), torch.zeros(1, CHUNK, 7))
    assert adapter.kind == "composed" and "MARS" in adapter.describe()
    adapter.matrix = None
    assert adapter.kind == "task" and "MARS-T" in adapter.describe()


def test_wrapper_passes_a_velocity_hook_only_with_a_task_module():
    policy = FakePolicy()
    adapter = ReversalAdapter(
        task_module=lambda x_t, t, v: (torch.eye(7).expand(1, CHUNK, 7, 7), torch.zeros(1, CHUNK, 7))
    )
    wrapper = FlowReversalSteeringPolicy(
        policy, Source(translation=(0.0, 0.0, 0.0)), postprocessor(), adapter=adapter
    )
    wrapper.select_action({})  # idle: the task module still shapes the policy's own chunk
    assert "velocity_hook" in policy.calls[0] and "noise_fn" not in policy.calls[0]
    wrapper = FlowReversalSteeringPolicy(
        policy, Source(translation=(1.0, 0.0, 0.0)), postprocessor(), adapter=adapter
    )
    wrapper.select_action({})  # steering: both passes carry the hook, only the first the reversal
    steered, plain = policy.calls[1:]
    assert "velocity_hook" in steered and "noise_fn" in steered
    assert "velocity_hook" in plain and "noise_fn" not in plain
    wrapper = FlowReversalSteeringPolicy(
        policy, Source(translation=(1.0, 0.0, 0.0)), postprocessor(), adapter=ReversalAdapter()
    )
    wrapper.select_action({})
    assert "velocity_hook" not in policy.calls[-2] and "velocity_hook" not in policy.calls[-1]
