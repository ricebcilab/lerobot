"""CPU tests for the shared-autonomy steering wrappers (no checkpoint needed)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.steering import (
    DEADBAND,
    GRIPPER_DIM,
    N_ACTION_DIMS,
    AdapterSpec,
    FlowControlPolicy,
    FlowReversalSteeringPolicy,
    LearnedTaskModule,
    LearnedUserModule,
    ReversalAdapter,
    build_reversal_adapter,
    delegation_mask,
    forward_flow,
    get_action_mean_std,
    load_adapter_module,
    parse_adapter_spec,
    reverse_flow,
    rotation_about_z,
    save_adapter_module,
    sinusoidal_time_embedding,
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

    def noise(self):
        return torch.zeros(1, CHUNK, MAX_DIM)

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(kwargs)
        x = self.noise()
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


class NoisyPolicy(FakePolicy):
    """Draws its noise from torch's generator, like the real sampler; hooks may reshape it."""

    def noise(self):
        return torch.randn(1, CHUNK, MAX_DIM)


class Source:
    def __init__(self, translation=(0.0, 0.0, 0.0), gripper=-1.0):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.gripper = gripper


def postprocessor(mean=None, std=None):
    mean = np.zeros(N_ACTION_DIMS) if mean is None else np.asarray(mean)
    std = np.ones(N_ACTION_DIMS) if std is None else np.asarray(std)
    return SimpleNamespace(steps=[SimpleNamespace(stats={"action": {"mean": mean, "std": std}})])


def _ones(x_t, t):
    return torch.ones_like(x_t)


def _frs(policy=None, translation=(1.0, 0.0, 0.0), post=None, **kw):
    policy, post = policy or FakePolicy(), post or postprocessor()
    return FlowReversalSteeringPolicy(policy, Source(translation), post, **kw)


def _fc(policy=None, translation=(1.0, 0.0, 0.0), post=None, **kw):
    return FlowControlPolicy(policy or FakePolicy(), Source(translation), post or postprocessor(), **kw)


# ---------------------------------------------------------------- matrices and adapters


def test_matrix_helpers_validate_rotate_and_build_translations():
    with pytest.raises(ValueError, match="3x3"):
        validate_matrix([[1, 0], [0, 1]], 3, "m")
    with pytest.raises(ValueError, match="finite"):
        validate_matrix([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], 3, "m")
    np.testing.assert_allclose(rotation_about_z(90) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation_about_z(0), np.eye(3))
    np.testing.assert_allclose(translation_matrix({"rotation_z_deg": 20}), rotation_about_z(20))
    np.testing.assert_allclose(translation_matrix({"scale": [1, 2, 3]}), np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(translation_matrix({"M": np.eye(3).tolist()}), np.eye(3))
    np.testing.assert_allclose(translation_matrix(np.eye(3).tolist()), np.eye(3))
    for bad, match in (({"rotate": 20}, "rotation_z_deg"), ({"scale": [1, 2]}, "scale")):
        with pytest.raises(ValueError, match=match):
            translation_matrix(bad)


def test_build_reversal_adapter_blocks_and_errors():
    m = rotation_about_z(20)
    f = build_reversal_adapter({"translation": "corruption", "orientation": "zero", "gripper": "zero"}, m)
    assert f.shape == (7, 7) and np.allclose(f[:3, :3], m) and not f[3:, :].any() and not f[:3, 3:].any()
    inv = build_reversal_adapter({"translation": "corruption_inverse"}, m)
    np.testing.assert_allclose(inv[:3, :3], np.linalg.inv(m))
    scaled = build_reversal_adapter({"translation": {"scale": [2, 2, 2]}})
    np.testing.assert_allclose(scaled[:3, :3], 2 * np.eye(3))
    np.testing.assert_allclose(scaled[3:, 3:], np.eye(4))
    assert build_reversal_adapter(None) is None
    np.testing.assert_allclose(build_reversal_adapter({"F": np.eye(7).tolist()}), np.eye(7))
    for spec, match in (
        ({"orientation": "half"}, "orientation"),
        ({"rotation": "zero"}, "unknown"),
        ({"F": np.eye(7).tolist(), "gripper": "zero"}, "F"),
    ):
        with pytest.raises(ValueError, match=match):
            build_reversal_adapter(spec)
    with pytest.raises(ValueError, match="corruption"):
        build_reversal_adapter({"translation": "corruption"}, None)


def test_reversal_adapter_kind_description_and_modules(tmp_path):
    adapter = ReversalAdapter()
    assert adapter.matrix is None and adapter.kind == "none" and "off" in adapter.describe()
    adapter.matrix, adapter.label = np.eye(7), "identity"
    assert adapter.kind == "user" and "identity" in adapter.describe() and "MARS-U" in adapter.describe()
    with pytest.raises(ValueError):
        adapter.matrix = np.eye(3)
    adapter.task_module = lambda x_t, t, v: (torch.eye(7).expand(1, CHUNK, 7, 7), torch.zeros(1, CHUNK, 7))
    assert adapter.kind == "composed" and "MARS" in adapter.describe()
    adapter.matrix = None
    assert adapter.kind == "task" and "MARS-T" in adapter.describe()
    user = LearnedUserModule(horizon=CHUNK, n_dims=N_ACTION_DIMS, d=32, tdim=8)
    save_adapter_module(user, tmp_path / "u.pt", kind="user", meta={"corruption": {"rotation_z_deg": 20}})
    task = LearnedTaskModule(horizon=CHUNK, n_dims=N_ACTION_DIMS, d=16, tdim=8)
    save_adapter_module(task, tmp_path / "t.pt", "task")
    assert load_adapter_module(tmp_path / "u.pt")[1:] == ("user", {"corruption": {"rotation_z_deg": 20}})
    with pytest.raises(ValueError, match="kind"):
        save_adapter_module(user, tmp_path / "bad.pt", kind="policy")
    loaded = ReversalAdapter()
    loaded.load_modules(user_path=tmp_path / "u.pt", task_path=tmp_path / "t.pt")
    assert loaded.kind == "composed" and "MARS" in loaded.describe()


def test_adapter_normalization_preserves_environment_velocity_transform():
    # Unequal scales and cross-group mixing expose copying F directly into model space.
    std = np.array([0.3, 0.4, 0.5, 0.04, 0.06, 0.08, 1.0])
    matrix = np.eye(7)
    matrix[:3, :3] = rotation_about_z(20)
    matrix[0, 3] = 0.2
    normalized_velocity = np.arange(1.0, 8.0)
    actual = std * (ReversalAdapter(matrix).normalized_matrix(std) @ normalized_velocity)
    np.testing.assert_allclose(actual, matrix @ (std * normalized_velocity))
    np.testing.assert_allclose(ReversalAdapter(np.eye(7)).normalized_matrix(std), np.eye(7))
    assert ReversalAdapter().normalized_matrix(std) is None


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
    ckpt = torch.load(path, weights_only=True)  # checkpoints saved before the option keep the MARS embedding
    del ckpt["config"]["time_embedding"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert load_adapter_module(tmp_path / "old.pt")[0].time_embedding == "mars"
    with pytest.raises(ValueError, match="time_embedding"):
        LearnedUserModule(horizon=CHUNK, time_embedding="rope")


# ---------------------------------------------------------------- the flows


def test_flows_have_constant_field_closed_forms_and_touch_only_action_dims():
    x, seen = torch.zeros(1, CHUNK, MAX_DIM), []

    def velocity(x_t, t):
        seen.append(round(t, 6))
        return torch.ones_like(x_t)

    torch.testing.assert_close(reverse_flow(x, velocity, NUM_STEPS), torch.ones_like(x))  # 10 of h=0.1, v=1
    assert seen == [round(k / NUM_STEPS, 6) for k in range(NUM_STEPS)]  # times 0 to 1
    torch.testing.assert_close(reverse_flow(x, _ones, NUM_STEPS, n_reversal_steps=3), torch.full_like(x, 0.3))
    seen.clear()
    stopped = forward_flow(x, velocity, NUM_STEPS, stop_at=4)  # t = 1 down to 0.4
    assert stopped[0, 0, 0].item() == pytest.approx(-0.6)
    assert seen == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    assert forward_flow(x, _ones, NUM_STEPS)[0, 0, 0].item() == pytest.approx(-1.0)
    adapter = torch.zeros(7, 7)
    adapter[0, 0] = 2.0  # double dx, kill the other six action dims
    out = reverse_flow(x, _ones, NUM_STEPS, adapter=adapter)
    torch.testing.assert_close(out[..., 0], torch.full((1, CHUNK), 2.0))
    torch.testing.assert_close(out[..., 1:7], torch.zeros(1, CHUNK, 6))
    torch.testing.assert_close(out[..., 7:], torch.ones(1, CHUNK, MAX_DIM - 7))  # padding untouched
    torch.testing.assert_close(reverse_flow(x, _ones, NUM_STEPS, adapter=torch.eye(7)), torch.ones_like(x))


def test_task_slot_acts_in_both_directions_and_composes_before_the_user_module():
    n, eye = N_ACTION_DIMS, torch.eye(N_ACTION_DIMS)
    task = lambda x_t, t, v: (2.0 * eye.expand(v.shape[0], v.shape[1], n, n), torch.zeros_like(v))  # noqa: E731
    user = lambda x_t, t: 3.0 * eye.expand(x_t.shape[0], x_t.shape[1], n, n)  # noqa: E731
    adapter = ReversalAdapter(user_module=user, task_module=task)
    assert adapter.kind == "composed"
    x0 = torch.zeros(1, CHUNK, MAX_DIM)
    back = reverse_flow(x0, _ones, NUM_STEPS, adapter=adapter)  # backward: user(task(v)) = 6 v
    torch.testing.assert_close(back[..., :n], torch.full((1, CHUNK, n), 6.0))
    torch.testing.assert_close(back[..., n:], torch.full((1, CHUNK, MAX_DIM - n), 1.0))
    fwd = forward_flow(x0, _ones, NUM_STEPS, adapter=adapter)  # forward: task(v) = 2 v
    torch.testing.assert_close(fwd[..., :n], torch.full((1, CHUNK, n), -2.0))
    torch.testing.assert_close(fwd[..., n:], torch.full((1, CHUNK, MAX_DIM - n), -1.0))


# ---------------------------------------------------------------- the wrappers' inputs and knobs


def test_get_action_mean_std_follows_the_normalization_mode():
    mean, std = get_action_mean_std(postprocessor(mean=np.arange(7), std=np.zeros(7)))
    np.testing.assert_allclose(mean, np.arange(7))
    assert np.all(std >= 1e-6)
    with pytest.raises(RuntimeError):
        get_action_mean_std(SimpleNamespace(steps=[]))
    q01, q99 = np.full(7, -0.02), np.full(7, 0.06)
    step = SimpleNamespace(stats={"action": {"mean": np.ones(7), "std": np.ones(7), "q01": q01, "q99": q99}})
    step.stats["action"].update(min=np.zeros(7), max=np.full(7, 4.0))
    # quantiles: 2 (x - q01) / (q99 - q01) - 1 == (x - mean) / std
    for mode, expect in (("QUANTILES", (0.02, 0.04)), ("MEAN_STD", (1.0, 1.0)), ("MIN_MAX", (2.0, 2.0))):
        step.norm_map = {"ACTION": mode}
        got = get_action_mean_std(SimpleNamespace(steps=[step]))
        np.testing.assert_allclose(got, np.outer(expect, np.ones(7)))


def test_wrappers_take_the_deadband_in_the_source_units():
    small = (0.01, 0.0, 0.0)  # a full push in a 0.01-per-unit action space; idle at DEADBAND (0.05)
    for deadband, active in ((0.001, True), (DEADBAND, False)):
        fc = _fc(translation=small, lam=3, deadband=deadband)
        fc.select_action({})
        assert fc.guided_steps == (3 if active else 0)
        frs = _frs(translation=small, deadband=deadband)
        frs.select_action({})
        assert frs.steered_chunks == int(active)


def test_the_wrappers_validate_their_knobs():
    for kw, match in (
        ({"tau": CHUNK + 1}, "tau"),
        ({"tau": 0}, "tau"),
        ({"lam": NUM_STEPS + 1}, "lam"),
        ({"ic_gain": 0.0}, "ic_gain"),
        ({"ic_gain": "noise"}, "ic_gain"),
    ):
        with pytest.raises(ValueError, match=match):
            _fc(**kw)
    wrapper = _frs()
    for bad in (0, 11, 2.5, "3"):
        with pytest.raises(ValueError):
            wrapper.n_reversal_steps = bad
    wrapper.n_reversal_steps = NUM_STEPS
    assert wrapper.n_reversal_steps is None  # N means full reversal


def test_the_delegation_mask_follows_the_operator_dims():
    mask = delegation_mask(CHUNK, MAX_DIM, ("translation",), pinned_steps=2)
    assert mask.shape == (CHUNK, MAX_DIM)
    assert not mask[:2, :3].any()  # translation, first two steps: the operator's
    assert mask[:2, 3:].all() and mask[2:, :].all()  # the rest and the chunk's tail: the policy's
    full = delegation_mask(CHUNK, MAX_DIM, ("translation", "rotation", "gripper"), pinned_steps=None)
    assert not full[:, :N_ACTION_DIMS].any() and full[:, N_ACTION_DIMS:].all()
    with pytest.raises(ValueError, match="operator_dims"):
        delegation_mask(CHUNK, MAX_DIM, ("translation", "wrist"), pinned_steps=2)
    wrapper = _frs()
    assert wrapper.operator_dims == ("translation",) and wrapper._delegated[0, 3].item() is True
    wrapper.operator_dims = ["translation", "rotation"]  # the setter rebuilds the mask
    assert wrapper._delegated[0, 3].item() is False
    with pytest.raises(ValueError, match="operator_dims"):
        wrapper.operator_dims = ["translation", "wrist"]


def test_reference_chunk_uses_last_gripper_and_normalization():
    post = postprocessor(mean=[0.0] * 6 + [0.5], std=[2.0] * 7)
    ref = _frs(post=post).reference_chunk(np.array([1.0, 0.0, 0.0]))
    assert ref.shape == (1, CHUNK, MAX_DIM) and ref[0, 0, 0].item() == pytest.approx(0.5)  # (1 - 0) / 2
    assert ref[0, 0, GRIPPER_DIM].item() == pytest.approx((-1.0 - 0.5) / 2.0)  # open at start
    wrapper = _frs(post=post, gripper_open=0.0)
    ref = wrapper.reference_chunk(np.array([1.0, 0.0, 0.0]))
    assert ref[0, 0, GRIPPER_DIM].item() == pytest.approx(-0.25)
    wrapper.reset()
    assert wrapper._gripper_ref == pytest.approx((0.0 - 0.5) / 2.0)  # a 0/1 gripper's open state


# ---------------------------------------------------------------- flow control


def test_flow_control_writes_the_command_into_tau_chunk_steps():
    # Kao et al.: ic_gain * command (noise-prior units, no mean/std) into the first `tau` chunk steps of x_t at the
    # start of each of the first `lam` Euler steps; the rest of the chunk and the other dims are untouched.
    policy, post = FakePolicy(), postprocessor(mean=[0.1] * 7, std=[0.5] * 7)
    wrapper = _fc(policy, (0.5, 0.0, -0.5), post, tau=2, lam=3, ic_gain=2.0)
    wrapper.reset()
    assert wrapper.select_action({}).shape == (1, MAX_DIM) and policy.resets == 1
    assert wrapper.guided_steps == 3  # one write per Euler step, lam of them
    written = policy.last_chunk  # the chunk as the hooks left it
    np.testing.assert_allclose(written[0, :2, :3].numpy(), np.array([[1.0, 0.0, -1.0]] * 2), atol=1e-6)
    torch.testing.assert_close(written[0, 2:, :], torch.zeros(CHUNK - 2, MAX_DIM))
    torch.testing.assert_close(written[0, :2, 3:], torch.zeros(2, MAX_DIM - 3))
    # the lab's recorded arm, the defaults: the whole chunk, lam 1, the command in action units ((c - mean) / std)
    wrapper = _fc(policy, (0.5, 0.0, -0.5), post)
    assert wrapper.tau == CHUNK and wrapper.lam == 1 and wrapper.ic_gain == "action"
    wrapper.select_action({})
    expected = (np.array([0.5, 0.0, -0.5]) - 0.1) / 0.5
    assert wrapper.guided_steps == 1
    np.testing.assert_allclose(policy.last_chunk[0, :, :3].numpy(), np.tile(expected, (CHUNK, 1)), atol=1e-6)


def test_flow_control_runs_the_policy_alone_when_idle_or_at_lambda_zero():
    policy = FakePolicy()
    wrapper = _fc(policy, lam=0)
    wrapper.select_action({})
    assert wrapper.guided_steps == 0 and len(policy.calls) == 1  # no paired pass
    assert "x_t_hook" not in policy.calls[0]
    policy = FakePolicy()
    wrapper = _fc(policy, (DEADBAND / 2, 0, 0), lam=5)
    first, second = wrapper.select_action({}), wrapper.select_action({})
    assert wrapper.guided_steps == 0 and len(policy.calls) == 1  # n_action_steps=2 actions per chunk
    torch.testing.assert_close(first, torch.zeros(1, MAX_DIM))
    torch.testing.assert_close(second, torch.zeros(1, MAX_DIM))

    class ChangingSource:  # one noisy command is read per chunk
        reads = 0

        @property
        def translation(self):
            self.reads += 1
            return np.array([self.reads / 10, 0.0, 0.0])

    source = ChangingSource()
    wrapper = FlowControlPolicy(FakePolicy(), source, postprocessor(), lam=NUM_STEPS)
    first, second = wrapper.select_action({}), wrapper.select_action({})
    assert source.reads == 1 and first[0, 0].item() == pytest.approx(0.1)
    torch.testing.assert_close(first, second)
    wrapper.select_action({})
    assert source.reads == 2


def test_flow_control_records_the_policy_plan_from_the_same_noise_without_disturbing_the_stream():
    wrapper = _fc(NoisyPolicy(), lam=2)
    torch.manual_seed(3)
    steered = wrapper.select_action({})
    plan = wrapper.last_policy_action
    after = torch.randn(3)  # whatever the run draws next
    torch.manual_seed(3)
    torch.testing.assert_close(
        plan, NoisyPolicy().predict_action_chunk({})[:, 0]
    )  # the steered pass's sample
    assert not torch.equal(steered[0, :3], plan[0, :3]) and torch.equal(steered[0, 3:], plan[0, 3:])
    # A single steered pass (no second one) leaves the generator exactly where the wrapper did.
    torch.manual_seed(3)
    alone = _fc(NoisyPolicy(), lam=2)
    alone._pending_command = np.array([1.0, 0.0, 0.0])
    NoisyPolicy().predict_action_chunk({}, x_t_hook=alone._x_t_hook)
    torch.testing.assert_close(torch.randn(3), after)
    idle = _fc(NoisyPolicy(), (0.0, 0.0, 0.0), lam=2)
    action = idle.select_action({})
    assert idle.last_policy_action is action and len(idle._policy.calls) == 1  # not steering: no second pass


# ---------------------------------------------------------------- flow reversal steering


def test_reverse_flow_steering_passes_a_schedule_only_for_a_partial_reversal():
    policy, source = FakePolicy(), Source()
    wrapper = FlowReversalSteeringPolicy(policy, source, postprocessor(), n_reversal_steps=4)
    wrapper.select_action({})
    assert "flow_start_time" not in policy.calls[-1] and wrapper.steered_chunks == 0  # idle
    source.translation = np.array([1.0, 0.0, 0.0])
    wrapper.select_action({})
    wrapper.select_action({})  # served from the queue, no new call
    steered, plain = policy.calls[1:]  # a steered chunk costs a second, plain pass for the policy's own plan
    assert len(policy.calls) == 3 and plain == {}
    assert steered["flow_start_time"] == pytest.approx(0.4) and steered["num_forward_steps"] == 4
    assert wrapper.steered_chunks == 1 and len(wrapper.reconstruction_errors) == 1
    full = _frs(policy)
    full.select_action({})
    assert full.n_reversal_steps is None and "noise_fn" in policy.calls[-2]
    assert "flow_start_time" not in policy.calls[-2]  # the full reversal: no schedule


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

            x = torch.zeros(1, CHUNK, MAX_DIM)
            if kwargs.get("noise_fn") is not None:
                x = kwargs["noise_fn"](velocity, x)
            start, steps = kwargs.get("flow_start_time", 1.0), kwargs.get("num_forward_steps", NUM_STEPS)
            self.forward_dt = dt = -start / steps
            for i in range(steps):
                x = x + dt * velocity(x, start + i * dt)
            return x

    policy = IntegratingPolicy()
    wrapper = _frs(policy, (0.5, 0.0, -0.5), n_reversal_steps=depth)
    actual = wrapper.select_action({})
    # Exact constant-field inversion preserves the commanded dimensions, while
    # delegated ones complete the ordinary noise -> action trajectory.
    torch.testing.assert_close(actual[0, :3], torch.tensor([0.5, 0.0, -0.5]))
    torch.testing.assert_close(actual[0, 3:], torch.full((MAX_DIM - 3,), -1.0))
    assert policy.forward_dt == pytest.approx(-1.0 / NUM_STEPS)
    steered, plain = policy.call_times[-2:]
    assert len(steered) == NUM_STEPS + depth  # n reverse, N - n plain, n forward
    assert len(plain) == NUM_STEPS  # the policy's own plan
    torch.testing.assert_close(wrapper.last_policy_action, torch.full((1, MAX_DIM), -1.0))


def test_reverse_flow_steering_delegates_uncommanded_elements():
    # FakePolicy: reference reversed under v=1 gains +h per step; the initial noise is zeros.
    policy = FakePolicy()
    _frs(policy).select_action({})  # full reversal
    latent = policy.last_latent
    assert latent[0, 0, 0].item() == pytest.approx(1.0 + 1.0)  # reference + 10 * 0.1
    assert latent[0, 0, 3:].abs().max().item() == 0.0  # rotation, gripper, padding = fresh noise
    assert latent[0, 2:, :].abs().max().item() == 0.0  # tail beyond n_action_steps = fresh noise
    _frs(policy, n_reversal_steps=4).select_action({})
    assert policy.last_latent[0, 0, 0].item() == pytest.approx(1.0 + 0.4)
    assert policy.last_latent[0, 0, 3].item() == pytest.approx(-0.6)  # unsteered state at t = 0.4
    _frs(policy, operator_dims=("translation", "rotation", "gripper"), pinned_steps=None).select_action({})
    assert policy.last_latent[0, 3, 5].item() == pytest.approx(0.0 + 1.0)  # rotation at the last step: kept
    assert policy.last_latent[0, 3, N_ACTION_DIMS].item() == 0.0  # padding: always the policy's


def test_wrapper_converts_adapter_before_reversal():
    policy, std = FakePolicy(), np.array([2.0, 4.0, 6.0, 1.0, 1.0, 1.0, 1.0])
    matrix = np.eye(7)
    matrix[:3, :3] = rotation_about_z(90)
    _frs(
        policy, post=postprocessor(std=std), adapter=ReversalAdapter(matrix), n_reversal_steps=2
    ).select_action({})
    physical_delta = matrix @ std  # FakePolicy uses normalized v = ones.
    expected = np.array([1.0, 0.0, 0.0]) / std[:3] + 0.2 * physical_delta[:3] / std[:3]
    np.testing.assert_allclose(policy.last_latent[0, 0, :3].numpy(), expected, atol=1e-6)


def test_a_replaced_adapter_matrix_is_used_even_when_it_reuses_the_old_one_s_memory():
    """Per-trial adapters: clearing the adapter and setting a new matrix must not serve the old one."""
    policy, adapter = FakePolicy(), ReversalAdapter()
    wrapper = _frs(policy, adapter=adapter, n_reversal_steps=1)
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


def test_wrapper_passes_a_velocity_hook_only_with_a_task_module():
    policy = FakePolicy()
    task = lambda x_t, t, v: (torch.eye(7).expand(1, CHUNK, 7, 7), torch.zeros(1, CHUNK, 7))  # noqa: E731
    adapter = ReversalAdapter(task_module=task)
    _frs(policy, (0.0, 0.0, 0.0), adapter=adapter).select_action({})  # idle: the task module still acts
    assert "velocity_hook" in policy.calls[0] and "noise_fn" not in policy.calls[0]
    _frs(policy, adapter=adapter).select_action({})  # steering: both passes carry the hook, one the reversal
    steered, plain = policy.calls[1:]
    assert "velocity_hook" in steered and "noise_fn" in steered
    assert "velocity_hook" in plain and "noise_fn" not in plain
    _frs(policy, adapter=ReversalAdapter()).select_action({})
    assert "velocity_hook" not in policy.calls[-2] and "velocity_hook" not in policy.calls[-1]
