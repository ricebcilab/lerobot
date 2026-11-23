"""CPU tests for the shared-autonomy steering wrappers (no checkpoint needed)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.steering import (
    DEADBAND,
    GRIPPER_DIM,
    N_ACTION_DIMS,
    AdapterModules,
    Flow,
    FlowControlSteering,
    FlowReversalSteering,
    LearnedUserModule,
    SteeringConfig,
    sinusoidal_time_embedding,
)
from lerobot.teleoperators import readout

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
    return FlowReversalSteering(policy, Source(translation), post, **kw)


def _fc(policy=None, translation=(1.0, 0.0, 0.0), post=None, **kw):
    return FlowControlSteering(policy or FakePolicy(), Source(translation), post or postprocessor(), **kw)


# ---------------------------------------------------------------- matrices and adapters


def test_build_reversal_adapter_blocks_and_errors():
    m = readout.rotation(20)
    f = SteeringConfig.build_adapter(
        {"translation": "corruption", "orientation": "zero", "gripper": "zero"}, m
    )
    assert f.shape == (7, 7) and np.allclose(f[:3, :3], m) and not f[3:, :].any() and not f[:3, 3:].any()
    inv = SteeringConfig.build_adapter({"translation": "corruption_inverse"}, m)
    np.testing.assert_allclose(inv[:3, :3], np.linalg.inv(m))
    scaled = SteeringConfig.build_adapter({"translation": {"scale": [2, 2, 2]}})
    np.testing.assert_allclose(scaled[:3, :3], 2 * np.eye(3))
    np.testing.assert_allclose(scaled[3:, 3:], np.eye(4))
    assert SteeringConfig.build_adapter(None) is None
    np.testing.assert_allclose(SteeringConfig.build_adapter({"F": np.eye(7).tolist()}), np.eye(7))
    for spec, match in (
        ({"orientation": "half"}, "orientation"),
        ({"rotation": "zero"}, "unknown"),
        ({"F": np.eye(7).tolist(), "gripper": "zero"}, "F"),
    ):
        with pytest.raises(ValueError, match=match):
            SteeringConfig.build_adapter(spec)
    with pytest.raises(ValueError, match="corruption"):
        SteeringConfig.build_adapter({"translation": "corruption"}, None)


def test_adapter_normalization_preserves_environment_velocity_transform():
    # Unequal scales and cross-group mixing expose copying F directly into model space.
    std = np.array([0.3, 0.4, 0.5, 0.04, 0.06, 0.08, 1.0])
    matrix = np.eye(7)
    matrix[:3, :3] = readout.rotation(20)
    matrix[0, 3] = 0.2
    normalized_velocity = np.arange(1.0, 8.0)
    actual = std * (AdapterModules(n_dims=None, matrix=matrix).normalized_matrix(std) @ normalized_velocity)
    np.testing.assert_allclose(actual, matrix @ (std * normalized_velocity))
    np.testing.assert_allclose(
        AdapterModules(n_dims=None, matrix=np.eye(7)).normalized_matrix(std), np.eye(7)
    )
    assert AdapterModules(n_dims=None).normalized_matrix(std) is None


def test_parse_adapter_spec_forms():
    m = readout.rotation(20)
    assert SteeringConfig.from_spec(None, m) == SteeringConfig()
    fixed = SteeringConfig.from_spec({"translation": "corruption", "gripper": "zero"}, m)
    np.testing.assert_allclose(fixed.fixed_matrix[:3, :3], m)
    assert fixed.adapter_user is None and fixed.adapter_task is None
    assert SteeringConfig.from_spec({"learned": "u.pt"}, m) == SteeringConfig(adapter_user="u.pt")
    composed = SteeringConfig.from_spec({"user": {"translation": "corruption"}, "task": "t.pt"}, m)
    np.testing.assert_allclose(composed.fixed_matrix[:3, :3], m)
    assert composed.adapter_task == "t.pt" and composed.adapter_user is None
    assert SteeringConfig.from_spec({"user": {"learned": "u.pt"}, "task": None}, m) == SteeringConfig(
        adapter_user="u.pt"
    )
    assert SteeringConfig.from_spec({"user": None, "task": "t.pt"}, m) == SteeringConfig(adapter_task="t.pt")
    np.testing.assert_allclose(SteeringConfig.from_spec({"F": np.eye(7).tolist()}, m).fixed_matrix, np.eye(7))
    for bad in (
        {"user": {"learned": "u"}, "F": np.eye(7).tolist()},
        {"learned": 1},
        {"task": 3},
        {"model": "x"},
        {"user": {"learned": "u", "translation": "corruption"}},
    ):
        with pytest.raises(ValueError):
            SteeringConfig.from_spec(bad, m)


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
    path = AdapterModules.save_module(module, tmp_path / "user.pt", kind="user")
    assert AdapterModules.load_module(path)[0].config["time_embedding"] == name
    ckpt = torch.load(path, weights_only=True)  # checkpoints saved before the option keep the MARS embedding
    del ckpt["config"]["time_embedding"]
    torch.save(ckpt, tmp_path / "old.pt")
    assert AdapterModules.load_module(tmp_path / "old.pt")[0].time_embedding == "mars"
    with pytest.raises(ValueError, match="time_embedding"):
        LearnedUserModule(horizon=CHUNK, time_embedding="rope")


# ---------------------------------------------------------------- the flows


def test_flows_have_constant_field_closed_forms_and_touch_only_action_dims():
    x, seen = torch.zeros(1, CHUNK, MAX_DIM), []

    def velocity(x_t, t):
        seen.append(round(t, 6))
        return torch.ones_like(x_t)

    torch.testing.assert_close(Flow(velocity).reverse(x, NUM_STEPS), torch.ones_like(x))  # 10 of h=0.1, v=1
    assert seen == [round(k / NUM_STEPS, 6) for k in range(NUM_STEPS)]  # times 0 to 1
    torch.testing.assert_close(Flow(_ones).reverse(x, NUM_STEPS, 3), torch.full_like(x, 0.3))
    seen.clear()
    stopped = Flow(velocity).forward(x, NUM_STEPS, 4)  # t = 1 down to 0.4
    assert stopped[0, 0, 0].item() == pytest.approx(-0.6)
    assert seen == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    assert Flow(_ones).forward(x, NUM_STEPS)[0, 0, 0].item() == pytest.approx(-1.0)
    adapter = torch.zeros(7, 7)
    adapter[0, 0] = 2.0  # double dx, kill the other six action dims
    out = Flow(_ones, adapter).reverse(x, NUM_STEPS)
    torch.testing.assert_close(out[..., 0], torch.full((1, CHUNK), 2.0))
    torch.testing.assert_close(out[..., 1:7], torch.zeros(1, CHUNK, 6))
    torch.testing.assert_close(out[..., 7:], torch.ones(1, CHUNK, MAX_DIM - 7))  # padding untouched
    torch.testing.assert_close(Flow(_ones, torch.eye(7)).reverse(x, NUM_STEPS), torch.ones_like(x))


def test_task_slot_acts_in_both_directions_and_composes_before_the_user_module():
    n, eye = N_ACTION_DIMS, torch.eye(N_ACTION_DIMS)
    task = lambda x_t, t, v: (2.0 * eye.expand(v.shape[0], v.shape[1], n, n), torch.zeros_like(v))  # noqa: E731
    user = lambda x_t, t: 3.0 * eye.expand(x_t.shape[0], x_t.shape[1], n, n)  # noqa: E731
    adapter = AdapterModules(n_dims=None, user=user, task=task)
    assert adapter.kind == "composed"
    x0 = torch.zeros(1, CHUNK, MAX_DIM)
    back = Flow(_ones, adapter).reverse(x0, NUM_STEPS)  # backward: user(task(v)) = 6 v
    torch.testing.assert_close(back[..., :n], torch.full((1, CHUNK, n), 6.0))
    torch.testing.assert_close(back[..., n:], torch.full((1, CHUNK, MAX_DIM - n), 1.0))
    fwd = Flow(_ones, adapter).forward(x0, NUM_STEPS)  # forward: task(v) = 2 v
    torch.testing.assert_close(fwd[..., :n], torch.full((1, CHUNK, n), -2.0))
    torch.testing.assert_close(fwd[..., n:], torch.full((1, CHUNK, MAX_DIM - n), -1.0))


# ---------------------------------------------------------------- the wrappers' inputs and knobs


def test_get_action_mean_std_follows_the_normalization_mode():
    mean, std = FlowControlSteering.action_mean_std(postprocessor(mean=np.arange(7), std=np.zeros(7)))
    np.testing.assert_allclose(mean, np.arange(7))
    assert np.all(std >= 1e-6)
    with pytest.raises(RuntimeError):
        FlowControlSteering.action_mean_std(SimpleNamespace(steps=[]))
    q01, q99 = np.full(7, -0.02), np.full(7, 0.06)
    step = SimpleNamespace(stats={"action": {"mean": np.ones(7), "std": np.ones(7), "q01": q01, "q99": q99}})
    step.stats["action"].update(min=np.zeros(7), max=np.full(7, 4.0))
    # quantiles: 2 (x - q01) / (q99 - q01) - 1 == (x - mean) / std
    for mode, expect in (("QUANTILES", (0.02, 0.04)), ("MEAN_STD", (1.0, 1.0)), ("MIN_MAX", (2.0, 2.0))):
        step.norm_map = {"ACTION": mode}
        got = FlowControlSteering.action_mean_std(SimpleNamespace(steps=[step]))
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
    mask = FlowReversalSteering.delegation_mask(CHUNK, MAX_DIM, ("translation",), pinned_steps=2)
    assert mask.shape == (CHUNK, MAX_DIM)
    assert not mask[:2, :3].any()  # translation, first two steps: the operator's
    assert mask[:2, 3:].all() and mask[2:, :].all()  # the rest and the chunk's tail: the policy's
    full = FlowReversalSteering.delegation_mask(
        CHUNK, MAX_DIM, ("translation", "rotation", "gripper"), pinned_steps=None
    )
    assert not full[:, :N_ACTION_DIMS].any() and full[:, N_ACTION_DIMS:].all()
    with pytest.raises(ValueError, match="operator_dims"):
        FlowReversalSteering.delegation_mask(CHUNK, MAX_DIM, ("translation", "wrist"), pinned_steps=2)
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


def test_a_fixed_gripper_reference_is_held_at_every_chunk():
    """`gripper_reference` (Tang et al. on LIBERO, App. D.2: the gripper halfway between open and closed): every
    reference holds that environment value, where the default moves on to the model's last executed command."""
    post = postprocessor(mean=[0.0] * 6 + [0.5], std=[2.0] * 7)
    opened, halfway = (-1.0 - 0.5) / 2.0, (0.0 - 0.5) / 2.0

    def gripper_after_a_chunk(**kw):
        wrapper = _frs(post=post, **kw)
        wrapper.select_action({})  # a steered chunk: the default takes the model's gripper command from it
        return wrapper.reference_chunk(np.array([1.0, 0.0, 0.0]))[0, 0, GRIPPER_DIM].item(), wrapper

    assert gripper_after_a_chunk()[0] != pytest.approx(opened)
    value, wrapper = gripper_after_a_chunk(gripper_reference=0.0)
    assert value == pytest.approx(halfway)
    wrapper.reset()
    assert wrapper._gripper_ref == pytest.approx(halfway)
    with pytest.raises(ValueError, match="gripper_reference"):  # an action space without the gripper dim
        FlowReversalSteering(FakePolicy(), Source(), None, reference_fn=lambda c: c, gripper_reference=0.0)


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


def test_flow_control_zeroes_the_named_dims_in_the_steered_steps():
    # Kao et al. on LIBERO (App. G.1): the dims the command does not set, there rotation, are 0 in the steered steps;
    # the gripper, the padding, the later steps and an idle row keep their noise.
    x_t = torch.ones(2, CHUNK, MAX_DIM)
    steer = torch.tensor([True, False])[:, None, None]
    out = FlowControlSteering.write(
        x_t, 0, torch.full((2, 1, 3), 1.5), tau=2, lam=1, dims=slice(0, 3), steer=steer, zeroed=(slice(3, 6),)
    )
    assert (out[0, :2, :3] == 1.5).all() and (out[0, :2, 3:6] == 0).all()
    assert (out[0, :2, 6:] == 1).all() and (out[0, 2:] == 1).all() and (out[1] == 1).all()
    core = _fc(
        FakePolicy(), post=postprocessor(mean=[0.0] * 7, std=[1.0] * 7), ic_gain=1.5, zeroed=(slice(3, 6),)
    )
    assert core.zeroed == (slice(3, 6),) and _fc(FakePolicy(), ic_gain=1.5).zeroed == ()


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
    wrapper = FlowControlSteering(FakePolicy(), source, postprocessor(), lam=NUM_STEPS)
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
    wrapper = FlowReversalSteering(policy, source, postprocessor(), n_reversal_steps=4)
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


@pytest.mark.parametrize("depth", [2, 10])
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
    matrix[:3, :3] = readout.rotation(90)
    _frs(
        policy,
        post=postprocessor(std=std),
        adapter=AdapterModules(n_dims=None, matrix=matrix),
        n_reversal_steps=2,
    ).select_action({})
    physical_delta = matrix @ std  # FakePolicy uses normalized v = ones.
    expected = np.array([1.0, 0.0, 0.0]) / std[:3] + 0.2 * physical_delta[:3] / std[:3]
    np.testing.assert_allclose(policy.last_latent[0, 0, :3].numpy(), expected, atol=1e-6)


def test_a_replaced_adapter_matrix_is_used_even_when_it_reuses_the_old_one_s_memory():
    """Per-trial adapters: clearing the adapter and setting a new matrix must not serve the old one."""
    policy, adapter = FakePolicy(), AdapterModules(n_dims=None)
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
    adapter = AdapterModules(n_dims=None, task=task)
    _frs(policy, (0.0, 0.0, 0.0), adapter=adapter).select_action({})  # idle: the task module still acts
    assert "velocity_hook" in policy.calls[0] and "noise_fn" not in policy.calls[0]
    _frs(policy, adapter=adapter).select_action({})  # steering: both passes carry the hook, one the reversal
    steered, plain = policy.calls[1:]
    assert "velocity_hook" in steered and "noise_fn" in steered
    assert "velocity_hook" in plain and "noise_fn" not in plain
    _frs(policy, adapter=AdapterModules(n_dims=None)).select_action({})
    assert "velocity_hook" not in policy.calls[-2] and "velocity_hook" not in policy.calls[-1]
