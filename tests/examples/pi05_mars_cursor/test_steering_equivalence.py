"""The cursor example's steered chunks and adapter losses, now on the lerobot steering core, against the
frozen cursor steering (`_reference_steering`) and the cursor trainer's original loss bodies.

The core's modules embed flow time with `sinusoidal_time_embedding` (10 ** x), the cursor's with
`time_embedding` (exp(x log 10)): the same frequencies up to float32 rounding. With the cursor's
embedding patched into the core (`exact`), the move is checked bitwise; as shipped, chunks without a
module are still bitwise, and the module cases agree to the rounding the perturbed modules amplify
(up to ~3e-5 measured, held to 1e-4 as in the Task 3 conversion test)."""

import _reference_steering as ref
import pytest
import torch
from cursor_sa import steering
from cursor_sa.policy import FlowPolicy, time_embedding
from train import train_adapter

import lerobot.policies.pi05.steering as core

B, K = 6, 3
# (mode, kwargs): the policy alone and the shared modes; teleop never builds a chunk
MODES = [
    ("policy", {}),
    ("flow_control", {"lam": 1}),
    ("flow_control", {"lam": 3}),
    ("frs", {"n_reverse_steps": None}),
    ("frs", {"n_reverse_steps": 4}),
]
ADAPTERS = ("none", "per_trial_F", "user", "task", "user_task")


# ---------------------------------------------------------------- the cursor trainer's losses at 7a3fddb6


def _ref_task_loss(policy, task, obs, chunk, z, t, lambda_id):
    a = chunk / policy.action_scale
    x = (1 - t[:, None, None]) * z + t[:, None, None] * a
    with torch.no_grad():
        v = policy(obs, x, t)
    A, b = task(obs, x, t, v)  # noqa: N806
    fm = ((torch.einsum("bsij,bsj->bsi", A, v) + b - (a - z)) ** 2).mean()
    identity = ((A - torch.eye(2, device=x.device)) ** 2).mean() + (b**2).mean()
    return fm + lambda_id * identity, {"fm": fm.item(), "identity": identity.item()}


def _ref_field_loss(policy, user, task, free, latent, bci, weight=None):
    t = torch.rand(len(latent), device=latent.device)
    x = (1 - t[:, None, None]) * latent + t[:, None, None] * bci
    with torch.no_grad():
        v = policy(free, x, t)
    if task is not None:
        A, b = task(free, x, t, v)  # noqa: N806
        v = torch.einsum("bsij,bsj->bsi", A, v) + b
    err = ((torch.einsum("bsij,bsj->bsi", user(x, t), v) - (bci - latent)) ** 2).mean((1, 2))
    return err.mean() if weight is None else (err * weight).sum() / weight.sum().clamp_min(1.0)


# ---------------------------------------------------------------- fixtures


def _obs():
    """Six trials with up to three obstacles each: trial 0 has none (only the null token is attended),
    trial 1 one, the rest all three."""
    g = torch.Generator().manual_seed(0)
    n = 5
    goal = torch.zeros(B, n, dtype=torch.bool)
    goal[torch.arange(B), torch.randint(n, (B,), generator=g)] = True
    obstacles = torch.cat(
        [torch.rand(B, K, 2, generator=g), 0.03 + 0.05 * torch.rand(B, K, 1, generator=g)], -1
    )
    obstacle_mask = torch.ones(B, K, dtype=torch.bool)
    obstacle_mask[0] = False
    obstacle_mask[1, 1:] = False
    return {
        "pos": torch.rand(B, 2, generator=g),
        "hist": 0.01 * torch.randn(B, 4, 2, generator=g),
        "targets": torch.rand(B, n, 2, generator=g),
        "mask": torch.ones(B, n, dtype=torch.bool),
        "goal": goal,
        "obstacles": obstacles,
        "obstacle_mask": obstacle_mask,
    }


@pytest.fixture(params=[False, True], ids=["shipped", "exact"])
def exact(request, monkeypatch):
    """With True, the core's modules embed time with the cursor's `time_embedding`."""
    if request.param:

        def cursor_embedding(t, dim, device=None, dtype=None):
            return time_embedding(torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1), dim)

        monkeypatch.setattr(core, "sinusoidal_time_embedding", cursor_embedding)
    return request.param


def _close(new, old, exact, modules=True):
    """Bitwise when exact (or when no module is in the field), else to the embedding's rounding."""
    if exact or not modules:
        assert torch.equal(new, old), (new - old).abs().max()
    else:
        torch.testing.assert_close(new, old, atol=1e-4, rtol=1e-4)


def _perturb(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return module.eval()


def _twin(old, new_cls):
    """A module of the example's class with the reference module's weights."""
    new = new_cls(**old.config)
    new.load_state_dict(old.state_dict())
    return new.eval()


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    policy = FlowPolicy().eval()
    old_user, old_task = (
        _perturb(ref.UserModule(policy.horizon), 1),
        _perturb(ref.TaskModule(policy.horizon), 2),
    )
    new_user, new_task = _twin(old_user, steering.UserModule), _twin(old_task, steering.TaskModule)
    f = torch.eye(2) + 0.3 * torch.randn(B, 2, 2, generator=torch.Generator().manual_seed(3))
    adapters = {
        "none": (None, None),
        "per_trial_F": (f, f),
        "user": (ref.ReversalAdapter(user=old_user), steering.ReversalAdapter(user=new_user)),
        "task": (ref.ReversalAdapter(task=old_task), steering.ReversalAdapter(task=new_task)),
        "user_task": (
            ref.ReversalAdapter(user=old_user, task=old_task),
            steering.ReversalAdapter(user=new_user, task=new_task),
        ),
    }
    # two of the six commands inside the deadband: their rows get the plain policy chunk
    command = torch.tensor([[0.8, 0.1], [0.01, -0.02], [-0.5, 0.6], [0.0, 0.04], [0.3, -0.9], [-1.0, 0.0]])
    return {
        "policy": policy,
        "obs": _obs(),
        "command": command,
        "adapters": adapters,
        "modules": (old_user, old_task, new_user, new_task),
    }


def _chunks(s, mode, kwargs, adapter, reference=None):
    old_adapter, new_adapter = s["adapters"][adapter]
    out = []
    for fn, a in ((ref.steered_chunk, old_adapter), (steering.steered_chunk, new_adapter)):
        torch.manual_seed(0)
        out.append(
            fn(s["policy"], s["obs"], s["command"], mode, tau=4, adapter=a, reference=reference, **kwargs)
        )
    return out


# ---------------------------------------------------------------- steered chunks


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    "mode,kwargs", MODES, ids=[f"{m}-{v}" for m, kw in MODES for v in kw.values() or [""]]
)
def test_steered_chunk_matches_the_reference(setup, exact, mode, kwargs, adapter):
    old, new = _chunks(setup, mode, kwargs, adapter)
    assert old.shape == (B, setup["policy"].horizon, 2)
    _close(new, old, exact, modules=adapter not in ("none", "per_trial_F"))


@pytest.mark.parametrize("adapter", ("none", "user_task"))
@pytest.mark.parametrize("mode", ("flow_control", "frs"))
def test_steered_chunk_with_a_reference_chunk_matches(setup, exact, mode, adapter):
    g = torch.Generator().manual_seed(4)
    reference = torch.rand(B, setup["policy"].horizon, 2, generator=g) * 2 - 1
    reference[1] = 0.01  # a whole reference inside the deadband: the plain chunk
    old, new = _chunks(setup, mode, {}, adapter, reference)
    _close(new, old, exact, modules=adapter != "none")


def test_all_idle_commands_give_the_plain_chunk(setup, exact):
    s = {**setup, "command": torch.full((B, 2), 0.01)}
    old, new = _chunks(s, "frs", {}, "user_task")
    _close(new, old, exact)


# ---------------------------------------------------------------- the adapter losses


def _loss_close(new, old, exact):
    """Bitwise when exact; as shipped, the losses (tens, with the perturbed modules) to float32 rounding."""
    if exact:
        assert torch.equal(new, old), (new - old).abs()
    else:
        torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)


def _loss_inputs(s):
    g = torch.Generator().manual_seed(5)
    h = s["policy"].horizon
    latent, bci = torch.randn(B, h, 2, generator=g), torch.randn(B, h, 2, generator=g)
    chunk, z = 0.01 * torch.randn(B, h, 2, generator=g), torch.randn(B, h, 2, generator=g)
    weight = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    return latent, bci, chunk, z, weight


@pytest.mark.parametrize("with_task", (False, True))
@pytest.mark.parametrize("weighted", (False, True))
def test_field_loss_matches_the_reference(setup, exact, with_task, weighted):
    s = setup
    old_user, old_task, new_user, new_task = s["modules"]
    latent, bci, _, _, weight = _loss_inputs(s)
    w = weight if weighted else None
    torch.manual_seed(0)
    expect = _ref_field_loss(s["policy"], old_user, old_task if with_task else None, s["obs"], latent, bci, w)
    torch.manual_seed(0)
    got = train_adapter.field_loss(
        s["policy"], new_user, new_task if with_task else None, s["obs"], latent, bci, w
    )
    _loss_close(got, expect, exact)


@pytest.mark.parametrize("with_task", (False, True))
def test_core_user_field_loss_matches_the_reference(setup, exact, with_task):
    from lerobot.policies.pi05.steering import user_field_loss

    s = setup
    old_user, old_task, new_user, new_task = s["modules"]
    latent, bci, _, _, weight = _loss_inputs(s)
    torch.manual_seed(0)
    expect = _ref_field_loss(
        s["policy"], old_user, old_task if with_task else None, s["obs"], latent, bci, weight
    )
    torch.manual_seed(0)
    t = torch.rand(B)
    got = user_field_loss(
        new_user,
        lambda x, t: s["policy"](s["obs"], x, t),
        latent,
        bci,
        t,
        task=new_task.bind(s["obs"]) if with_task else None,
        weight=weight,
    )
    _loss_close(got, expect, exact)


def test_task_loss_matches_the_reference(setup, exact):
    s = setup
    _, old_task, _, new_task = s["modules"]
    _, _, chunk, z, _ = _loss_inputs(s)
    t = torch.rand(B, generator=torch.Generator().manual_seed(6))
    expect, expect_parts = _ref_task_loss(s["policy"], old_task, s["obs"], chunk, z, t, 0.01)
    got, parts = train_adapter.task_loss(s["policy"], new_task, s["obs"], chunk, z, t, 0.01)
    _loss_close(got, expect, exact)
    assert parts.keys() == expect_parts.keys()
    for key in parts:
        assert parts[key] == pytest.approx(expect_parts[key], rel=0 if exact else 1e-5)


def test_core_task_field_loss_matches_the_reference(setup, exact):
    from lerobot.policies.pi05.steering import task_field_loss

    s = setup
    _, old_task, _, new_task = s["modules"]
    _, _, chunk, z, _ = _loss_inputs(s)
    t = torch.rand(B, generator=torch.Generator().manual_seed(6))
    expect, expect_parts = _ref_task_loss(s["policy"], old_task, s["obs"], chunk, z, t, 0.01)
    got, parts = task_field_loss(
        new_task.bind(s["obs"]),
        lambda x, t: s["policy"](s["obs"], x, t),
        chunk / s["policy"].action_scale,
        z,
        t,
        0.01,
        n_dims=2,
    )
    _loss_close(got, expect, exact)
    for key in parts:
        assert parts[key] == pytest.approx(expect_parts[key], rel=0 if exact else 1e-5)


# ---------------------------------------------------------------- checkpoints


@pytest.mark.parametrize("kind", ("user", "task"))
def test_reversal_adapter_loads_cursor_converted_and_trainer_checkpoints(setup, tmp_path, kind):
    old_user, old_task, new_user, new_task = setup["modules"]
    old, new = (old_user, new_user) if kind == "user" else (old_task, new_task)
    cursor = {"config": old.config, "state_dict": old.state_dict(), "args": {"name": "m"}, "obstacle": None}
    if kind == "task":
        cursor["kind"] = "task"  # user modules from the first adapters carry no kind
    # what train_adapter.save writes now: the example's module config (with n_dims), no arch
    trainer = {
        "kind": kind,
        "config": new.config,
        "state_dict": new.state_dict(),
        "args": {},
        "obstacle": None,
    }
    paths = {}
    for name, ckpt in (
        ("cursor", cursor),
        ("converted", core.convert_cursor_module(cursor)),
        ("trainer", trainer),
    ):
        paths[name] = tmp_path / f"{name}.pt"
        torch.save(ckpt, paths[name])
    obs, x, v, t = setup["obs"], torch.randn(B, 16, 2), torch.randn(B, 16, 2), torch.rand(B)
    expect = new_user(x, t) if kind == "user" else new_task(x, t, v, obs)
    for name, path in paths.items():
        adapter = steering.ReversalAdapter.load(path)
        module = getattr(adapter, kind)
        assert adapter.kind == kind and isinstance(
            module, steering.UserModule if kind == "user" else steering.TaskModule
        ), name
        got = module(x, t) if kind == "user" else module(x, t, v, obs)
        for a, b in zip(
            got if kind == "task" else (got,), expect if kind == "task" else (expect,), strict=True
        ):
            assert torch.equal(a, b), name
