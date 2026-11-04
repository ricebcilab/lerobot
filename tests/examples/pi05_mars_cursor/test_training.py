"""The adapter trainer: user modules from expert labels, task modules with a user module in the loop."""

import json

import numpy as np
import pytest
import torch

from cursor_sa.readout import corruption_spec
from cursor_sa.env import TaskConfig, obstacle_spec, sample_layouts
from cursor_sa.demos import DemoSampler, make_demos
from cursor_sa.policy import FlowPolicy
from cursor_sa.steering import ReversalAdapter, TaskModule, UserModule
from train import train_adapter

COMMON = ["--n-demos", "60", "--val-demos", "40", "--steps", "4", "--log-every", "2", "--batch", "32"]


@pytest.fixture
def ckpt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(train_adapter, "adapter_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(train_adapter, "adapter_path", lambda name: tmp_path / name / "adapter.pt")
    return tmp_path


def test_task_loss_matches_the_formula():
    policy, task = FlowPolicy().eval(), TaskModule()
    torch.nn.init.normal_(task.out.weight, std=0.1)
    tcfg = TaskConfig(min_reach=0.3, max_reach=0.6, obstacle=obstacle_spec({"kind": "disc", "radius": 0.04, "along": 0.5}))
    rng = np.random.default_rng(0)
    layouts = sample_layouts(rng, 6, 3, tcfg)
    sampler = DemoSampler(layouts, *make_demos(rng, layouts, tcfg), policy.horizon, policy.hist, "cpu")
    obs, chunk = sampler.sample(5)
    z, t = torch.randn(5, policy.horizon, 2), torch.rand(5)
    loss, parts = train_adapter.task_loss(policy, task, obs, chunk, z, t, lambda_id=0.01)
    a = chunk / policy.action_scale
    x = (1 - t[:, None, None]) * z + t[:, None, None] * a
    with torch.no_grad():
        v = policy(obs, x, t)
        A, b = task(x, t, v, obs)
        fm = ((torch.einsum("bsij,bsj->bsi", A, v) + b - (a - z)) ** 2).mean()
        identity = ((A - torch.eye(2)) ** 2).mean() + (b ** 2).mean()
    torch.testing.assert_close(loss, fm + 0.01 * identity)
    assert parts["fm"] == pytest.approx(fm.item()) and parts["identity"] == pytest.approx(identity.item())


def test_field_loss_matches_the_formula_and_trains_only_the_free_module():
    policy, user, task = FlowPolicy().eval().requires_grad_(False), UserModule(), TaskModule()
    torch.nn.init.normal_(user.net[-1].weight, std=0.05)
    torch.nn.init.normal_(task.out.weight, std=0.05)
    user.requires_grad_(False)
    B, H = 4, policy.horizon
    g = torch.Generator().manual_seed(0)
    free = dict(pos=torch.rand(B, 2, generator=g), hist=torch.zeros(B, policy.hist, 2), targets=torch.rand(B, 3, 2, generator=g),
                mask=torch.ones(B, 3, dtype=torch.bool), goal=torch.zeros(B, 3, dtype=torch.bool),
                obstacles=torch.rand(B, 2, 3, generator=g) * 0.1, obstacle_mask=torch.ones(B, 2, dtype=torch.bool))
    latent, bci = torch.randn(B, H, 2, generator=g), torch.randn(B, H, 2, generator=g)
    weight = torch.tensor([1.0, 0.0, 1.0, 1.0])
    torch.manual_seed(3)
    loss = train_adapter.field_loss(policy, user, task, free, latent, bci, weight)
    torch.manual_seed(3)
    t = torch.rand(B)
    x = (1 - t[:, None, None]) * latent + t[:, None, None] * bci
    with torch.no_grad():
        A, b = task(x, t, policy(free, x, t), free)
        v = torch.einsum("bsij,bsj->bsi", A, policy(free, x, t)) + b
        err = ((torch.einsum("bsij,bsj->bsi", user(x, t), v) - (bci - latent)) ** 2).mean((1, 2))
    torch.testing.assert_close(loss, (err * weight).sum() / 3)
    loss.backward()
    assert task.out.weight.grad.abs().sum() > 0 and user.net[-1].weight.grad is None


def test_train_a_user_module_then_a_task_module_with_it(ckpt_dir):
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{curl: 20, tau: 0.1}", "--name", "u",
                                                  *COMMON]))
    ckpt = torch.load(ckpt_dir / "u" / "adapter.pt")
    assert ckpt["kind"] == "user" and ckpt["meta"]["obstacle"] is None and ckpt["meta"]["args"]["corruption"] == {"curl": 20, "tau": 0.1}
    assert ReversalAdapter.load(ckpt_dir / "u" / "adapter.pt").kind == "user"
    log = json.loads((ckpt_dir / "u" / "train_log.json").read_text())["log"]
    assert len(log) == 2 and {"heading_cos_none", "heading_cos_user"} <= set(log[-1])
    # recalibrated closed loop (ReFIT-style): the initial module runs the open-field trials, their states join the batches
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{curl: 20, tau: 0.1}", "--name", "u_refit",
                                                  "--closed-loop", "u", "--closed-loop-trials", "4", "--closed-loop-depth", "2",
                                                  "--closed-loop-time-limit", "1.0", *COMMON]))
    ckpt = torch.load(ckpt_dir / "u_refit" / "adapter.pt")
    assert ckpt["meta"]["args"]["closed_loop"] == ["u"] and ckpt["meta"]["args"]["closed_loop_mix"] == 0.5
    log = json.loads((ckpt_dir / "u_refit" / "train_log.json").read_text())["log"]
    assert {"heading_cos_user", "closed_heading_cos_none", "closed_heading_cos_user"} <= set(log[-1])
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: ring, around: start, gap_angle: [15, 30]}",
                                                  "--name", "t", "--user-module", "u", "--corruption", "{curl: 20, tau: 0.1}",
                                                  "--lambda-field", "0.1", *COMMON]))
    ckpt = torch.load(ckpt_dir / "t" / "adapter.pt")
    assert ckpt["kind"] == "task" and ckpt["meta"]["obstacle"]["kind"] == "ring" and ckpt["meta"]["args"]["user_module"] == "u"
    loaded = ReversalAdapter.load(ckpt_dir / "t" / "adapter.pt")
    assert loaded.kind == "task" and isinstance(loaded.task, TaskModule)
    log = json.loads((ckpt_dir / "t" / "train_log.json").read_text())["log"]
    assert {"val_fm", "chunk_collision_hidden", "chunk_collision_policy_given", "collide_user", "collide_composed"} <= set(log[-1])
    for bad in (["--kind", "task", "--obstacle", "{kind: disc, radius: 0.04, along: 0.5}", "--corruption", "{rotation_deg: 60}"],
                ["--kind", "user", "--obstacle", "{kind: disc, radius: 0.04, along: 0.5}", "--corruption", "{rotation_deg: 60}"]):
        with pytest.raises(SystemExit):
            train_adapter.parse_args([*bad, "--name", "x"])


def test_retrain_the_policy_for_a_task_then_calibrate_a_user_module_on_it(ckpt_dir, monkeypatch):
    """The grid's retrained policy: an obstacle head, trained for one obstacle task (a share of its layouts without
    the obstacle, as a task module) from scratch, or fine-tuned from a trained policy (--init); and a user module
    calibrated on it (--model) as the others are on the frozen policy."""
    from train import train_policy
    monkeypatch.setattr(train_policy, "policy_dir", lambda name: ckpt_dir / name)
    monkeypatch.setattr(train_policy, "policy_path", lambda name: ckpt_dir / name / "policy.pt")
    monkeypatch.setattr(train_adapter, "policy_path", lambda name: ckpt_dir / name / "policy.pt")
    tiny = ["--n_demos", "30", "--val_demos", "10", "--steps", "4", "--batch", "16", "--log_every", "2", "--warmup", "2",
            "--goal_dropout", "0.25", "--targets", "4", "4", "--reach", "0.35", "0.5"]
    ring = ["--obstacle", "{kind: ring, around: goal, gap_angle: [15, 30]}"]
    train_policy.main(["--name", "scratch", *ring, *tiny])
    scratch = torch.load(ckpt_dir / "scratch" / "policy.pt")
    assert scratch["config"]["obstacle_tokens"] is True and scratch["args"]["obstacle"]["kind"] == "ring"
    assert scratch["args"]["p_free"] == 0.25 and scratch["args"]["init"] is None
    log = json.loads((ckpt_dir / "scratch" / "train_log.json").read_text())
    assert len(log["log"]) == 2 and np.isfinite(log["log"][-1]["val"])
    train_policy.main(["--name", "base", "--d_model", "32", "--layers", "1", "--heads", "2", *tiny])
    train_policy.main(["--name", "tuned", "--init", "base", *ring, *tiny])
    base, tuned = torch.load(ckpt_dir / "base" / "policy.pt"), torch.load(ckpt_dir / "tuned" / "policy.pt")
    assert tuned["config"] == {**base["config"], "obstacle_tokens": True}  # the base's architecture, an obstacle head added
    torch.testing.assert_close(tuned["state_dict"]["action_scale"], base["state_dict"]["action_scale"])  # the base's units
    assert json.loads((ckpt_dir / "tuned" / "train_log.json").read_text())["action_scale"] == pytest.approx(
        base["state_dict"]["action_scale"].item())
    assert hasattr(FlowPolicy.load(ckpt_dir / "tuned" / "policy.pt"), "obstacle_in")
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--model", "scratch", "--corruption", "{rotation_deg: 60}",
                                                  "--name", "u_scratch", *COMMON]))
    assert torch.load(ckpt_dir / "u_scratch" / "adapter.pt")["meta"]["args"]["model"] == "scratch"


def test_the_experts_pushes_and_the_readout_core_line_up_with_the_demonstrations():
    policy = FlowPolicy().eval()
    args = train_adapter.parse_args(["--name", "x", "--corruption", "{scale: [1.0, 0.25]}", "--n-demos", "40",
                                     "--val-demos", "10"])
    data = train_adapter.expert_splits(args, None, policy, "cpu", np.random.default_rng(0), *corruption_spec(args.corruption))["train"]
    obs, chunk, push, core, i, k = data.sample(200)
    moving = push.norm(dim=-1) > 0.99  # full deflection: far from the goal
    assert moving.any()
    # the core is the corruption of the push, M @ push, and the push points at the goal
    torch.testing.assert_close(core, torch.clamp(push * torch.tensor([1.0, 0.25]), -1, 1))
    goal = obs["targets"][torch.arange(len(i)), data.sampler.goal[i].float().argmax(1)]
    cos_goal = torch.nn.functional.cosine_similarity(push, goal - obs["pos"], dim=-1)
    assert (cos_goal[moving] > 0.999).all()


def test_frozen_noise_serves_one_draw_per_demo_step():
    policy = FlowPolicy().eval()
    args = train_adapter.parse_args(["--name", "x", "--corruption", "{rotation_deg: 60}", "--n-demos", "20",
                                     "--val-demos", "10", "--frozen-noise"])
    data = train_adapter.expert_splits(args, None, policy, "cpu", np.random.default_rng(0), *corruption_spec(args.corruption))
    train, val = data["train"], data["val"]
    assert train.noise is not None and val.noise is None  # validation keeps fresh commands
    obs, chunk, push, core, i, k = train.sample(400)
    sigma = torch.ones(len(i))
    served = train_adapter.bci_command(policy, push, core, sigma, train.noise_at(i, k))
    # a demo step serves the same command every time it is drawn; fresh draws differ
    torch.testing.assert_close(served, train_adapter.bci_command(policy, push, core, sigma, train.noise_at(i, k)))
    active = train_adapter.active_of(push)
    assert active.any() and not torch.allclose(served[active], train_adapter.bci_command(policy, push, core, sigma)[active])
    with pytest.raises(SystemExit):
        train_adapter.parse_args(["--name", "x", "--corruption", "{rotation_deg: 60}", "--frozen-noise", "--closed-loop", "u"])


def test_a_time_budget_keeps_the_demonstrations_that_fit_in_it():
    """--demo-seconds: as many of the task's demonstrations as fit in the budget, each taking its duration up to the
    goal's selection; a larger budget extends the same draw, and about a quarter of it is without the obstacle."""
    args = train_adapter.parse_args(["--name", "x", "--corruption", "{scale: [1.0, 0.25]}"])
    ring = obstacle_spec({"kind": "ring", "around": "goal", "gap_angle": [15, 30]})
    def draw(seconds):
        return train_adapter.budget_demos(args, ring, np.random.default_rng(0), seconds, p_free=0.25, chunk=16)

    (small, task, _, small_lengths), (large, _, _, large_lengths) = draw(12.0), draw(40.0)
    n = len(small)
    assert 0 < n < len(large)
    assert small_lengths.sum() * task.dt <= 12.0 < large_lengths[:n + 1].sum() * task.dt  # all that fit and no more
    np.testing.assert_array_equal(small.start, large.start[:n])  # the same draw, extended
    np.testing.assert_array_equal(small_lengths, large_lengths[:n])
    assert 0 < (~large.obstacle_mask.any(1)).mean() < 0.5


def test_a_fine_tune_on_a_time_budget_keeps_its_best_scored_weights(ckpt_dir, monkeypatch):
    """train_policy.py --demo_seconds draws the demonstrations as the adapter trainer does (a task module given the
    same budget and seed gets the same ones); --eval_steps scores the EMA weights alone with the goal given and saves
    the best, without changing the training itself."""
    from train import train_policy
    monkeypatch.setattr(train_policy, "policy_dir", lambda name: ckpt_dir / name)
    monkeypatch.setattr(train_policy, "policy_path", lambda name: ckpt_dir / name / "policy.pt")
    tiny = ["--val_demos", "10", "--steps", "4", "--batch", "16", "--log_every", "2", "--warmup", "2",
            "--goal_dropout", "0.25", "--targets", "4", "4", "--reach", "0.35", "0.5"]
    ring = "{kind: ring, around: goal, gap_angle: [15, 30]}"
    train_policy.main(["--name", "base", "--n_demos", "30", "--d_model", "32", "--layers", "1", "--heads", "2", *tiny])
    ft = ["--init", "base", "--obstacle", ring, "--demo_seconds", "12", *tiny]
    train_policy.main(["--name", "scored", *ft, "--eval_steps", "2", "--eval_trials", "4"])
    train_policy.main(["--name", "plain", *ft])
    log, plain = (json.loads((ckpt_dir / n / "train_log.json").read_text()) for n in ("scored", "plain"))
    assert [e["step"] for e in log["eval"]] == [2, 4]  # the step asked for and the last
    assert log["best_step"] == max(log["eval"], key=lambda e: (e["success"], e["step"]))["step"]
    for key in ("train", "val"):  # scoring left the training's random state alone
        assert [e[key] for e in log["log"]] == pytest.approx([e[key] for e in plain["log"]], rel=1e-4)
    assert "eval" not in plain
    n = log["args"]["n_demos"]
    args = train_adapter.parse_args(["--kind", "task", "--obstacle", ring, "--user-module", "u", "--corruption",
                                     "{scale: [1.0, 0.25]}", "--name", "t", "--demo-seconds", "12", "--val-demos", "10"])
    data = train_adapter.expert_splits(args, args.obstacle, FlowPolicy().eval(), "cpu", np.random.default_rng(args.seed),
                                       *corruption_spec(args.corruption), args.p_free)
    assert 0 < n == args.n_demos == len(data["train"].sampler.lengths)


def test_train_an_image_task_module_from_the_render(ckpt_dir):
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "t_cnn",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", "--token-source", "cnn",
                                                  "--render-size", "64", *COMMON]))
    ckpt = torch.load(ckpt_dir / "t_cnn" / "adapter.pt")
    assert ckpt["config"]["token_source"] == "cnn" and ckpt["config"]["image_size"] == 64 and ckpt["meta"]["token_source"] == "cnn"
    loaded = ReversalAdapter.load(ckpt_dir / "t_cnn" / "adapter.pt")
    assert loaded.task.token_source == "cnn"
    log = json.loads((ckpt_dir / "t_cnn" / "train_log.json").read_text())["log"]
    assert {"val_fm", "chunk_collision_hidden", "collide_composed"} <= set(log[-1])


def test_vlm_sources_use_the_encoder_once_per_split(ckpt_dir, monkeypatch):
    calls = []
    class Stub:
        def __init__(self, *a, **k): pass
        def image_tokens(self, images):
            calls.append(len(images))
            return np.zeros((len(images), 256, 2048), np.float16)

        def close(self): calls.append("closed")
    monkeypatch.setattr(train_adapter, "SceneEncoder", Stub)
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "t_vlm",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", "--token-source", "vlm_image",
                                                  "--render-size", "56", *COMMON]))
    assert calls[-1] == "closed" and len(calls) == 3                      # train split, val split, closed
    assert torch.load(ckpt_dir / "t_vlm" / "adapter.pt")["config"]["scene_dim"] == 2048


def test_train_a_pooled_task_module_over_two_contexts(ckpt_dir, tmp_path):
    for name, corr in (("u_rot", "{rotation_deg: 60}"), ("u_gain", "{scale: [1.0, 0.25]}")):
        train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", corr, "--name", name, *COMMON]))
    (tmp_path / "pooled.yaml").write_text(
        "entries:\n  - {obstacle: {kind: disc, radius: 0.08, along: 0.5}, user_module: u_rot, corruption: {rotation_deg: 60}}\n"
        "  - {obstacle: {kind: ring, around: start, gap_angle: [15, 30]}, user_module: u_gain, corruption: {scale: [1.0, 0.25]}}\n")
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--pooled", str(tmp_path / "pooled.yaml"), "--name", "t_all",
                                                  "--token-source", "cnn", "--render-size", "64", *COMMON]))
    ckpt = torch.load(ckpt_dir / "t_all" / "adapter.pt")
    assert ckpt["meta"]["obstacle"] is None and [e["user_module"] for e in ckpt["meta"]["pooled"]] == ["u_rot", "u_gain"]
    log = json.loads((ckpt_dir / "t_all" / "train_log.json").read_text())["log"]
    assert {"val_fm", "val_fm_0", "val_fm_1", "chunk_collision_hidden_1"} <= set(log[-1])
    with pytest.raises(SystemExit):   # --pooled and --obstacle together
        train_adapter.parse_args(["--kind", "task", "--pooled", str(tmp_path / "pooled.yaml"), "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}",
                                  "--name", "x", "--corruption", "{rotation_deg: 60}"])


POOLED = (
    ("u_rot", "{rotation_deg: 60}", "{kind: disc, radius: 0.08, along: 0.5}"),
    ("u_gain", "{scale: [1.0, 0.25]}", "{kind: ring, around: start, gap_angle: [15, 30]}"),
)


@pytest.fixture
def pooled_fixture(ckpt_dir, tmp_path):
    """Two user modules and a two-entry pooled file (a disc task with a rotation, a ring task with a gain)."""
    for name, corr, _ in POOLED:
        train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", corr, "--name", name, *COMMON]))
    (tmp_path / "pooled.yaml").write_text(
        "entries:\n" + "".join(f"  - {{obstacle: {obs}, user_module: {name}, corruption: {corr}}}\n" for name, corr, obs in POOLED)
    )
    return tmp_path / "pooled.yaml"


def _record_splits(monkeypatch):
    """Wrap expert_splits: the list every call's train split lands in (its layouts' arrays and frozen noise)."""
    splits, real = [], train_adapter.expert_splits

    def wrapped(*a, **k):
        out = real(*a, **k)
        splits.append(out)
        return out

    monkeypatch.setattr(train_adapter, "expert_splits", wrapped)
    return splits


def test_the_pooled_arm_trains_on_the_per_task_modules_demonstrations(pooled_fixture, monkeypatch):
    """Each pooled entry draws its layouts, demos and frozen noise as a single-context run of it at the same seed."""
    extra = ["--token-source", "cnn", "--render-size", "64", "--frozen-noise", "--lambda-field", "0.1"]
    splits = _record_splits(monkeypatch)
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--pooled", str(pooled_fixture), "--name", "t_all", *extra, *COMMON]))
    pooled = list(splits)
    assert len(pooled) == 2
    for j, (name, corr, obstacle) in enumerate(POOLED):
        splits.clear()
        train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", obstacle, "--user-module", name, "--corruption", corr,
                                                      "--name", f"t_{j}", *extra, *COMMON]))
        (single,) = splits
        for split in ("train", "val"):
            a, b = pooled[j][split].sampler, single[split].sampler
            for field in ("obstacles", "obstacle_mask", "targets", "pos", "lengths"):
                torch.testing.assert_close(getattr(a, field), getattr(b, field), msg=f"entry {j} {split} {field}")
            torch.testing.assert_close(a.scene["cnn"], b.scene["cnn"])
        torch.testing.assert_close(pooled[j]["train"].noise, single["train"].noise, msg=f"entry {j} frozen noise")
        torch.testing.assert_close(pooled[j]["train"].push, single["train"].push)
    assert not torch.equal(pooled[0]["train"].sampler.obstacles, pooled[1]["train"].sampler.obstacles)


def test_the_pooled_trainer_alternates_the_contexts_each_with_its_own_user_module(pooled_fixture, monkeypatch):
    splits = _record_splits(monkeypatch)  # one per entry, in order
    users = {}  # id of a loaded user module -> the module's name
    load = train_adapter.ReversalAdapter.load

    def recording_load(path, *a, **k):
        adapter = load(path, *a, **k)
        users[id(adapter.user)] = path.parent.name
        return adapter

    monkeypatch.setattr(train_adapter.ReversalAdapter, "load", staticmethod(recording_load))
    drawn, used = [], []  # per training step: the entry whose train data was drawn, the user module the field loss used
    real_sample, real_field = train_adapter.ExpertData.sample, train_adapter.field_loss

    def sample(self, *a, **k):
        entries = [j for j, s in enumerate(splits) if s["train"] is self]
        if entries:  # not the validation draws
            drawn.append(entries[0])
        return real_sample(self, *a, **k)

    def field(policy, user, *a, **k):
        used.append(users[id(user)])
        return real_field(policy, user, *a, **k)

    monkeypatch.setattr(train_adapter.ExpertData, "sample", sample)
    monkeypatch.setattr(train_adapter, "field_loss", field)
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--pooled", str(pooled_fixture), "--name", "t_all", "--token-source", "cnn",
                                                  "--render-size", "64", "--lambda-field", "0.1", *COMMON, "--steps", "6"]))
    assert drawn == [step % 2 for step in range(1, 7)]  # step % len(entries), the steps counting from 1
    assert used == [POOLED[j][0] for j in drawn] == ["u_gain", "u_rot"] * 3  # each context with its own user module
