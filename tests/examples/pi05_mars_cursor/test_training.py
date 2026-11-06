"""The adapter trainer (train/train_adapter.py): its data, one tiny run per end-to-end path (a module fixture's user
module, its closed-loop recalibration and a disc task module; the image module with jitter and rigid copies; the pooled
module; the teacher's student; the two-view module; the user module on a fine-tuned policy) and the parser's refusals.
The losses are pinned against the cursor reference in test_steering_equivalence.py."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from cursor_sa import config
from cursor_sa.readout import corruption_spec, curl_commands
from cursor_sa.env import TaskConfig, obstacle_spec
from cursor_sa.demos import demo_positions, make_demos, replay
from cursor_sa.policy import FlowPolicy
from cursor_sa.steering import ReversalAdapter, TaskModule, UserModule
from lerobot.policies.steering import LearnedTaskModule, save_adapter_module
from train import train_adapter
from _view_stub import TokenStub, ViewStub

COMMON = ["--n-demos", "60", "--val-demos", "40", "--steps", "4", "--log-every", "2", "--batch", "32"]
ROT = ["--corruption", "{rotation_deg: 60}"]
TASK = ["--kind", "task", "--user-module", "u", *ROT]
DISC = "{kind: disc, radius: 0.03, along: 0.5}"  # the disc task module's (and so the teacher's) task
IMAGE = "{kind: disc, radius: 0.08, along: 0.5}"
RING = "{kind: ring, around: goal, gap_angle: [15, 30]}"
CNN = ["--token-source", "cnn", "--render-size", "64"]
VIEWS = ["--token-source", "vlm_views", "--render-size", "56", "--scene-rank", "2", "--view-affine", "0.5", "--token-drop", "0.3",
         "--frozen-noise", "--render-workers", "0"]


def _args(*argv):
    return train_adapter.parse_args(list(argv))


def _log(root, name):
    return json.loads((root / name / "train_log.json").read_text())["log"]


def _recording(monkeypatch, name):
    """Wrap train_adapter.<name>: the list every call's result lands in, and the real function."""
    calls, real = [], getattr(train_adapter, name)
    monkeypatch.setattr(train_adapter, name, lambda *a, **k: calls.append(real(*a, **k)) or calls[-1])
    return calls, real


def _base_policy():
    return FlowPolicy.load(train_adapter.policy_path(config.load_arm("base.yaml")["model"]), "cpu")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The checkpoint folder every run of this module writes to, holding three tiny runs: the user module `u`, its
    ReFIT-style closed-loop recalibration `u_cl`, and the disc task module `t` with `u` in the loop, saved every 2 steps."""
    root = tmp_path_factory.mktemp("adapters")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train_adapter, "adapter_dir", lambda name: root / name)
        mp.setattr(train_adapter, "adapter_path", lambda name: root / name / "adapter.pt")
        for argv in (["--kind", "user", *ROT, "--name", "u"],
                     ["--kind", "user", *ROT, "--name", "u_cl", "--closed-loop", "u", "--closed-loop-trials", "4",
                      "--closed-loop-depth", "2", "--closed-loop-time-limit", "1.0"],
                     [*TASK, "--obstacle", DISC, "--name", "t", "--save-every", "2"]):
            train_adapter.train(_args(*argv, *COMMON))
        yield root


def test_expert_data_serves_the_pushes_cores_and_frozen_noise():
    policy = FlowPolicy().eval()
    args = _args("--name", "x", "--corruption", "{scale: [1.0, 0.25]}", "--n-demos", "40", "--val-demos", "10", "--frozen-noise")
    data = train_adapter.expert_splits(args, None, policy, "cpu", np.random.default_rng(0), *corruption_spec(args.corruption))
    train = data["train"]
    obs, chunk, push, core, i, k = train.sample(400)
    torch.testing.assert_close(core, torch.clamp(push * torch.tensor([1.0, 0.25]), -1, 1))  # the core is M @ push
    goal = obs["targets"][torch.arange(len(i)), train.sampler.goal[i].float().argmax(1)]
    moving = push.norm(dim=-1) > 0.99  # full deflection: far from the goal, where the push points
    assert moving.any() and (torch.nn.functional.cosine_similarity(push, goal - obs["pos"], dim=-1)[moving] > 0.999).all()
    assert train.noise is not None and data["val"].noise is None  # validation keeps fresh commands
    sigma, active = torch.ones(len(i)), train_adapter.active_of(push)
    served = train_adapter.bci_command(policy, push, core, sigma, train.noise_at(i, k))
    torch.testing.assert_close(served, train_adapter.bci_command(policy, push, core, sigma, train.noise_at(i, k)))  # one draw per step
    assert active.any() and not torch.allclose(served[active], train_adapter.bci_command(policy, push, core, sigma)[active])
    # a curl's core is its velocity driven from rest by the pushes in turn (the trainer's readout_core), not M @ push
    args = _args("--name", "x", "--corruption", "{curl: 20, tau: 0.1}", "--n-demos", "10", "--val-demos", "4")
    _, curl = corruption_spec(args.corruption)
    curled = train_adapter.expert_splits(args, None, policy, "cpu", np.random.default_rng(0), None, curl)["train"]
    torch.testing.assert_close(curled.core, torch.as_tensor(curl_commands(curled.push.numpy(), curl), dtype=torch.float32))
    assert not torch.allclose(curled.core, curled.push, atol=1e-2)


def test_the_field_loss_trains_the_task_module_through_the_frozen_user_module():
    """field_loss's gradient reaches the task module (bound to the observation) through the frozen user module."""
    policy, user, task = FlowPolicy().eval().requires_grad_(False), UserModule().requires_grad_(False), TaskModule()
    torch.nn.init.normal_(user.net[-1].weight, std=0.05)
    torch.nn.init.normal_(task.out.weight, std=0.05)
    B, g = 4, torch.Generator().manual_seed(0)
    free = dict(pos=torch.rand(B, 2, generator=g), hist=torch.zeros(B, policy.hist, 2), targets=torch.rand(B, 3, 2, generator=g),
                mask=torch.ones(B, 3, dtype=torch.bool), goal=torch.zeros(B, 3, dtype=torch.bool),
                obstacles=torch.rand(B, 2, 3, generator=g) * 0.1, obstacle_mask=torch.ones(B, 2, dtype=torch.bool))
    latent, bci = torch.randn(B, policy.horizon, 2, generator=g), torch.randn(B, policy.horizon, 2, generator=g)
    loss = train_adapter.field_loss(policy, user, task, free, latent, bci, torch.tensor([1.0, 0.0, 1.0, 1.0]))
    loss.backward()
    assert task.out.weight.grad is not None and task.out.weight.grad.abs().sum() > 0


def test_a_time_budget_keeps_the_demonstrations_that_fit_in_it():
    """--demo-seconds: as many of the task's demonstrations as fit in the budget, each taking its duration up to the
    goal's selection; a larger budget extends the same draw, and about a quarter of it is without the obstacle."""
    args = _args("--name", "x", "--corruption", "{scale: [1.0, 0.25]}")
    ring = obstacle_spec({"kind": "ring", "around": "goal", "gap_angle": [15, 30]})
    (small, task, _, small_lengths), (large, _, _, large_lengths) = (
        train_adapter.budget_demos(args, ring, np.random.default_rng(0), s, p_free=0.25, chunk=16) for s in (12.0, 40.0))
    n = len(small)
    assert 0 < n < len(large)
    assert small_lengths.sum() * task.dt <= 12.0 < large_lengths[:n + 1].sum() * task.dt  # all that fit and no more
    np.testing.assert_array_equal(small.start, large.start[:n])  # the same draw, extended
    np.testing.assert_array_equal(small_lengths, large_lengths[:n])
    assert 0 < (~large.obstacle_mask.any(1)).mean() < 0.5


def test_move_pictures_moves_a_share_and_fills_with_the_background():
    images = np.full((200, 3, 32, 32), 255, np.uint8)
    images[:, :, 12:20, 12:20] = 0  # a dark square on the picture's centre
    out, moved = train_adapter.move_pictures(images, np.random.default_rng(0), 0.5)
    assert out.shape == images.shape and out.dtype == np.uint8 and 60 < moved.sum() < 140
    assert np.array_equal(out[~moved], images[~moved]) and not np.array_equal(out[moved], images[moved])
    assert (out[:, :, 0, 0] == 255).all() and (out[:, :, -1, -1] == 255).all()  # the uncovered border is background
    ys, xs = np.mgrid[:32, :32]
    for img in out[moved][:20]:
        w = 255.0 - img[0]
        assert abs((w * ys).sum() / w.sum() - 15.5) <= 2.2 and abs((w * xs).sum() / w.sum() - 15.5) <= 2.2  # 5% of 32 px, and some blur
    again, moved2 = train_adapter.move_pictures(images, np.random.default_rng(0), 0.5)
    assert np.array_equal(again, out) and np.array_equal(moved2, moved)
    assert not train_adapter.move_pictures(images, np.random.default_rng(0), 0.0)[1].any()


def test_the_frame_table_is_the_same_rendered_in_a_process_pool(monkeypatch):
    """--render-workers 2 renders the chunks in spawned processes: the same frame table as rendering in-process, the
    pictures' moves and the reduction's sample drawn in the trainer either way; --scene-rank 0 keeps the full width."""
    monkeypatch.setattr(train_adapter, "VIEW_CHUNK", 8)
    monkeypatch.setattr(train_adapter, "REDUCTION_SAMPLE", 16)
    args = _args(*TASK, "--obstacle", IMAGE, "--name", "x", *VIEWS, "--view-stride", "4")
    rng = np.random.default_rng(0)
    layouts, task = train_adapter.layouts_for(args, args.obstacle, rng, 6, args.p_free)
    actions, lengths = make_demos(rng, layouts, task, allow_failed=True)
    tables, cuda = {}, torch.cuda.is_available()
    for workers in (2, 0):
        args.render_workers = workers
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all() if cuda else []
        tables[workers] = train_adapter.view_frames(args, layouts, actions, lengths, ViewStub(), "cpu", affine_rng=np.random.default_rng(1))
        assert torch.equal(torch.get_rng_state(), cpu_state)  # the encoder's draws left torch's streams alone
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all() if cuda else [], cuda_state, strict=True))
    (pooled, start, info), (inline, start0, info0) = tables[2], tables[0]
    assert info["frames"] == len(inline) > 2 * 8 and info["moved"] == info0["moved"] > 0  # several chunks through the pool
    assert torch.equal(pooled, inline) and np.array_equal(start, start0) and pooled.shape[1:] == (512, 2)
    args.scene_rank = 0
    full, _, info = train_adapter.view_frames(args, layouts, actions, lengths, ViewStub(), "cpu")
    assert full.shape == (info["frames"], 512, 2048) and info["kept"] is None and info["reduction"] is None and info["moved"] == 0


def test_a_user_module_trains_and_recalibrates_closed_loop(trained):
    ckpt = torch.load(trained / "u" / "adapter.pt")
    assert ckpt["kind"] == "user" and ckpt["meta"]["obstacle"] is None and ckpt["meta"]["args"]["corruption"] == {"rotation_deg": 60}
    assert [ckpt["meta"][k] for k in ("jitter", "jitter_steps", "rigid_motions", "rigid_kept")] == [None] * 4  # the task module's flags
    assert ReversalAdapter.load(trained / "u" / "adapter.pt").kind == "user"
    log = _log(trained, "u")
    assert len(log) == 2 and {"heading_cos_none", "heading_cos_user"} <= set(log[-1])
    # recalibrated closed loop: the initial module runs the open-field trials, their states join the batches
    args = torch.load(trained / "u_cl" / "adapter.pt")["meta"]["args"]
    assert args["closed_loop"] == ["u"] and args["closed_loop_mix"] == 0.5
    assert {"heading_cos_user", "closed_heading_cos_none", "closed_heading_cos_user"} <= set(_log(trained, "u_cl")[-1])


def test_a_disc_task_module_trains_with_the_user_module_in_the_loop(trained):
    final = torch.load(trained / "t" / "adapter.pt")
    meta = final["meta"]
    assert final["kind"] == "task" and meta["obstacle"]["kind"] == "disc" and meta["args"]["user_module"] == "u"
    assert meta["token_geometry"] == "absolute" and "token_geometry" not in final["config"]
    assert (meta["jitter"], meta["jitter_steps"], meta["rigid_motions"], meta["rigid_kept"]) == (0.0, 4, 0, None)  # off, and said so
    assert isinstance(ReversalAdapter.load(trained / "t" / "adapter.pt").task, TaskModule)
    log = _log(trained, "t")
    assert {"val_fm", "chunk_collision_hidden", "chunk_collision_policy_given", "collide_user", "collide_composed"} <= set(log[-1])
    assert all(np.isfinite(row[k]) and row[k] >= 0 for row in log for k in ("train_fm", "train_field")), log  # the loss parts
    for n in (2, 4):  # --save-every 2: the module at each of those steps, as a sibling
        step = torch.load(trained / f"t_step{n}" / "adapter.pt")
        assert step["meta"]["step"] == n and step["config"] == final["config"] and "step" not in meta
        assert {k: v for k, v in step["meta"].items() if k != "step"} == meta
        assert ReversalAdapter.load(trained / f"t_step{n}" / "adapter.pt").task is not None
        same = all(torch.equal(step["state_dict"][k], v) for k, v in final["state_dict"].items())
        assert same == (n == 4)  # the module as it was at that step


def test_an_image_module_trains_on_jittered_and_moved_copies(trained, monkeypatch, capsys):
    """The image path (cnn, relative geometry) with --jitter and --rigid-motions: copies of the train split's
    demonstrations moved rigidly (their pushes moved with them, not recomputed), the jitter on its sampler only."""
    (splits, _), (made, _) = _recording(monkeypatch, "expert_splits"), _recording(monkeypatch, "augment_demos")
    pushed, real_pushes = _recording(monkeypatch, "expert_pushes")
    capsys.readouterr()
    train_adapter.train(_args(*TASK, "--obstacle", IMAGE, "--name", "t_aug", *CNN, "--token-geometry", "relative", "--jitter", "0.02",
                              "--jitter-steps", "3", "--rigid-motions", "2", *COMMON))
    out = capsys.readouterr().out.splitlines()
    ckpt = torch.load(trained / "t_aug" / "adapter.pt")
    cfg, meta = ckpt["config"], ckpt["meta"]
    assert (cfg["token_source"], cfg["image_size"], cfg["token_geometry"]) == ("cnn", 64, "relative") and "scene_pos" not in ckpt["state_dict"]
    assert (meta["token_source"], meta["token_geometry"], meta["teacher"], meta["teacher_demos"], meta["scene_center"]) == ("cnn", "relative", None, None, None)
    assert (meta["scene_store"], meta["scene_rank"], meta["args"]["scene_rank"]) == (None, None, None) and "scene_rank" not in cfg
    assert ReversalAdapter.load(trained / "t_aug" / "adapter.pt").task.token_geometry == "relative"
    assert {"val_fm", "chunk_collision_hidden", "collide_composed"} <= set(_log(trained, "t_aug")[-1])
    assert (meta["jitter"], meta["jitter_steps"], meta["rigid_motions"]) == (0.02, 3, 2) and type(meta["rigid_kept"]) is int
    (layouts, actions, lengths, kept, _), = made  # the val split never augments
    assert 0 < kept.sum() == meta["rigid_kept"] <= 2 * 60 and [len(p) for p in pushed] == [60, 40]  # train's originals, then val: no copy's
    train, val = splits[-1]["train"], splits[-1]["val"]
    assert train.sampler.jitter == 0.02 and train.sampler.jitter_steps == 3 and val.sampler.jitter == 0.0
    assert len(train.sampler.lengths) == len(train.sampler.scene["cnn"]) == 60 + kept.sum() and len(val.sampler.lengths) == 40  # every copy rendered
    push = train.push.numpy()
    origin, dots = np.repeat(np.arange(60), kept), (push * actions).sum(-1)
    assert (np.abs(dots[60:]).sum(1) > 0).all() and np.allclose(dots[60:], dots[origin], atol=1e-6)  # a rigid motion keeps push . action
    fresh = real_pushes(layouts, demo_positions(layouts, actions))
    assert np.allclose(push[:60], fresh[:60], atol=1e-6) and not np.allclose(push[60:], fresh[60:], atol=1e-3)  # a copy's are its original's moved
    assert f"train: 60 demonstrations and {kept.sum()} moved copies ({kept.min()}-{kept.max()} per demonstration)" in out
    seconds = f"train: 60 demonstrations ({lengths[:60].sum() * TaskConfig().dt:.0f} s of demonstrating)"
    assert any(line.startswith(seconds) for line in out), out  # the originals' seconds, not the copies'


def test_a_pooled_module_alternates_the_contexts_on_their_own_demonstrations(trained, tmp_path, monkeypatch):
    """--pooled: each entry draws its layouts, demos, frozen noise and rigid copies as a single-context run of it at the
    same seed would, and the batches alternate the entries, each with its own user module in the field loss."""
    entries = (("u", "{rotation_deg: 60}", IMAGE), ("u_cl", "{scale: [1.0, 0.25]}", "{kind: ring, around: start, gap_angle: [15, 30]}"))
    (tmp_path / "pooled.yaml").write_text("entries:\n" + "".join(f"  - {{obstacle: {o}, user_module: {u}, corruption: {c}}}\n" for u, c, o in entries))
    splits, real_splits = _recording(monkeypatch, "expert_splits")
    users, load = {}, train_adapter.ReversalAdapter.load  # id of a loaded user module -> its name

    def recording_load(path, *a, **k):
        adapter = load(path, *a, **k)
        users[id(adapter.user)] = path.parent.name
        return adapter

    drawn, used = [], []  # per training step: the entry whose train data was drawn, the user module the field loss used
    real_sample, real_field = train_adapter.ExpertData.sample, train_adapter.field_loss

    def sample(self, *a, **k):
        drawn.extend(j for j, s in enumerate(splits) if s["train"] is self)  # not the validation draws
        return real_sample(self, *a, **k)

    monkeypatch.setattr(train_adapter.ReversalAdapter, "load", staticmethod(recording_load))
    monkeypatch.setattr(train_adapter.ExpertData, "sample", sample)
    monkeypatch.setattr(train_adapter, "field_loss", lambda policy, user, *a, **k: used.append(users[id(user)]) or real_field(policy, user, *a, **k))
    flags = [*CNN, "--frozen-noise", "--lambda-field", "0.1", "--rigid-motions", "1", *COMMON]
    train_adapter.train(_args("--kind", "task", "--pooled", str(tmp_path / "pooled.yaml"), "--name", "t_all", *flags, "--steps", "6"))
    meta = torch.load(trained / "t_all" / "adapter.pt")["meta"]
    assert meta["obstacle"] is None and [e["user_module"] for e in meta["pooled"]] == ["u", "u_cl"]
    assert {"val_fm", "val_fm_0", "val_fm_1", "chunk_collision_hidden_1"} <= set(_log(trained, "t_all")[-1])
    assert drawn == [step % 2 for step in range(1, 7)]  # step % len(entries), the steps counting from 1
    assert used == [entries[j][0] for j in drawn] == ["u_cl", "u"] * 3
    assert all(len(s["train"].sampler.lengths) > 60 for s in splits)
    assert meta["rigid_kept"] == sum(len(s["train"].sampler.lengths) - 60 for s in splits)  # every entry's copies counted
    policy = _base_policy()
    for j, (name, corr, obstacle) in enumerate(entries):
        args = _args("--kind", "task", "--obstacle", obstacle, "--user-module", name, "--corruption", corr, "--name", "x", *flags)
        torch.manual_seed(args.seed)
        single = real_splits(args, obstacle_spec(args.obstacle), policy, "cpu", np.random.default_rng(args.seed), *corruption_spec(args.corruption),
                             args.p_free)
        for split in ("train", "val"):
            a, b = splits[j][split].sampler, single[split].sampler
            for field in ("obstacles", "obstacle_mask", "targets", "pos", "lengths"):
                torch.testing.assert_close(getattr(a, field), getattr(b, field), msg=f"entry {j} {split} {field}")
            torch.testing.assert_close(a.scene["cnn"], b.scene["cnn"])
        torch.testing.assert_close(splits[j]["train"].noise, single["train"].noise, msg=f"entry {j} frozen noise")
        torch.testing.assert_close(splits[j]["train"].push, single["train"].push)
    assert not torch.equal(splits[0]["train"].sampler.obstacles, splits[1]["train"].sampler.obstacles)


def test_the_teacher_loop_is_guarded(trained, monkeypatch):
    policy = _base_policy()

    def demos(*argv):
        args = _args(*TASK, "--name", "s", "--teacher-chunk", "8", *argv)
        return train_adapter.teacher_demos(args, args.obstacle, policy, "cpu", np.random.default_rng(0), 4, args.p_free)

    with pytest.raises(ValueError, match="obstacle"):  # the teacher was trained on a disc
        demos("--obstacle", RING, "--teacher", "t")
    meta = dict(obstacle={"kind": "disc", "radius": 0.03, "along": 0.5})
    save_adapter_module(LearnedTaskModule(horizon=16, token_source="cnn", image_size=64), trained / "t_cnn" / "adapter.pt", "task", meta=meta)
    with pytest.raises(ValueError, match="discs"):  # an image module is not a teacher
        demos("--obstacle", DISC, "--teacher", "t_cnn")
    calls = []
    monkeypatch.setattr(train_adapter, "run_trials", lambda arm, layouts, *a, **k: calls.append(len(layouts)) or [dict(outcome="timeout", path=[[0.0, 0.0]])] * len(layouts))
    with pytest.raises(RuntimeError, match="hits"):
        demos("--obstacle", DISC, "--teacher", "t")
    assert sum(calls) == 88  # 20 * 4 + 8 layouts drawn, then the cap


def test_a_task_module_trains_on_the_teachers_hits(trained, monkeypatch):
    """--teacher: the teacher's hit rollouts as demonstrations, resampled to the policy's step rate (`planned_steps`:
    every second position of a half-speed path, an odd final control step dropped)."""
    path = np.cumsum(np.repeat([[0.01, 0.0]], 6, axis=0), axis=0)  # 6 positions: 5 control steps
    assert train_adapter.planned_steps(path).shape == (2, 2)  # control steps 0-1 and 2-3; the 5th is half a step
    assert np.allclose(train_adapter.planned_steps(path), [[0.02, 0.0], [0.02, 0.0]])
    assert train_adapter.planned_steps(np.vstack([path, path[-1:] + [0.01, 0.0]])).shape == (3, 2)
    made, _ = _recording(monkeypatch, "teacher_demos")
    train_adapter.train(_args(*TASK, "--obstacle", DISC, "--name", "s", "--teacher", "t", "--teacher-chunk", "8", *COMMON, "--n-demos", "4",
                              "--val-demos", "2"))
    meta = torch.load(trained / "s" / "adapter.pt")["meta"]
    assert meta["teacher"] == "t" and meta["teacher_demos"]["train"]["kept"] == 4 and meta["teacher_demos"]["val"]["kept"] == 2
    assert {"val_fm", "collide_composed"} <= set(_log(trained, "s")[-1])
    (layouts, task, actions, lengths, info), _ = made
    assert len(layouts) == 4 and info["drawn"] % 8 == 0 and sum(info["outcomes"].values()) == info["drawn"]
    for i, r in enumerate(info["records"]):
        path = np.asarray(r["path"])
        assert r["outcome"] == "hit" and lengths[i] == (len(path) - 1) // 2
        assert np.allclose(actions[i, :lengths[i]], train_adapter.planned_steps(path))
        assert np.allclose(np.abs(np.diff(path[::2], axis=0)).sum(), np.abs(np.diff(path, axis=0)).sum(), atol=2e-3)  # same travel
    assert np.all(actions[np.arange(actions.shape[1])[None] >= lengths[:, None]] == 0)
    assert replay(actions, layouts, replace(task, cursor_noise=0.0)).success.all()  # a valid demonstration at the policy's rate


def test_a_two_view_module_trains_on_the_scripted_draws(trained, monkeypatch):
    class Encoder(ViewStub):
        closed = 0

        def close(self):
            Encoder.closed += 1

    monkeypatch.setattr(train_adapter, "SceneEncoder", Encoder)
    monkeypatch.setattr(train_adapter, "VIEW_CHUNK", 32)
    monkeypatch.setattr(train_adapter, "REDUCTION_SAMPLE", 64)
    ViewStub.made = 0
    splits, real_splits = _recording(monkeypatch, "expert_splits")
    seen, real_fm, real_field = {"fm": [], "field": []}, train_adapter.task_loss, train_adapter.field_loss
    monkeypatch.setattr(train_adapter, "task_loss", lambda policy, task, obs, *a: seen["fm"].append(obs.get("scene_keep")) or real_fm(policy, task, obs, *a))
    monkeypatch.setattr(train_adapter, "field_loss", lambda p, u, t, free, *a: seen["field"].append("scene_keep" in free) or real_field(p, u, t, free, *a))
    train_adapter.train(_args(*TASK, "--obstacle", IMAGE, "--name", "t_views", *VIEWS, "--scene-center", "no", *COMMON))
    assert ViewStub.made == 1 and Encoder.closed == 1  # one encoder for both splits, closed after
    ckpt = torch.load(trained / "t_views" / "adapter.pt")
    cfg, meta, state = ckpt["config"], ckpt["meta"], ckpt["state_dict"]
    assert (cfg["token_source"], cfg["wrist_span"], cfg["scene_rank"], cfg["scene_tokens"]) == ("vlm_views", 0.5, 2, 512)
    assert state["scene_basis"].abs().sum() > 0 and not state["scene_mean"].any()  # the fitted uncentred reduction
    assert (meta["view_stride"], meta["token_drop"], meta["view_affine"], meta["wrist_span"], meta["scene_rank"], meta["scene_center"]) == (2, 0.3, 0.5, 0.5, 2, "no")
    (views,) = splits
    info, lengths = meta["views"], views["train"].sampler.lengths.cpu().numpy()
    assert info["train"]["frames"] == int(np.ceil(lengths / 2).sum()) and 0 < info["train"]["moved"] < 2 * info["train"]["frames"]
    assert info["val"]["moved"] == 0 and 0 < info["train"]["kept"] <= 1 and info["val"]["kept"] is None
    assert views["train"].sampler.sample(4)[0]["scene"]["vlm_views"].shape == (4, 512, 2)
    assert {"val_fm", "chunk_collision_hidden", "collide_composed", "train_fm", "train_field"} <= set(_log(trained, "t_views")[-1])
    assert ReversalAdapter.load(trained / "t_views" / "adapter.pt").task.wrist_span == 0.5
    keeps = [k for k in seen["fm"] if k is not None]  # --token-drop hides tokens from the flow matching loss only
    assert len(keeps) == 4 and all(k.shape == (32, 512) and k.dtype == torch.bool for k in keeps)  # COMMON: 4 steps of batch 32
    assert 0.6 < torch.stack(keeps).float().mean() < 0.8 and any(k is None for k in seen["fm"])  # validation sees every token
    assert seen["field"] and not any(seen["field"])
    # the same seed with another source draws the same demonstrations
    args = _args(*TASK, "--obstacle", IMAGE, "--name", "x", *CNN, "--frozen-noise", *COMMON)
    policy = _base_policy()  # loaded first: building it draws from torch's generator
    torch.manual_seed(args.seed)
    cnn = real_splits(args, obstacle_spec(args.obstacle), policy, "cpu", np.random.default_rng(args.seed), *corruption_spec(args.corruption),
                      args.p_free)
    for split in ("train", "val"):
        for field in ("obstacles", "targets", "pos", "lengths"):
            torch.testing.assert_close(getattr(views[split].sampler, field), getattr(cnn[split].sampler, field))
    torch.testing.assert_close(views["train"].noise, cnn["train"].noise)


STORED = ["--token-source", "vlm_image", "--render-size", "32", "--rigid-motions", "1", *COMMON]


def test_a_stored_run_equals_the_resident_run(trained, tmp_path, monkeypatch, capsys):
    """--scene-store keeps the tokens in a disk store gathered per batch: the same module and log as the run whose
    layouts carry their tokens, the moved copies and the val split included."""
    monkeypatch.setattr(train_adapter, "SceneEncoder", TokenStub)
    train_adapter.train(_args(*TASK, "--obstacle", IMAGE, "--name", "t_resident", *STORED))
    capsys.readouterr()
    splits, _ = _recording(monkeypatch, "expert_splits")
    train_adapter.train(_args(*TASK, "--obstacle", IMAGE, "--name", "t_stored", *STORED, "--scene-store", str(tmp_path / "store")))
    out = capsys.readouterr().out.splitlines()
    resident, stored = (torch.load(trained / n / "adapter.pt") for n in ("t_resident", "t_stored"))
    assert resident["config"] == stored["config"] and resident["state_dict"].keys() == stored["state_dict"].keys()
    assert all(torch.equal(v, stored["state_dict"][k]) for k, v in resident["state_dict"].items())
    assert _log(trained, "t_resident") == _log(trained, "t_stored")
    (folder,) = (p.name for p in (tmp_path / "store").iterdir())
    assert folder.startswith("vlm_image_32_") and stored["meta"]["scene_store"] == folder and resident["meta"]["scene_store"] is None
    assert (stored["meta"]["scene_rank"], stored["meta"]["scene_center"]) == (None, None)
    (data,) = splits
    train = data["train"].sampler
    n = len(train.lengths)
    assert train.scene is None and n > 60 and f"train: {n} scenes, {n} newly encoded" in "\n".join(out)
    assert any(line.startswith("val: 40 scenes, 40 newly encoded") and line.endswith(f"store {tmp_path / 'store' / folder}")
               for line in out), out


def test_a_reduced_image_module_trains_from_the_store_and_carries_its_basis(trained, tmp_path, monkeypatch):
    """--scene-rank with vlm_image: the PCA is fitted on the train split's stored rows, the sampler feeds the reduced
    tokens and the module carries the reduction (relative geometry added after the projection)."""
    monkeypatch.setattr(train_adapter, "SceneEncoder", TokenStub)
    splits, _ = _recording(monkeypatch, "expert_splits")
    train_adapter.train(_args(*TASK, "--obstacle", IMAGE, "--name", "t_reduced", *STORED, "--token-geometry", "relative",
                              "--scene-rank", "8", "--scene-store", str(tmp_path / "store")))
    ckpt = torch.load(trained / "t_reduced" / "adapter.pt")
    cfg, meta, state = ckpt["config"], ckpt["meta"], ckpt["state_dict"]
    assert (cfg["scene_rank"], cfg["token_geometry"], meta["scene_rank"], meta["scene_center"]) == (8, "relative", 8, "yes")
    assert state["scene_basis"].shape == (2048, 8) and state["scene_basis"].abs().sum() > 0 and state["scene_mean"].any()
    task = ReversalAdapter.load(trained / "t_reduced" / "adapter.pt").task
    assert task.scene_in.in_features == 8 and task.scene_rank == 8
    (data,) = splits
    for split in ("train", "val"):
        obs, _ = data[split].sampler.sample(4)
        assert obs["scene"]["vlm_image"].shape == (4, 256, 8) and obs["scene"]["vlm_image"].dtype == torch.float16
        torch.testing.assert_close(data[split].sampler.reduction[1], state["scene_basis"])  # val reuses train's
    assert {"val_fm", "collide_composed"} <= set(_log(trained, "t_reduced")[-1])


def test_a_fine_tune_on_a_time_budget_keeps_its_best_scored_weights(trained, monkeypatch):
    """train_policy.py --init fine-tunes a trained policy for an obstacle task (its architecture and units, an obstacle
    head added); --demo_seconds draws the demonstrations as the adapter trainer does (a task module given the same
    budget and seed gets the same ones); --eval_steps scores the EMA weights alone with the goal given and saves the
    best, without changing the training itself; a user module calibrates on the result with --model."""
    from train import train_policy
    monkeypatch.setattr(train_policy, "policy_dir", lambda name: trained / name)
    monkeypatch.setattr(train_policy, "policy_path", lambda name: trained / name / "policy.pt")
    tiny = ["--val_demos", "10", "--steps", "4", "--batch", "16", "--log_every", "2", "--warmup", "2",
            "--goal_dropout", "0.25", "--targets", "4", "4", "--reach", "0.35", "0.5"]
    train_policy.main(["--name", "base", "--n_demos", "30", "--d_model", "32", "--layers", "1", "--heads", "2", *tiny])
    ft = ["--init", "base", "--obstacle", RING, "--demo_seconds", "12", *tiny]
    train_policy.main(["--name", "scored", *ft, "--eval_steps", "2", "--eval_trials", "4"])
    train_policy.main(["--name", "plain", *ft])
    base, scored = (torch.load(trained / n / "policy.pt") for n in ("base", "scored"))
    assert scored["config"] == {**base["config"], "obstacle_tokens": True} and hasattr(FlowPolicy.load(trained / "scored" / "policy.pt"), "obstacle_in")
    assert scored["args"]["init"] == "base" and scored["args"]["obstacle"]["kind"] == "ring" and scored["args"]["p_free"] == 0.25
    torch.testing.assert_close(scored["state_dict"]["action_scale"], base["state_dict"]["action_scale"])  # the base's units
    log, plain = (json.loads((trained / n / "train_log.json").read_text()) for n in ("scored", "plain"))
    assert log["action_scale"] == pytest.approx(base["state_dict"]["action_scale"].item())
    assert [e["step"] for e in log["eval"]] == [2, 4]  # the step asked for and the last
    assert log["best_step"] == max(log["eval"], key=lambda e: (e["success"], e["step"]))["step"]
    for key in ("train", "val"):  # scoring left the training's random state alone
        assert [e[key] for e in log["log"]] == pytest.approx([e[key] for e in plain["log"]], rel=1e-4)
    assert "eval" not in plain
    args = _args(*TASK, "--obstacle", RING, "--name", "t", "--demo-seconds", "12", "--val-demos", "10")
    data = train_adapter.expert_splits(args, args.obstacle, FlowPolicy().eval(), "cpu", np.random.default_rng(args.seed),
                                       *corruption_spec(args.corruption), args.p_free)
    assert 0 < log["args"]["n_demos"] == args.n_demos == len(data["train"].sampler.lengths)
    monkeypatch.setattr(train_adapter, "policy_path", lambda name: trained / name / "policy.pt")
    train_adapter.train(_args("--kind", "user", *ROT, "--model", "scored", "--name", "u_scored", *COMMON))
    assert torch.load(trained / "u_scored" / "adapter.pt")["meta"]["args"]["model"] == "scored"


CURLED = "{curl: 2.0, tau: 0.1}"
POOLED = ["--kind", "task", "--pooled", "POOLED", "--name", "x"]
VIEWS_TASK = [*TASK, "--obstacle", IMAGE, "--name", "x", "--token-source", "vlm_views"]
DISCS_TASK = [*TASK, "--obstacle", IMAGE, "--name", "x"]


@pytest.mark.parametrize("argv, outcome", [  # True: refused; else the parsed values
    (["--kind", "task", "--obstacle", IMAGE, *ROT, "--name", "x"], True),            # a task module needs its user module
    (["--kind", "user", "--obstacle", IMAGE, *ROT, "--name", "x"], True),           # a user module calibrates on the open field
    (["--kind", "user", *ROT, "--name", "x", "--frozen-noise", "--closed-loop", "u"], True),
    (["--kind", "user", *ROT, "--name", "x", "--cross-step-dims", "0"], True),     # needs --cross-step-rank
    (["--kind", "user", *ROT, "--name", "x", "--cross-step-rank", "2", "--cross-step-dims", "1"], {"cross_step_dims": [1], "step_rank": None}),
    (["--kind", "user", *ROT, "--name", "x"], {"cross_step_dims": None, "step_rank": None}),
    ([*DISCS_TASK, "--token-geometry", "relative"], True),                         # discs have no screen tokens
    ([*POOLED, "--obstacle", IMAGE], True),                                        # --pooled carries its own obstacles
    ([*DISCS_TASK, "--save-every", "3"], True),                                    # not a multiple of COMMON's --log-every 2
    ([*DISCS_TASK, "--teacher", "t", "--teacher-chunk", "0"], True),
    ([*DISCS_TASK, "--teacher", "t", "--demo-seconds", "10"], True),               # a time budget is the scripted expert's
    (["--kind", "user", *ROT, "--name", "x", "--teacher", "t"], True),
    ([*POOLED, "--teacher", "t"], True),                                           # no pooled teacher run
    ([*VIEWS_TASK, "--jitter", "0.02"], True),                                     # the picture would show the cursor elsewhere
    ([*VIEWS_TASK, "--rigid-motions", "2"], True),
    ([*TASK[:2], "--user-module", "u", "--corruption", CURLED, "--obstacle", IMAGE, "--name", "x", "--rigid-motions", "2"], True),  # a curl does not mirror
    ([*TASK[:2], "--user-module", "u", "--corruption", CURLED, "--obstacle", IMAGE, "--name", "x", "--jitter", "0.02"], {"jitter": 0.02}),
    (["--kind", "task", "--pooled", "CURLED_POOL", "--name", "x", "--rigid-motions", "2"], True),
    ([*POOLED, "--jitter", "0.02"], {"jitter": 0.02}),
    ([*DISCS_TASK, "--teacher", "t", "--rigid-motions", "2"], True),               # a teacher's demonstrations are rollouts
    ([*DISCS_TASK, "--teacher", "t", "--jitter", "0.02"], {"jitter": 0.02}),
    ([*DISCS_TASK, "--jitter", "-0.1"], True), ([*DISCS_TASK, "--jitter-steps", "0"], True), ([*DISCS_TASK, "--rigid-motions", "-1"], True),
    (["--kind", "user", *ROT, "--name", "x", "--jitter", "0.02"], True),            # the task module's
    (["--kind", "user", *ROT, "--name", "x", "--rigid-motions", "2"], True),
    ([*VIEWS_TASK, "--token-geometry", "relative"], True), ([*VIEWS_TASK, "--teacher", "t"], True),
    ([*POOLED, "--token-source", "vlm_views"], True), ([*DISCS_TASK, "--token-drop", "0.3"], True),
    ([*DISCS_TASK, *CNN, "--view-affine", "0.5"], True), ([*VIEWS_TASK, "--wrist-span", "0"], True),
    ([*VIEWS_TASK, "--view-stride", "0"], True), ([*VIEWS_TASK, "--scene-rank", "2048"], True),
    ([*VIEWS_TASK, "--token-drop", "1.0"], True), ([*VIEWS_TASK, "--view-affine", "1.5"], True),
    ([*VIEWS_TASK, "--scene-center", "no", "--scene-rank", "0"], True), ([*DISCS_TASK, *CNN, "--scene-center", "no"], True),
    ([*DISCS_TASK, "--scene-rank", "8"], True), ([*DISCS_TASK, *CNN, "--scene-rank", "8"], True),  # discs have no tokens; cnn's are not 2048 wide
    ([*DISCS_TASK, "--scene-store", "s"], True), ([*VIEWS_TASK, "--scene-store", "s"], True),     # no scene; the views' frame table
    ([*DISCS_TASK, "--token-source", "vlm_image", "--scene-center", "no"], True),                 # no reduction to centre
    ([*DISCS_TASK, "--token-source", "vlm_image"], {"scene_rank": None, "scene_store": None}),
    ([*DISCS_TASK, "--token-source", "vlm_prefix", "--scene-rank", "8", "--scene-center", "no", "--scene-store", "s"],
     {"scene_rank": 8, "scene_center": "no", "scene_store": "s"}),
    ([*VIEWS_TASK], {"scene_rank": 256}), ([*DISCS_TASK], {"scene_rank": None}),
])
def test_the_parser_refuses_what_makes_no_sense(argv, outcome, tmp_path):
    entries = {"POOLED": ROT[1], "CURLED_POOL": CURLED}
    for name, corruption in entries.items():
        (tmp_path / f"{name}.yaml").write_text(f"entries:\n  - {{obstacle: {IMAGE}, user_module: u, corruption: {corruption}}}\n")
    argv = [str(tmp_path / f"{a}.yaml") if a in entries else a for a in argv]
    if outcome is True:
        with pytest.raises(SystemExit):
            _args(*argv, *COMMON)
    else:
        args = _args(*argv, *COMMON)
        assert {k: getattr(args, k) for k in outcome} == outcome
