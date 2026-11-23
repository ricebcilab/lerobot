"""The adapter trainer (train/train_adapter.py): its data, one tiny run per end-to-end path the shipped studies use (a
module fixture's user module, its closed-loop recalibration and a disc task module; the image module with jitter and
rigid copies; the user module on a fine-tuned policy) and the parser's refusals. The pooled, two-view, reduced-token
and teacher variants, which no shipped sweep uses, have no tiny run here (their parser rules are still checked).
The losses are pinned against the cursor reference in test_steering_equivalence.py."""

import functools
import json

import numpy as np
import pytest
import torch

from cursor_sa import config
from lerobot.teleoperators.readout import Readout, corruption_spec
from lerobot.envs.cursor import TaskConfig, obstacle_spec
from cursor_sa.demos import demo_positions, make_demos
from cursor_sa.policy import FlowPolicy, load_policy
from lerobot.teleoperators.readout import DEADBAND
from lerobot.policies.steering import AdapterModules, LearnedTaskModule, LearnedUserModule, active_rows
from train import train_adapter
from _view_stub import ViewStub

COMMON = ["--n-demos", "60", "--val-demos", "40", "--steps", "4", "--log-every", "2", "--batch", "8"]
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
    return load_policy(train_adapter.policy_path(config.load_arm("base.yaml")["model"]), "cpu").model


def _small_check(mp):
    """What a tiny run spends its time on, made small: the trainer's held-out check on 64 samples instead of its
    2 048, and a time budget's demonstrations drawn 32 layouts at a time instead of 500."""
    mp.setattr(train_adapter, "expert_check", functools.partial(train_adapter.expert_check, n=64))
    mp.setattr(train_adapter, "budget_demos", functools.partial(train_adapter.budget_demos, chunk=32))


@pytest.fixture(autouse=True)
def small_check(monkeypatch):
    _small_check(monkeypatch)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The checkpoint folder every run of this module writes to, holding three tiny runs: the user module `u`, its
    ReFIT-style closed-loop recalibration `u_cl`, and the disc task module `t` with `u` in the loop, saved every 2 steps."""
    root = tmp_path_factory.mktemp("adapters")
    with pytest.MonkeyPatch.context() as mp:
        _small_check(mp)
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
    sigma, active = torch.ones(len(i)), active_rows(push, DEADBAND)
    served = train_adapter.bci_command(policy, core, sigma, train.noise_at(i, k))
    torch.testing.assert_close(served, train_adapter.bci_command(policy, core, sigma, train.noise_at(i, k)))  # one draw per step
    assert active.any() and not torch.allclose(served[active], train_adapter.bci_command(policy, core, sigma)[active])
    # a curl's core is its velocity driven from rest by the pushes in turn (the readout's core), not M @ push
    args = _args("--name", "x", "--corruption", "{curl: 20, tau: 0.1}", "--n-demos", "10", "--val-demos", "4")
    _, curl = corruption_spec(args.corruption)
    curled = train_adapter.expert_splits(args, None, policy, "cpu", np.random.default_rng(0), None, curl)["train"]
    torch.testing.assert_close(curled.core, torch.as_tensor(Readout(curl=curl).core(curled.push.numpy()), dtype=torch.float32))
    assert not torch.allclose(curled.core, curled.push, atol=1e-2)


def test_the_field_loss_trains_the_task_module_through_the_frozen_user_module():
    """field_loss's gradient reaches the task module (bound to the observation) through the frozen user module."""
    policy, user, task = FlowPolicy().eval().requires_grad_(False), LearnedUserModule(16, n_dims=2).requires_grad_(False), LearnedTaskModule()
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
    assert AdapterModules.load(trained / "u" / "adapter.pt").kind == "user"
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
    assert isinstance(AdapterModules.load(trained / "t" / "adapter.pt").task, LearnedTaskModule)
    log = _log(trained, "t")
    assert {"val_fm", "chunk_collision_hidden", "chunk_collision_policy_given", "collide_user", "collide_composed"} <= set(log[-1])
    assert all(np.isfinite(row[k]) and row[k] >= 0 for row in log for k in ("train_fm", "train_field")), log  # the loss parts
    for n in (2, 4):  # --save-every 2: the module at each of those steps, as a sibling
        step = torch.load(trained / f"t_step{n}" / "adapter.pt")
        assert step["meta"]["step"] == n and step["config"] == final["config"] and "step" not in meta
        assert {k: v for k, v in step["meta"].items() if k != "step"} == meta
        assert AdapterModules.load(trained / f"t_step{n}" / "adapter.pt").task is not None
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
    assert AdapterModules.load(trained / "t_aug" / "adapter.pt").task.token_geometry == "relative"
    assert {"val_fm", "chunk_collision_hidden", "collide_composed"} <= set(_log(trained, "t_aug")[-1])
    assert (meta["jitter"], meta["jitter_steps"], meta["rigid_motions"]) == (0.02, 3, 2) and type(meta["rigid_kept"]) is int
    (layouts, actions, lengths, kept, _), = made  # the val split never augments
    assert 0 < kept.sum() == meta["rigid_kept"] <= 2 * 60 and [len(p) for p in pushed] == [60, 40]  # train's originals, then val: no copy's
    train, val = splits[-1]["train"], splits[-1]["val"]
    assert train.sampler.jitter == 0.02 and train.sampler.jitter_steps == 3 and val.sampler.jitter == 0.0
    assert len(train.sampler.lengths) == len(train.sampler.scene["cnn"]) == 60 + kept.sum() and len(val.sampler.lengths) == 40  # every copy rendered
    push = train.push.cpu().numpy()  # the split sits on the training device
    origin, dots = np.repeat(np.arange(60), kept), (push * actions).sum(-1)
    assert (np.abs(dots[60:]).sum(1) > 0).all() and np.allclose(dots[60:], dots[origin], atol=1e-6)  # a rigid motion keeps push . action
    fresh = real_pushes(layouts, demo_positions(layouts, actions))
    assert np.allclose(push[:60], fresh[:60], atol=1e-6) and not np.allclose(push[60:], fresh[60:], atol=1e-3)  # a copy's are its original's moved
    assert f"train: 60 demonstrations and {kept.sum()} moved copies ({kept.min()}-{kept.max()} per demonstration)" in out
    seconds = f"train: 60 demonstrations ({lengths[:60].sum() * TaskConfig().dt:.0f} s of demonstrating)"
    assert any(line.startswith(seconds) for line in out), out  # the originals' seconds, not the copies'


def test_the_teacher_loop_is_guarded(trained, monkeypatch):
    policy = _base_policy()

    def demos(*argv):
        args = _args(*TASK, "--name", "s", "--teacher-chunk", "8", *argv)
        return train_adapter.teacher_demos(args, args.obstacle, policy, "cpu", np.random.default_rng(0), 4, args.p_free)

    with pytest.raises(ValueError, match="obstacle"):  # the teacher was trained on a disc
        demos("--obstacle", RING, "--teacher", "t")
    meta = dict(obstacle={"kind": "disc", "radius": 0.03, "along": 0.5})
    AdapterModules.save_module(LearnedTaskModule(horizon=16, token_source="cnn", image_size=64), trained / "t_cnn" / "adapter.pt", "task", meta=meta)
    with pytest.raises(ValueError, match="discs"):  # an image module is not a teacher
        demos("--obstacle", DISC, "--teacher", "t_cnn")
    calls = []
    monkeypatch.setattr(train_adapter, "run_trials", lambda arm, layouts, *a, **k: calls.append(len(layouts)) or [dict(outcome="timeout", path=[[0.0, 0.0]])] * len(layouts))
    with pytest.raises(RuntimeError, match="hits"):
        demos("--obstacle", DISC, "--teacher", "t")
    assert sum(calls) == 88  # 20 * 4 + 8 layouts drawn, then the cap


STORED = ["--token-source", "vlm_image", "--render-size", "32", "--rigid-motions", "1", *COMMON]


def test_a_fine_tune_on_a_time_budget_keeps_its_best_scored_weights(trained, monkeypatch):
    """train_policy.py --init fine-tunes a trained policy for an obstacle task (its architecture and units, an obstacle
    head added); --demo_seconds draws the demonstrations as the adapter trainer does (a task module given the same
    budget and seed gets the same ones); --eval_steps scores the EMA weights alone with the goal given and saves the
    best, without changing the training itself; a user module calibrates on the result with --model."""
    from train import train_policy
    monkeypatch.setattr(train_policy, "policy_path", lambda name: trained / name)
    monkeypatch.setattr(train_policy, "budget_demos", train_adapter.budget_demos)  # the small draw, as the adapter trainer's
    tiny = ["--val_demos", "10", "--steps", "4", "--batch", "16", "--log_every", "2", "--warmup", "2",
            "--goal_dropout", "0.25", "--targets", "4", "4", "--reach", "0.35", "0.5"]
    train_policy.main(["--name", "base", "--n_demos", "30", "--d_model", "32", "--layers", "1", "--heads", "2", *tiny])
    ft = ["--init", "base", "--obstacle", RING, "--demo_seconds", "12", *tiny]
    train_policy.main(["--name", "scored", *ft, "--eval_steps", "2", "--eval_trials", "4"])
    train_policy.main(["--name", "plain", *ft])
    base, scored = (load_policy(trained / n).model for n in ("base", "scored"))  # lerobot policy folders
    assert not any((trained / n / "policy.pt").exists() for n in ("base", "scored", "plain"))
    assert all((trained / "scored" / f"policy_{p}processor.json").is_file() for p in ("pre", "post"))  # its pipelines
    assert scored.config == {**base.config, "obstacle_tokens": True} and hasattr(scored, "obstacle_in")
    log, plain = (json.loads((trained / n / "train_log.json").read_text()) for n in ("scored", "plain"))
    assert log["args"]["init"] == "base" and log["args"]["obstacle"]["kind"] == "ring" and log["args"]["p_free"] == 0.25
    torch.testing.assert_close(scored.action_scale, base.action_scale)  # the base's units
    assert log["action_scale"] == pytest.approx(base.action_scale.item())
    assert [e["step"] for e in log["eval"]] == [2, 4]  # the step asked for and the last
    assert log["best_step"] == max(log["eval"], key=lambda e: (e["success"], e["step"]))["step"]
    for key in ("train", "val"):  # scoring left the training's random state alone
        assert [e[key] for e in log["log"]] == pytest.approx([e[key] for e in plain["log"]], rel=1e-4)
    assert "eval" not in plain
    assert log["args"]["eval_obstacle"] == log["args"]["obstacle"]  # scored on the task's own layouts by default
    args = _args(*TASK, "--obstacle", RING, "--name", "t", "--demo-seconds", "12", "--val-demos", "10")
    data = train_adapter.expert_splits(args, args.obstacle, FlowPolicy().eval(), "cpu", np.random.default_rng(args.seed),
                                       *corruption_spec(args.corruption), args.p_free)
    assert 0 < log["args"]["n_demos"] == args.n_demos == len(data["train"].sampler.lengths)
    monkeypatch.setattr(train_adapter, "policy_path", lambda name: trained / name)
    train_adapter.train(_args("--kind", "user", *ROT, "--model", "scored", "--name", "u_scored", *COMMON))
    assert torch.load(trained / "u_scored" / "adapter.pt")["meta"]["args"]["model"] == "scored"


POOLED = ["--kind", "task", "--pooled", "POOLED", "--name", "x"]
VIEWS_TASK = [*TASK, "--obstacle", IMAGE, "--name", "x", "--token-source", "vlm_views"]
DISCS_TASK = [*TASK, "--obstacle", IMAGE, "--name", "x"]


@pytest.mark.parametrize("argv, outcome", [  # True: refused; else the parsed values. One case per kind of rule.
    (["--kind", "task", "--obstacle", IMAGE, *ROT, "--name", "x"], True),            # a task module needs its user module
    (["--kind", "user", "--obstacle", IMAGE, *ROT, "--name", "x"], True),           # a user module calibrates on the open field
    (["--kind", "user", *ROT, "--name", "x", "--cross-step-rank", "2", "--cross-step-dims", "1"], {"cross_step_dims": [1], "step_rank": None}),
    ([*POOLED, "--obstacle", IMAGE], True),                                        # --pooled carries its own obstacles
    ([*VIEWS_TASK, "--jitter", "0.02"], True),                                     # the picture would show the cursor elsewhere
    ([*DISCS_TASK, "--scene-rank", "8"], True),                                    # discs have no tokens
    ([*DISCS_TASK, "--token-source", "vlm_prefix", "--scene-rank", "8", "--scene-center", "no", "--scene-store", "s"],
     {"scene_rank": 8, "scene_center": "no", "scene_store": "s"}),
    ([*VIEWS_TASK], {"scene_rank": 256}),
])
def test_the_parser_refuses_what_makes_no_sense(argv, outcome, tmp_path):
    entries = {"POOLED": ROT[1]}
    for name, corruption in entries.items():
        (tmp_path / f"{name}.yaml").write_text(f"entries:\n  - {{obstacle: {IMAGE}, user_module: u, corruption: {corruption}}}\n")
    argv = [str(tmp_path / f"{a}.yaml") if a in entries else a for a in argv]
    if outcome is True:
        with pytest.raises(SystemExit):
            _args(*argv, *COMMON)
    else:
        args = _args(*argv, *COMMON)
        assert {k: getattr(args, k) for k in outcome} == outcome
