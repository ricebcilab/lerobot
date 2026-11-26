import numpy as np
import pytest
import torch
from train_visual_task import DemoChunks, trials
from visual_task_scene import segment_distance

from lerobot.policies.steering import AdapterModules, LearnedTaskModule


def head(masked=False, camera_stage="pre"):
    return LearnedTaskModule(
        horizon=4,
        n_dims=3,
        d=16,
        heads=2,
        layers=1,
        token_source="pi05_cameras",
        scene_tokens=8,
        scene_dim=12,
        mask_scene=masked,
        camera_stage=camera_stage,
    )


def test_pi05_time_target_and_mask():
    a, z = torch.randn(2, 4, 32), torch.randn(2, 4, 32)
    times = torch.tensor([0.0, 1.0])
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    v = (z - a).requires_grad_()

    def velocity(x, t):
        torch.testing.assert_close(x[0], a[0])
        torch.testing.assert_close(x[1], z[1])
        torch.testing.assert_close(t, times)
        return v * 1

    task = head()
    patches = torch.randn(2, 8, 12)
    loss = task.pi05_field_loss(velocity, patches, a, z, times, valid, 0.01)
    assert loss.item() == 0
    with torch.no_grad():
        v[~valid] += 10
        v[..., 3:] -= 20
    assert task.pi05_field_loss(velocity, patches, a, z, times, valid, 0.01).item() == 0
    with torch.no_grad():
        v[..., 0] += 1
    assert task.pi05_field_loss(velocity, patches, a, z, times, torch.zeros_like(valid), 0.01).item() == 0
    loss = task.pi05_field_loss(velocity, patches, a, z, times, valid, 0.01)
    torch.testing.assert_close(loss, torch.tensor(1 / 3))
    loss.backward()
    assert v.grad is None and task.out.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(("masked", "stage"), [(False, "pre"), (True, "pre"), (False, "post")])
def test_checkpoint_scene_sensitivity_and_dims(tmp_path, masked, stage):
    torch.manual_seed(0)
    task = head(masked, stage)
    with torch.no_grad():
        task.out.weight.normal_(0, 0.05)
    ckpt = AdapterModules.save_module(task, tmp_path / "task.pt", "task")
    loaded, kind, _ = AdapterModules.load_module(ckpt)
    assert kind == "task" and loaded.camera_stage == stage
    patches = torch.randn(1, 8, 12)
    x, v = torch.randn(1, 4, 32), torch.randn(1, 4, 32)
    obs = {"scene": {"pi05_cameras": patches}}
    holder = AdapterModules(task=loaded, n_dims=None)
    out = holder.field(obs).adapt(x, 0.4, v, False)
    torch.testing.assert_close(out, AdapterModules(task=task, n_dims=None).field(obs).adapt(x, 0.4, v, True))
    torch.testing.assert_close(out[..., 3:], v[..., 3:])
    other = holder.field({"scene": {"pi05_cameras": patches + 3}}).adapt(x, 0.4, v, False)
    assert torch.equal(out, other) == masked


def test_sampling_episode_tail_and_splits(tmp_path):
    path = tmp_path / "demo.npz"
    np.savez(
        path,
        frame_0=np.zeros((2, 4, 4, 3), np.uint8),
        state=np.ones((1, 8), np.float32),
        action=np.ones((2, 7), np.float32),
        stride=10,
    )
    batch, mask = DemoChunks([path], 4).batch(np.random.default_rng(0), 1)
    assert batch["action"].shape == (1, 4, 7) and mask.tolist() == [[True, True, False, False]]

    class States:
        @staticmethod
        def initial_state_count(suite, task):
            return 50

    splits = [
        {(t.task_id, t.init_state_id) for t, _, _ in trials(States, name, 50)}
        for name in ("train", "dev", "test")
    ]
    assert list(map(len, splits)) == [40, 20, 40]
    assert len(set.union(*splits)) == 100
    assert segment_distance(np.array([-1, 0]), np.array([1, 0]), np.zeros(2)) == 0


def test_terminal_crossing_disqualifies_success():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from visual_task_scene import PilotScene

    scene = PilotScene.__new__(PilotScene)
    scene.env = SimpleNamespace(
        _env=SimpleNamespace(step=Mock(return_value=({}, 1, True, {})), check_success=lambda: True)
    )
    scene.position = Mock(return_value=np.array([0.04, 0, 0.9]))
    scene.observation = lambda raw: raw
    scene.obstacle, scene.control_step = "disk", 0
    scene.violation = False
    scene.previous, scene.center = np.array([-0.04, 0]), np.zeros(2)
    _, done, success = scene.step(np.zeros(7))
    assert done and success and scene.violation


def test_report_uses_matched_layouts_and_counts_training_seeds_once_per_arm(tmp_path):
    import json

    from visual_task_report import report

    (tmp_path / "manifest.json").write_text(json.dumps({"states": 1}))
    ckpt = tmp_path / "checkpoints/visual_s0/task.pt"
    AdapterModules.save_module(head(), ckpt, "task", {"parameters": 42, "policy": "pi05", "revision": "test"})
    for name in ("frs", "masked_s0", "visual_s0", "mismatch_s0"):
        directory = tmp_path / "test" / name
        directory.mkdir(parents=True)
        for task in (1, 8):
            placed = name == "visual_s0" or task == 1
            row = {
                "task_id": task,
                "init_state_id": 7,
                "pair_id": str(task),
                "layout_seed": 42,
                "initial_state_hash": str(task),
                "center": [0, 0],
                "success": placed,
                "violation": False,
                "joint": placed,
                "blocked": task == 1,
                "replan_ms": 5,
            }
            (directory / f"{task}.json").write_text(json.dumps(row))
    report(tmp_path, [0])
    result = (tmp_path / "report.md").read_text()
    assert "| visual | 100.0% | 0.0% | 100.0% |" in result and "Visual minus frs: +50.0" in result
    path = tmp_path / "test/mismatch_s0/1.json"
    row = json.loads(path.read_text())
    row["center"] = [0.1, 0.1]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="unmatched"):
        report(tmp_path, [0])


def test_holder_has_solid_crock_and_utensils():
    import xml.etree.ElementTree as ET

    from visual_task_scene import holder_xml

    body = ET.fromstring(holder_xml("<mujoco><worldbody/></mujoco>")).find("worldbody/body")
    geoms = body.findall("geom")
    assert body.get("mocap") == "true"
    assert {"holder_cup", "holder_turner", "holder_spoon"} <= {g.get("name") for g in geoms}
    assert all(g.get("contype") == g.get("conaffinity") == "1" for g in geoms)


def test_failure_category_distinguishes_pick_from_placement():
    from visual_task_report import failure_category

    row = {"success": False, "violation": True, "picked": ["cream_cheese_1"], "bowl_final_on": []}
    assert failure_category(row) == "other object only"
    row["picked"].append("akita_black_bowl_1")
    assert failure_category(row) == "bowl lifted, missed goal"
    assert failure_category({**row, "bowl_final_on": ["plate"]}) == "bowl on other goal"
    assert failure_category({**row, "success": True}) == "placed with violation"


def test_gripper_projection_offsets_only_external_patches(tmp_path):
    cfg = {
        "horizon": 4,
        "n_dims": 3,
        "d": 8,
        "heads": 2,
        "layers": 1,
        "token_source": "pi05_cameras",
        "scene_tokens": 512,
        "scene_dim": 4,
        "camera_stage": "post",
    }
    # (x,y,1) projects to (x,y), with the same normalized units as the 16x16 patch grid.
    torch.manual_seed(3)
    base = LearnedTaskModule(**cfg)
    torch.manual_seed(3)
    task = LearnedTaskModule(**cfg, gripper_projection=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    patches = {"scene": {"pi05_cameras": torch.randn(2, 512, 4)}}
    obs = {**patches, "gripper_xyz": torch.tensor([[0.5, 0.25, 0], [0.4, 0.25, 0]])}
    ref = base.tokens(obs, 2)[0]
    torch.testing.assert_close(task.tokens(obs, 2)[0], ref, atol=0, rtol=0)
    with torch.no_grad():
        task.offset_weight[:, :2].copy_(torch.eye(2))
    path = AdapterModules.save_module(task, tmp_path / "geometry.pt", "task")
    loaded, _, _ = AdapterModules.load_module(path)
    delta = loaded.tokens(obs, 2)[0] - ref
    torch.testing.assert_close(delta[:, 257:], torch.zeros_like(delta[:, 257:]))
    for k in (0, 15, 16, 255):
        center = torch.tensor([k % 16 + 0.5, k // 16 + 0.5]) / 16
        torch.testing.assert_close(delta[:, k + 1, :2], center - obs["gripper_xyz"][:, :2])
    with pytest.raises(ValueError, match="gripper XYZ"):
        loaded.tokens(patches, 2)


def test_mixed_batch_keeps_teacher_suffix_aligned(tmp_path):
    def dataset(name, values):
        path = tmp_path / (name + ".npz")
        starts = values[::2]
        np.savez(
            path,
            state=np.array(starts)[:, None],
            action=np.array(values)[:, None],
            stride=2,
            **{f"frame_{i}": np.full((2, 2, 2, 3), value, dtype=np.uint8) for i, value in enumerate(starts)},
        )
        return DemoChunks([path], 4, "avoid")

    original = dataset("original", [5, 6, 7])
    recovery = dataset("recovery", [50, 51, 52, 53, 54])
    for seed in range(8):
        batch, valid = original.batch(np.random.default_rng(seed), 2, recovery)
        starts = batch["observation.state"][:, 0]
        assert starts[0] < 10 and starts[1] >= 50
        torch.testing.assert_close(batch["observation.images.image"][:, 0, 0, 0] * 255, starts.float())
        for i, end in enumerate([7, 54]):
            expected = torch.arange(starts[i], min(int(starts[i]) + 4, end + 1)).float()
            torch.testing.assert_close(batch["action"][i, valid[i], 0], expected)


def test_two_obstacles_score_either_contact_order():
    from types import SimpleNamespace
    from xml.etree import ElementTree

    from visual_task_scene import PilotScene, two_obstacle_xml

    root = ElementTree.fromstring(two_obstacle_xml("<mujoco><worldbody/></mujoco>"))
    for name in ("mars_marker", "mars_pepper_mill"):
        geoms = root.findall(f"worldbody/body[@name='{name}']/geom")
        assert geoms and all(g.get("contype") == "1" for g in geoms)

    scene = PilotScene.__new__(PilotScene)
    scene.obstacle_geoms = {4: "utensil_holder", 7: "pepper_mill"}
    contacts = [
        SimpleNamespace(geom1=a, geom2=b, dist=d)
        for a, b, d in [(4, 11, 0.0), (12, 7, -0.001), (4, 12, 0.01), (11, 12, -0.001)]
    ]
    scene.domain = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(contact=contacts, ncon=4)))
    assert list(scene.obstacle_contacts()) == [(4, 11), (7, 12)]
