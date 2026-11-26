from types import SimpleNamespace

import numpy as np
import pytest
import torch
from holder_intervention_train import prefix
from holder_interventions import lateral, perturb

from lerobot.policies.steering.modeling_steering import (
    AdapterModules,
    FlowReversalSteering,
    LearnedTaskModule,
)


def test_xyz_branches_keep_orientation_and_gripper():
    actions = np.tile(np.array([0, 0, 0.9, 0.1, -0.2, 0.3, 1.0]), (10, 1))
    scene = SimpleNamespace(start=np.zeros(3), goal=np.array([1.0, 0, 0]))
    left = lateral(actions, scene)
    np.testing.assert_allclose(left, [0, 1, 0])
    updated = perturb(actions, left, 0.15)
    np.testing.assert_allclose(updated[:, 1], 0.15)
    np.testing.assert_array_equal(actions[:, 1], 0)
    np.testing.assert_array_equal(updated[:, 3:], actions[:, 3:])
    np.testing.assert_array_equal(perturb(actions, np.array([0, 0, 1]), 0.3)[:, 2], 1)


@pytest.mark.parametrize("command", [[0.3, -0.4, 0.2], [0.0, 0.0, 0.0]])
def test_prefix_matches_runtime_equations_and_backpropagates(command):
    torch.manual_seed(42)
    cfg = SimpleNamespace(
        num_inference_steps=10, chunk_size=12, max_action_dim=8, n_action_steps=10, device="cpu"
    )
    task = LearnedTaskModule(
        horizon=12,
        n_dims=3,
        d=8,
        tdim=8,
        heads=2,
        layers=1,
        token_source="pi05_cameras",
        scene_dim=5,
        scene_tokens=4,
        camera_stage="post",
    )
    patches = torch.randn(1, 4, 5)

    def vel(x, t):
        return x * 0.17 + 0.3 * t

    holder = AdapterModules(task=task, n_dims=None)
    core = SimpleNamespace(
        n_reversal_steps=4,
        _delegated=FlowReversalSteering.delegation_mask(12, 8, ("translation",), 10),
        _active=lambda c: np.max(np.abs(c), axis=1) >= 0.05,
        reference_chunk=lambda cmd: torch.tensor(np.pad(cmd, ((0, 0), (0, 5))), dtype=torch.float32)[
            :, None
        ].expand(-1, 12, -1),
    )
    s = SimpleNamespace(
        policy=SimpleNamespace(config=cfg, camera_field=lambda batch, stage: (vel, patches)),
        adapter=holder,
        steering_policy=lambda: SimpleNamespace(core=core),
    )
    data = {"command": np.array(command), "noise": torch.randn(12, 8).numpy(), "gripper_ref": -1.0}
    result = prefix(s, {}, data)
    with torch.no_grad():
        field = holder.field({"scene": {"pi05_cameras": patches}})

        def advance(x, time, dt, n, rev=False):
            for i in range(n):
                t = time + dt * i
                x = x + dt * field.adapt(x, t, vel(x, t), rev)
            return x

        noise = torch.from_numpy(data["noise"])[None]
        if any(command):
            ref = core.reference_chunk(data["command"][None])
            latent = torch.where(
                core._delegated, advance(noise, 1.0, -0.1, 6), advance(ref, 0.0, 0.1, 4, True)
            )
            expected = advance(latent, 0.4, -0.1, 4)
        else:
            expected = advance(noise, 1.0, -0.1, 10)
    torch.testing.assert_close(result, expected[:, :10, :3])
    result.square().mean().backward()
    assert torch.isfinite(task.out.weight.grad).all() and task.out.weight.grad.norm() > 0


def test_contact_probe_scores_only_events_in_executed_prefix(tmp_path, monkeypatch):
    import json

    import holder_contact_probe as probe

    directory = tmp_path / "pairs"
    directory.mkdir()
    row = {
        "init_state_id": 17,
        "rescue": True,
        "contact": {"step": 29, "with": "robot"},
        "branches": [
            {
                "start": 20,
                "joint": False,
                "direction": "up",
                "magnitude": 0.15,
                "contact": {"with": "bowl", "step": 30},
            },
            {"start": 20, "joint": True},
        ],
    }
    (directory / "case.json").write_text(json.dumps(row))
    original = np.zeros((10, 7), dtype=np.float32)
    np.savez(directory / "case.npz", original=original, action=original + 0.1)
    monkeypatch.setattr(probe, "encode", lambda *args: (torch.zeros(32, 64), torch.zeros(8)))
    session = SimpleNamespace(steering_policy=lambda: SimpleNamespace(_normalized_chunk=lambda a: a))
    (r,) = probe.build_dataset(session, tmp_path)
    assert r["labels"].tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert r["actions"].shape == (3, 10, 3)
