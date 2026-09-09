import numpy as np
import pytest
from experiment import TrialRecorder, build_schedule, expand_blocks, prompt_for
from teleop import KeyboardReader, TeleopChain


def test_build_schedule_orders():
    assert build_schedule([1, 2, 3], 7, "sequential", 0) == [1, 2, 3, 1, 2, 3, 1]
    shuffled = build_schedule([1, 2, 3], 7, "shuffled", 0)
    assert sorted(shuffled[:3]) == [1, 2, 3] and sorted(shuffled[3:6]) == [1, 2, 3] and len(shuffled) == 7
    random_a = build_schedule([1, 2, 3], 20, "random", 5)
    assert random_a == build_schedule([1, 2, 3], 20, "random", 5)
    assert random_a != build_schedule([1, 2, 3], 20, "random", 6)
    assert set(random_a) <= {1, 2, 3}


def test_prompt_for():
    assert prompt_for("do something", "pick up the bowl") == "do something"
    assert prompt_for("task", "pick up the bowl") == "pick up the bowl"


def test_trial_recorder_rows_and_per_trial_reads(tmp_path):
    kb = KeyboardReader(clock=lambda: 0.0)
    kb.update({"ArrowUp"}, toggles=0)
    chain = TeleopChain(kb)
    chain.source.translation  # noqa: B018 -- a read before the trial must not count
    recorder = TrialRecorder(chain)
    obs = {"robot_state": {"eef": {"pos": np.zeros((1, 3)), "quat": np.zeros((1, 4))}}}
    chain.source.translation  # noqa: B018
    recorder(
        step=1,
        observation=obs,
        action=np.ones((1, 7)),
        reward=np.array([0.0]),
        terminated=np.array([False]),
        truncated=np.array([False]),
        info={},
    )
    recorder(
        step=2,
        observation={},
        action=np.zeros((1, 7)),
        reward=np.array([1.0]),
        terminated=np.array([True]),
        truncated=np.array([False]),
        info={},
    )
    assert recorder.total_reads == 1
    assert [r["user_reads"] for r in recorder.rows] == [1, 0]
    assert np.isnan(recorder.rows[1]["eef_pos"]).all() and recorder.rows[1]["joint_pos"].shape == (7,)
    recorder.save(tmp_path / "t.npz", success=True, task_id=3)
    z = np.load(tmp_path / "t.npz")
    assert z["action"].shape == (2, 7) and bool(z["success"]) and int(z["task_id"]) == 3
    assert z["user_translation_raw"].shape == (2, 3) and z["terminated"].tolist() == [False, True]


def test_expand_blocks_names_and_sets():
    [single] = expand_blocks("flow_control_rotz20", [], None)
    assert single.name == "flow_control_rotz20" and single.sets == []
    [one] = expand_blocks("flow_control_rotz20", ["control.corruption=null"], None)
    assert one.name == "flow_control_rotz20_corruption-null"
    blocks = expand_blocks("flow_reversal_rotz20", ["experiment.seed=1"], "control.n_reversal_steps=2, 4,10")
    assert [b.name for b in blocks] == [
        "flow_reversal_rotz20_seed-1_n_reversal_steps-2",
        "flow_reversal_rotz20_seed-1_n_reversal_steps-4",
        "flow_reversal_rotz20_seed-1_n_reversal_steps-10",
    ]
    assert blocks[0].sets == ["experiment.seed=1", "control.n_reversal_steps=2"]
    with pytest.raises(ValueError, match="--sweep"):
        expand_blocks("x", [], "control.n_reversal_steps")


def test_summarize_scene_motion_picks_the_largest_change():
    from experiment import summarize_scene_motion

    start = {
        "object_names": ["bowl", "plate"],
        "object_pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "articulation_names": ["cabinet_top", "cabinet_middle"],
        "articulation_qpos": np.array([0.0, 0.0]),
    }
    end = {
        "object_names": ["bowl", "plate"],
        "object_pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.03, 0.0]]),
        "articulation_names": ["cabinet_top", "cabinet_middle"],
        "articulation_qpos": np.array([0.0, -0.15]),
    }
    summary = summarize_scene_motion(start, end)
    assert summary["mode"] == "cabinet_middle"
    assert summary["moved"] == pytest.approx({"plate": 0.03, "cabinet_middle": 0.15})
    assert summarize_scene_motion(start, start) == {"mode": None, "moved": {}}
    assert summarize_scene_motion(None, end) == {"mode": None, "moved": {}}


def test_trial_recorder_keeps_scene_rows_and_saves_them(tmp_path):
    from experiment import summarize_scene_motion

    recorder = TrialRecorder(TeleopChain(KeyboardReader(clock=lambda: 0.0)))
    common = {"observation": {}, "action": np.zeros((1, 7)), "reward": np.array([0.0]), "info": {}}

    def scene(x):
        return {
            "object_names": ["bowl"],
            "object_pos": np.array([[x, 0.0, 0.0]], dtype=np.float32),
            "articulation_names": ["drawer"],
            "articulation_qpos": np.array([0.0], dtype=np.float32),
        }

    recorder(step=1, terminated=np.array([False]), truncated=np.array([False]), scene=scene(0.0), **common)
    recorder(step=2, terminated=np.array([False]), truncated=np.array([False]), scene=None, **common)
    recorder(step=3, terminated=np.array([True]), truncated=np.array([False]), scene=scene(0.3), **common)
    assert recorder.scene_first["object_pos"][0, 0] == 0.0 and recorder.scene_last["object_pos"][0, 0] == 0.3
    assert summarize_scene_motion(recorder.scene_first, recorder.scene_last)["mode"] == "bowl"
    recorder.save(tmp_path / "t.npz", success=True)
    z = np.load(tmp_path / "t.npz")
    assert z["object_pos"].shape == (3, 1, 3) and np.isnan(z["object_pos"][1]).all()
    assert list(z["object_names"]) == ["bowl"] and z["articulation_qpos"].shape == (3, 1)
