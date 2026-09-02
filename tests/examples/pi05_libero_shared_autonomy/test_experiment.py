import numpy as np
from experiment import TrialRecorder, build_schedule, prompt_for
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
    chain.reader.translation  # noqa: B018 -- a read before the trial must not count
    recorder = TrialRecorder(chain)
    obs = {"robot_state": {"eef": {"pos": np.zeros((1, 3)), "quat": np.zeros((1, 4))}}}
    chain.reader.translation  # noqa: B018
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
