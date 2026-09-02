import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("scipy")
NOTEBOOKS = Path(__file__).resolve().parents[3] / "examples/pi05/libero_shared_autonomy/notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
from analyze import (  # noqa: E402
    compare,
    describe_rotation,
    find_runs,
    input_activity,
    label_for,
    load_runs,
    paired_comparisons,
    success_table,
    wilson,
)


def make_run(root: Path, name: str, mode: str, outcomes: list[bool], corruption=None):
    run = root / name
    run.mkdir(parents=True)
    config = {
        "mode": mode,
        "tau": 8,
        "corruption_matrix": corruption,
        "flow_adapter_matrix": None,
        "schedule": [0, 1, 0],
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    with open(run / "trials.jsonl", "w") as f:
        for i, ok in enumerate(outcomes):
            f.write(
                json.dumps(
                    {
                        "trial": i,
                        "task_id": i % 2,
                        "task_description": f"task {i % 2}",
                        "success": ok,
                        "steps": 100 + i,
                        "duration_s": 5.0,
                        "steps_file": f"trial_{i:03d}.npz",
                    }
                )
                + "\n"
            )
            raw = np.zeros((3, 3), dtype=np.float32)
            raw[1] = [0.5, 0.0, 0.0]
            np.savez(
                run / f"trial_{i:03d}.npz",
                user_translation_raw=raw,
                user_translation=raw * 0.9,
                user_gripper=np.array([-1.0, -1.0, 1.0]),
                user_reads=np.array([1, 0, 1]),
            )
    return run


def test_end_to_end_on_synthetic_runs(tmp_path):
    rot = [[0.94, -0.34, 0.0], [0.34, 0.94, 0.0], [0.0, 0.0, 1.0]]
    a = make_run(tmp_path, "20260101_000000_flow_control", "shared_flow_control", [True, False, True])
    b = make_run(
        tmp_path, "20260101_000001_reverse_flow", "shared_reverse_flow_steering", [True, True, True], rot
    )
    (tmp_path / "20260101_000002_aborted").mkdir()
    assert find_runs(tmp_path) == [a, b]

    trials, configs = load_runs([a, b])
    assert len(trials) == 6 and set(trials["run"]) == {a.name, b.name}
    assert label_for(configs[a.name]) == "FC tau=8" and label_for(configs[b.name]) == "RFS+M"

    table = success_table(trials, configs)
    assert table.loc[a.name, "successes"] == 2 and table.loc[b.name, "success_rate"] == 1.0
    low, high = wilson(2, 3)
    assert 0 <= low < 2 / 3 < high <= 1
    assert wilson(0, 0) == (pytest.approx(np.nan, nan_ok=True), pytest.approx(np.nan, nan_ok=True))

    result = compare(trials, a.name, b.name)
    assert result["paired_trials"] == 3 and result["B_only"] == 1 and result["A_only"] == 0
    assert 0 < result["mcnemar_p"] <= 1
    assert len(paired_comparisons(trials)) == 1

    activity = input_activity(trials)
    assert activity["commanding_frac"].iloc[0] == pytest.approx(1 / 3)
    assert activity["reads_this_trial"].iloc[0] == 2

    assert "rotation about z" in describe_rotation(np.array(rot))
    assert describe_rotation(np.eye(3)) == "identity (no-op)"
