"""The report's statistics: Wilson's interval, the paired sign test, and the measured authority.

All inline and all offline: the run directories are written by the test, and the only dependency is numpy.
"""

import json
import math

import numpy as np
import pytest
import yaml
from report import load_run, main, sign_test, trial_authority, wilson

ARM = "mars_fixed"


def _npz(path, *, authority=0.5, steps=2, pushing=True):
    """A step file whose executed action sits `authority` of the way from the plan to the command."""
    plan = np.zeros((steps, 3))
    command = np.tile([1.0, 0.0, 0.0], (steps, 1))  # served command, in [-1, 1]
    action = np.zeros((steps, 8))
    action[:, :3] = authority * command * 0.5  # qdot_full = 0.5
    raw = np.tile([1.0, 0.0, 0.0], (steps, 1)) if pushing else np.zeros((steps, 3))
    np.savez_compressed(
        path,
        action=action,
        policy_translation=plan,
        user_translation=command,
        user_translation_raw=raw,
        joint_pos=np.zeros((steps, 7)),
        user_gripper=np.zeros(steps),
        user_reads=np.ones(steps),
        t=np.arange(steps, dtype=np.float64),
        success=np.asarray(False),
    )


def _write_run(root, arm, trials, *, mode="shared_flow_reversal_steering"):
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "arm": arm,
                "mode": mode,
                "qdot_full": 0.5,
                "tau": 14,
                "lambda": 1,
                "n_reversal_steps": 5,
            }
        )
    )
    (root / "trials.jsonl").write_text("".join(json.dumps(trial) + "\n" for trial in trials))
    for trial in trials:
        _npz(root / trial["steps_file"])
    return root


def _trial(i, success, pair_id, state_hash="same"):
    return {
        "trial": i,
        "pair_id": pair_id,
        "initial_state_hash": state_hash,
        "arm": ARM,
        "success": success,
        "duration_s": 10.0 + i,
        "steps_file": f"trial_{i:03d}.npz",
    }


# ---------------------------------------------------------------- the statistics


def test_wilson_is_nan_with_no_trials():
    assert all(math.isnan(value) for value in wilson(0, 0))


def test_wilson_at_ten_of_ten_is_near_one():
    low, high = wilson(10, 10)
    assert high == pytest.approx(1.0)
    assert low > 0.7
    low, high = wilson(0, 10)
    assert low == 0.0 and high < 0.3


def test_sign_test_counts_the_discordant_pairs():
    assert sign_test(0, 0) != sign_test(0, 0)  # NaN: no discordant pair
    assert sign_test(5, 0) == pytest.approx(2 / 2**5)
    assert sign_test(3, 3) == pytest.approx(1.0)
    assert 0.0 <= sign_test(4, 1) <= 1.0


def test_trial_authority_is_the_barycentric_coordinate_over_pushing_steps(tmp_path):
    path = tmp_path / "trial_000.npz"
    _npz(path, authority=0.5)
    assert trial_authority(path, qdot_full=0.5) == pytest.approx(0.5)
    _npz(path, authority=0.5, pushing=False)
    assert math.isnan(trial_authority(path, qdot_full=0.5))


# ---------------------------------------------------------------- the report


def test_the_report_prints_one_line_per_arm(tmp_path, capsys):
    first = _write_run(tmp_path / "a", ARM, [_trial(0, True, "p0"), _trial(1, False, "p1")])
    second = _write_run(tmp_path / "b", "frs", [_trial(0, True, "p0"), _trial(1, True, "p1")])

    assert load_run(first)[1][0]["arm"] == ARM
    main([str(first), str(second)])
    out = capsys.readouterr().out
    for arm in (ARM, "frs"):
        lines = [line for line in out.splitlines() if line.startswith(arm)]
        assert len(lines) == 1, (arm, out)
    assert "paired" in out


def test_the_report_skips_pairs_whose_resets_differ(tmp_path, capsys):
    first = _write_run(tmp_path / "a", ARM, [_trial(0, True, "p0", "one")])
    second = _write_run(tmp_path / "b", "frs", [_trial(0, True, "p0", "two")])
    main([str(first), str(second)])
    out = capsys.readouterr().out
    assert "0 matched" in out
