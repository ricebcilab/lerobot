"""The DROID statistics conversion: OpenPI's asset to a LeRobot normalizer's override."""

import json

import numpy as np
import pytest

import stats


def _openpi(tmp_path, *, keys=("state", "actions"), width=8, wrap=True):
    block = {
        keys[0]: {"mean": [0.0] * width, "std": [1.0] * width, "q01": [-2.0] * width, "q99": [2.0] * width},
        keys[1]: {"mean": [0.0] * width, "std": [3.0] * width, "q01": [-1.0] * width, "q99": [1.0] * width},
    }
    payload = {"norm_stats": block} if wrap else block
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(payload))
    return path


def test_openpi_export_maps_to_lerobot_features(tmp_path):
    found = stats.read_openpi_stats(_openpi(tmp_path))
    assert set(found) == {stats.STATE, stats.ACTION}
    assert stats.action_width(found) == 8
    np.testing.assert_allclose(found[stats.ACTION]["q01"], np.full(8, -1.0))
    np.testing.assert_allclose(found[stats.ACTION]["std"], np.full(8, 3.0))


def test_a_bare_mapping_reads_and_a_singular_actions_key_reads(tmp_path):
    found = stats.read_openpi_stats(_openpi(tmp_path, keys=("state", "action"), wrap=False))
    assert stats.action_width(found) == 8


def test_missing_quantiles_raise_rather_than_normalize_with_zeros(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": {"state": {"q01": [0.0], "q99": [1.0]}, "actions": {}}}))
    with pytest.raises(ValueError, match="has no q01, q99"):
        stats.read_openpi_stats(path)


def test_missing_a_feature_names_both(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"state": {"q01": [0.0], "q99": [1.0]}}))
    with pytest.raises(ValueError, match="expected statistics for both"):
        stats.read_openpi_stats(path)


def test_overrides_carry_the_statistics_and_the_camera_map(tmp_path):
    found = stats.read_openpi_stats(_openpi(tmp_path))
    pre, post = stats.overrides(found, {"front": "observation.images.base_0_rgb"})
    assert pre["normalizer_processor"]["stats"] is not post["unnormalizer_processor"]["stats"]
    np.testing.assert_allclose(pre["normalizer_processor"]["stats"][stats.ACTION]["std"], np.full(8, 3.0))
    assert pre["rename_observations_processor"]["rename_map"] == {"front": "observation.images.base_0_rgb"}
    assert "rename_observations_processor" not in post
