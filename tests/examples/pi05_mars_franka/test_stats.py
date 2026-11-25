"""The bundle's statistics: openpi's DROID quantiles installed on the checkpoint's QUANTILES steps.

pi0.5-DROID's published processors declare no features and no statistics: the q01/q99 of openpi's
``assets/droid/norm_stats.json`` are what makes the checkpoint's QUANTILES normalization mean anything,
and ``stats.install_stats`` is the one place they are put on the steps (``stats.py``'s docstring cites
the ``normalize_processor.py`` lines the install follows). The real asset is gitignored, so these tests
build the same 32-wide structure (dims 0-7 meaningful, zero after) in ``tmp_path`` and a second test runs
on the real file when a checkout has it.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stats import CHECKPOINT_WIDTH, STATION_DIMS, install_stats

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.types import TransitionKey

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"
REAL_STATS = EXAMPLE_DIR / "assets" / "droid" / "norm_stats.json"
"""openpi's droid statistics, gitignored: present in a checkout that downloaded the assets."""

# The synthetic statistics: the same layout as openpi's file, with numbers a transposed or mean/std-based
# install cannot reproduce. The gripper dim (7) has q01 = 0 and q99 != 0, as the real file's does.
STATE_Q01 = np.array([-0.5, -0.7, -0.4, -0.9, -0.6, -0.6, -0.7, 0.0], dtype=np.float64)
STATE_Q99 = np.array([0.5, 0.8, 0.4, 0.8, 0.6, 0.7, 0.7, 1.0], dtype=np.float64)
ACTIONS_Q01 = 0.5 * STATE_Q01
ACTIONS_Q99 = 0.5 * STATE_Q99
FILE_MEAN = np.full(CHECKPOINT_WIDTH, 1.0)  # deliberately not the quantile midpoint
FILE_STD = np.full(CHECKPOINT_WIDTH, 2.0)  # deliberately not the half range


def _padded(values: np.ndarray, meaningful: int, width: int) -> np.ndarray:
    """``values`` in the first ``meaningful`` dims, zeros after, ``width`` wide."""
    padded = np.zeros(width, dtype=np.float64)
    known = min(meaningful, values.shape[0])
    padded[:known] = values[:known]
    for index in range(known, meaningful):
        padded[index] = 0.1 * (index + 1)  # still meaningful, but not q01 == q99 == 0
    return padded


def _stats_file(path: Path, *, meaningful: int = STATION_DIMS, width: int = CHECKPOINT_WIDTH) -> Path:
    """A hermetic ``norm_stats.json``: state and actions groups in openpi's structure."""
    payload = {
        "norm_stats": {
            "state": {
                "mean": FILE_MEAN[:width].tolist(),
                "std": FILE_STD[:width].tolist(),
                "q01": _padded(STATE_Q01, meaningful, width).tolist(),
                "q99": _padded(STATE_Q99, meaningful, width).tolist(),
            },
            "actions": {
                "mean": FILE_MEAN[:width].tolist(),
                "std": FILE_STD[:width].tolist(),
                "q01": _padded(ACTIONS_Q01, meaningful, width).tolist(),
                "q99": _padded(ACTIONS_Q99, meaningful, width).tolist(),
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Pipeline:
    """What install_stats reads of a policy processor: its steps (a PolicyProcessorPipeline does the same)."""

    def __init__(self, *steps):
        self.steps = list(steps)


def _real_steps() -> tuple[NormalizerProcessorStep, UnnormalizerProcessorStep]:
    """A real normalizer and unnormalizer, constructed directly (no tokenizer needed for either).

    The configs are the cached checkpoint JSONs' (``features: {}`` and this normalization_mapping)."""
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.QUANTILES,
        FeatureType.ACTION: NormalizationMode.QUANTILES,
    }
    return (
        NormalizerProcessorStep(features={}, norm_map=dict(norm_map)),
        UnnormalizerProcessorStep(features={}, norm_map=dict(norm_map)),
    )


def _installed(tmp_path: Path):
    normalizer, unnormalizer = _real_steps()
    mean, std = install_stats(
        _Pipeline(normalizer), _Pipeline(unnormalizer), _stats_file(tmp_path / "stats.json")
    )
    return normalizer, unnormalizer, mean, std


def test_install_stats_installs_quantiles_where_the_quantiles_map_reads_them(tmp_path):
    """The checkpoint's normalization_mapping is QUANTILES for STATE and ACTION, so the steps read q01/q99
    of the two keys; the 32-wide arrays (dims 0-7 meaningful) go on the pre-processor's normalizer and the
    post-processor's unnormalizer, together with the feature entries the normalizer needs to see the
    state (the cached JSONs ship ``features: {}``)."""
    normalizer, unnormalizer, mean, std = _installed(tmp_path)

    np.testing.assert_array_equal(normalizer.stats["observation.state"]["q01"], _padded(STATE_Q01, 8, 32))
    np.testing.assert_array_equal(normalizer.stats["observation.state"]["q99"], _padded(STATE_Q99, 8, 32))
    np.testing.assert_array_equal(normalizer.stats["action"]["q01"], _padded(ACTIONS_Q01, 8, 32))
    np.testing.assert_array_equal(unnormalizer.stats["action"]["q01"], _padded(ACTIONS_Q01, 8, 32))
    np.testing.assert_array_equal(unnormalizer.stats["action"]["q99"], _padded(ACTIONS_Q99, 8, 32))

    state_feature = normalizer.features["observation.state"]
    assert state_feature.type is FeatureType.STATE and state_feature.shape == (CHECKPOINT_WIDTH,)
    assert unnormalizer.features["action"].type is FeatureType.ACTION
    assert normalizer.norm_map[FeatureType.ACTION] is NormalizationMode.QUANTILES

    # the tensor statistics were rebuilt from the installed dict, so the transform reads them
    assert "q01" in normalizer._tensor_stats["observation.state"]
    assert "q01" in unnormalizer._tensor_stats["action"]

    # the returned action statistics are the affine form of the QUANTILES map, 8 wide
    assert mean.shape == (STATION_DIMS,) and std.shape == (STATION_DIMS,)
    np.testing.assert_allclose(mean, (ACTIONS_Q01[:8] + ACTIONS_Q99[:8]) / 2)
    np.testing.assert_allclose(std, (ACTIONS_Q99[:8] - ACTIONS_Q01[:8]) / 2)
    assert not np.allclose(mean, FILE_MEAN[:8])  # the file's mean/std are not what QUANTILES reads


def test_an_eight_wide_action_round_trips_through_the_real_steps(tmp_path):
    """The model's action is 32 wide (dims 0-7 real), so the realistic action is that padded vector: its
    first 8 dims normalize to the quantile map and unnormalize back to the raw action units, and the pad
    dims (q01 == q99 == 0) round trip through the eps-denominator branch to 0."""
    normalizer, unnormalizer, _, _ = _installed(tmp_path)
    action = np.array([0.05, -0.2, 0.1, 0.0, -0.05, 0.2, 0.3, 1.0])
    raw = np.zeros(CHECKPOINT_WIDTH, dtype=np.float32)
    raw[:STATION_DIMS] = action

    normalized = normalizer({TransitionKey.ACTION: torch.as_tensor(raw[None])})[TransitionKey.ACTION]
    q01 = torch.as_tensor(_padded(ACTIONS_Q01, 8, 32), dtype=torch.float32)
    q99 = torch.as_tensor(_padded(ACTIONS_Q99, 8, 32), dtype=torch.float32)
    expected = 2.0 * (torch.as_tensor(raw[None]) - q01) / (q99 - q01) - 1.0
    np.testing.assert_allclose(
        normalized[0, :STATION_DIMS].numpy(), expected[0, :STATION_DIMS].numpy(), atol=1e-6
    )

    restored = unnormalizer({TransitionKey.ACTION: normalized})[TransitionKey.ACTION]
    np.testing.assert_allclose(restored[0, :STATION_DIMS].numpy(), action, atol=1e-6)
    np.testing.assert_allclose(restored[0, STATION_DIMS:].numpy(), 0.0, atol=1e-6)


def test_a_32_wide_state_normalizes_and_its_pad_statistics_stay_zero(tmp_path):
    """The station sends 32 wide with zeros beyond dim 7 (DESIGN); the installed statistics keep those pad
    dims exactly zero, and a padded state's normalized pad dims round trip to 0 (forward they are the
    mixin's eps-denominator branch's -1, as openpi's zero-range quantile map is)."""
    normalizer, unnormalizer, _, _ = _installed(tmp_path)
    state = np.zeros(CHECKPOINT_WIDTH, dtype=np.float32)
    state[:STATION_DIMS] = np.linspace(-1.0, 1.0, STATION_DIMS)

    normalized = normalizer({TransitionKey.OBSERVATION: {"observation.state": torch.as_tensor(state[None])}})[
        TransitionKey.OBSERVATION
    ]["observation.state"]
    q01 = torch.as_tensor(_padded(STATE_Q01, 8, 32), dtype=torch.float32)
    q99 = torch.as_tensor(_padded(STATE_Q99, 8, 32), dtype=torch.float32)
    expected = 2.0 * (torch.as_tensor(state[None]) - q01) / (q99 - q01) - 1.0
    np.testing.assert_allclose(
        normalized[0, :STATION_DIMS].numpy(), expected[0, :STATION_DIMS].numpy(), atol=1e-6
    )
    assert torch.isfinite(normalized).all()

    # the pad statistics are the file's zeros, not fabricated values
    assert np.all(np.asarray(normalizer.stats["observation.state"]["q01"])[STATION_DIMS:] == 0.0)
    assert np.all(np.asarray(normalizer.stats["observation.state"]["q99"])[STATION_DIMS:] == 0.0)
    restored = unnormalizer({TransitionKey.OBSERVATION: {"observation.state": normalized}})[
        TransitionKey.OBSERVATION
    ]["observation.state"]
    np.testing.assert_allclose(restored[0, STATION_DIMS:].numpy(), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("meaningful", "width"),
    [(STATION_DIMS - 1, CHECKPOINT_WIDTH), (STATION_DIMS + 1, CHECKPOINT_WIDTH), (STATION_DIMS, 31)],
    ids=["seven-meaningful", "nine-meaningful", "not-32-wide"],
)
def test_install_stats_refuses_a_width_that_is_not_the_checkpoints(tmp_path, meaningful, width):
    """DESIGN: the file is meaningful in 8 dims and 32 wide. A file that disagrees would silently
    normalize a different action space."""
    normalizer, unnormalizer = _real_steps()
    path = _stats_file(tmp_path / "stats.json", meaningful=meaningful, width=width)
    with pytest.raises(ValueError, match="norm_stats"):
        install_stats(_Pipeline(normalizer), _Pipeline(unnormalizer), path)


def test_install_stats_requires_the_steps_it_installs_on(tmp_path):
    """A pipeline without the normalizer/unnormalizer steps cannot be normalized at all: say so."""
    path = _stats_file(tmp_path / "stats.json")
    with pytest.raises(ValueError, match="NormalizerProcessorStep"):
        install_stats(_Pipeline(), _Pipeline(*_real_steps()[1:]), path)
    with pytest.raises(ValueError, match="UnnormalizerProcessorStep"):
        install_stats(_Pipeline(_real_steps()[0]), _Pipeline(), path)


@pytest.mark.skipif(
    not REAL_STATS.is_file(),
    reason="openpi's assets/droid/norm_stats.json is not in this checkout (gitignored; see README)",
)
def test_install_stats_on_the_real_openpi_asset():
    """The real file: 32 wide, meaningful in 8 dims, and the returned mean/std are its action quantiles'
    affine form. This runs wherever the assets were downloaded; the synthetic tests above cover the rest."""
    normalizer, unnormalizer = _real_steps()
    mean, std = install_stats(_Pipeline(normalizer), _Pipeline(unnormalizer), REAL_STATS)
    actions = json.loads(REAL_STATS.read_text(encoding="utf-8"))["norm_stats"]["actions"]
    q01 = np.asarray(actions["q01"], dtype=np.float64)
    q99 = np.asarray(actions["q99"], dtype=np.float64)
    assert q01.shape == (CHECKPOINT_WIDTH,) and q99.shape == (CHECKPOINT_WIDTH,)
    assert np.all(q01[STATION_DIMS:] == 0.0) and np.all(q99[STATION_DIMS:] == 0.0)
    np.testing.assert_allclose(mean, (q01[:STATION_DIMS] + q99[:STATION_DIMS]) / 2)
    np.testing.assert_allclose(std, (q99[:STATION_DIMS] - q01[:STATION_DIMS]) / 2)
    assert np.all(np.asarray(normalizer.stats["observation.state"]["q01"])[STATION_DIMS:] == 0.0)
