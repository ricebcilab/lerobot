"""The states file: the frozen format's round trip, its meta, and every refusal that names its reason.

Hermetic and offline: a synthetic batch in the shape ``station.observation_batch`` returns, known
vectors, and ``numpy``'s own archive. The format's dtypes and shapes are asserted literally because a
trainer written against DESIGN.md's table must be able to trust the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bridge import ARM_DIMS, TWIST_DIMS
from statefiles import FORMAT_VERSION, Recorder, load
from station import IMAGE_SIZE

JOINT = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
Q0 = JOINT + 0.01
GRIPPER = np.array([1.0])
DEFLECTION = np.array([0.2, -0.1, 0.3, 0.05, -0.05, 0.1])
SERVED = np.linspace(-0.4, 0.6, ARM_DIMS)
PROMPT = "pick the cube"
META = {
    "config": "/tmp/franka.toml",
    "config_sha256": "ab" * 32,
    "policy": "lerobot/pi05_droid",
    "stats": "/tmp/norm_stats.json",
    "station": {"host": "10.42.0.1", "port": 9090, "arm": "follower", "rate": 15.0},
    "corruption": {"rotation_z_deg": 20},
    "timestamp": "2026-11-24T23:54:40+00:00",
}

ARRAY_KEYS = {
    "image_exterior",
    "image_wrist",
    "joint_position",
    "gripper",
    "deflection",
    "served",
    "q0",
    "episode",
    "tick",
    "prompt",
    "mode",
}


# ---------------------------------------------------------------- the fakes


def _images(offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Model-ready batch images whose values are the byte pattern ``k/255``: the round trip is exact."""
    pattern = (np.arange(3 * IMAGE_SIZE * IMAGE_SIZE) + offset) % 256
    exterior = (pattern / 255.0).astype(np.float32).reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    wrist = (np.flip(pattern) / 255.0).astype(np.float32).reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    return exterior, wrist


def _batch(offset: int = 0) -> dict:
    exterior, wrist = _images(offset)
    return {
        "observation.images.base_0_rgb": exterior,
        "observation.images.left_wrist_0_rgb": wrist,
        "observation.state": np.zeros(32, dtype=np.float32),
        "task": PROMPT,
    }


def _add(
    recorder: Recorder,
    *,
    batch: dict | None = None,
    joint_position: np.ndarray = JOINT,
    gripper: np.ndarray = GRIPPER,
    deflection: np.ndarray = DEFLECTION,
    served: np.ndarray = SERVED,
    q0: np.ndarray = Q0,
    episode: int = 0,
    tick: int = 0,
    prompt: str = PROMPT,
    mode: str = "steered",
) -> None:
    recorder.add(
        _batch() if batch is None else batch,
        joint_position=joint_position,
        gripper=gripper,
        deflection=deflection,
        served=served,
        q0=q0,
        episode=episode,
        tick=tick,
        prompt=prompt,
        mode=mode,
    )


def _filled() -> Recorder:
    """Two states: the first at episode 0 tick 0, the second at episode 2 tick 4."""
    recorder = Recorder(META)
    _add(recorder)
    _add(
        recorder,
        batch=_batch(offset=7),
        joint_position=JOINT + 1.0,
        gripper=np.zeros(1),
        deflection=-DEFLECTION,
        served=-SERVED,
        episode=2,
        tick=4,
        prompt="stack the blocks",
        mode="teleop",
    )
    return recorder


def _write(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> Path:
    """Re-save a hand-built archive, so a test can break exactly one thing."""
    values = {**arrays, "meta": np.array(json.dumps(meta), dtype=np.str_)}
    np.savez(path, **values)
    return path


def _valid(tmp_path: Path) -> Path:
    return _filled().save(tmp_path / "states.npz")


# ---------------------------------------------------------------- the round trip


def test_the_round_trip_returns_every_array_with_its_dtype_shape_and_values(tmp_path):
    recorder = _filled()
    assert len(recorder) == 2

    arrays, meta = load(recorder.save(tmp_path / "states.npz"))

    assert set(arrays) == ARRAY_KEYS
    assert meta == {**META, "format_version": FORMAT_VERSION}
    n = 2
    expected = {
        "image_exterior": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "image_wrist": (n, 3, IMAGE_SIZE, IMAGE_SIZE),
        "joint_position": (n, ARM_DIMS),
        "gripper": (n, 1),
        "deflection": (n, TWIST_DIMS),
        "served": (n, ARM_DIMS),
        "q0": (ARM_DIMS,),
        "episode": (n,),
        "tick": (n,),
        "prompt": (n,),
        "mode": (n,),
    }
    for name, shape in expected.items():
        assert arrays[name].shape == shape, name
    assert arrays["image_exterior"].dtype == np.uint8
    assert arrays["image_wrist"].dtype == np.uint8
    for name in ("joint_position", "gripper", "deflection", "served", "q0"):
        assert arrays[name].dtype == np.float64, name
    assert arrays["episode"].dtype == np.int32 and arrays["tick"].dtype == np.int32
    assert arrays["prompt"].dtype.kind == "U" and arrays["mode"].dtype.kind == "U"

    exterior_one, wrist_one = _images()
    exterior_two, wrist_two = _images(offset=7)
    np.testing.assert_array_equal(arrays["image_exterior"][0], np.rint(exterior_one * 255.0))
    np.testing.assert_array_equal(arrays["image_wrist"][0], np.rint(wrist_one * 255.0))
    np.testing.assert_array_equal(arrays["image_exterior"][1], np.rint(exterior_two * 255.0))
    np.testing.assert_array_equal(arrays["image_wrist"][1], np.rint(wrist_two * 255.0))
    np.testing.assert_array_equal(arrays["joint_position"], np.stack([JOINT, JOINT + 1.0]))
    np.testing.assert_array_equal(arrays["gripper"], np.array([[1.0], [0.0]]))
    np.testing.assert_array_equal(arrays["deflection"], np.stack([DEFLECTION, -DEFLECTION]))
    np.testing.assert_array_equal(arrays["served"], np.stack([SERVED, -SERVED]))
    np.testing.assert_array_equal(arrays["q0"], Q0)
    np.testing.assert_array_equal(arrays["episode"], np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(arrays["tick"], np.array([0, 4], dtype=np.int32))
    assert arrays["prompt"].tolist() == [PROMPT, "stack the blocks"]
    assert arrays["mode"].tolist() == ["steered", "teleop"]


def test_the_stored_uint8_is_the_models_float_after_rounding(tmp_path):
    """The batch's float [0,1] becomes ``round(255*x)``; ``float(byte)/255`` is the model's input again."""
    batch = _batch()
    recorder = Recorder(META)
    _add(recorder, batch=batch, joint_position=JOINT.astype(np.float32))

    arrays, _ = load(recorder.save(tmp_path / "states.npz"))

    stored = arrays["image_exterior"][0]
    assert stored.dtype == np.uint8
    np.testing.assert_array_equal(stored, np.rint(batch["observation.images.base_0_rgb"] * 255.0))
    np.testing.assert_allclose(stored.astype(np.float32) / 255.0, batch["observation.images.base_0_rgb"])
    assert arrays["joint_position"].dtype == np.float64  # the row type is the file's, not the input's


def test_the_meta_is_one_zero_dimensional_unicode_json_array(tmp_path):
    path = _valid(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        stored = archive["meta"]
    assert stored.shape == () and stored.dtype.kind == "U"
    assert json.loads(str(stored[()])) == {**META, "format_version": FORMAT_VERSION}

    _, meta = load(path)
    assert meta["format_version"] == FORMAT_VERSION


def test_save_creates_its_parent_and_returns_the_path_it_wrote(tmp_path):
    recorder = _filled()
    path = tmp_path / "assets" / "states" / "pilot" / "states.npz"

    assert recorder.save(path) == path
    assert path.is_file()
    assert not (tmp_path / "assets" / "states" / "pilot" / "states.npz.npz").exists()
    assert load(path)[1]["format_version"] == FORMAT_VERSION

    suffixless = tmp_path / "other" / "states"  # written exactly there: np.savez on a string would append
    assert recorder.save(suffixless) == suffixless
    assert suffixless.is_file() and not suffixless.with_suffix(".npz").exists()


def test_an_empty_recorder_round_trips(tmp_path):
    recorder = Recorder(META)
    assert len(recorder) == 0

    arrays, meta = load(recorder.save(tmp_path / "empty" / "states.npz"))

    assert meta == {**META, "format_version": FORMAT_VERSION}
    assert arrays["image_exterior"].shape == (0, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert arrays["joint_position"].shape == (0, ARM_DIMS)
    assert arrays["episode"].shape == (0,)
    assert arrays["prompt"].shape == (0,)
    np.testing.assert_array_equal(arrays["q0"], np.zeros(ARM_DIMS))


# ---------------------------------------------------------------- the refusals


def test_a_caller_cannot_write_another_format_version(tmp_path):
    recorder = Recorder({"format_version": FORMAT_VERSION})
    path = recorder.save(tmp_path / "states.npz")
    assert load(path)[1]["format_version"] == FORMAT_VERSION
    with pytest.raises(ValueError, match="format_version"):
        Recorder({"format_version": FORMAT_VERSION + 1})


def test_the_meta_must_be_json_serializable():
    with pytest.raises(ValueError, match="JSON"):
        Recorder({"config": Path("franka.toml")})
    with pytest.raises(ValueError, match="JSON"):
        Recorder({"bad": float("nan")})


def test_load_refuses_another_format_version_and_names_the_file(tmp_path):
    arrays, meta = load(_valid(tmp_path))
    path = _write(tmp_path / "v2.npz", arrays, {**meta, "format_version": FORMAT_VERSION + 1})
    with pytest.raises(ValueError, match="format_version") as error:
        load(path)
    assert str(path) in str(error.value)


def test_load_refuses_a_missing_or_non_json_meta(tmp_path):
    arrays, meta = load(_valid(tmp_path))
    np.savez(tmp_path / "no_meta.npz", **arrays)
    with pytest.raises(ValueError, match="meta"):
        load(tmp_path / "no_meta.npz")
    path = tmp_path / "bad_json.npz"
    np.savez(path, **{**arrays, "meta": np.array("{not json", dtype=np.str_)})
    with pytest.raises(ValueError, match="meta"):
        load(path)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("joint_position", lambda a: {**a, "joint_position": a["joint_position"][:1]}),
        ("image_wrist", lambda a: {**a, "image_wrist": a["image_wrist"][:1]}),
        ("image_exterior", lambda a: {**a, "image_exterior": a["image_exterior"].astype(np.float32)}),
        ("episode", lambda a: {**a, "episode": a["episode"].astype(np.int64)}),
        ("tick", lambda a: {**a, "tick": a["tick"].astype(np.int64)}),
        ("served", lambda a: {**a, "served": a["served"][:, :6]}),
        ("prompt", lambda a: {**a, "prompt": np.zeros(len(a["prompt"]), dtype=np.float64)}),
        ("q0", lambda a: {**a, "q0": a["q0"][:6]}),
        ("q0", lambda a: {**a, "q0": np.zeros(ARM_DIMS, dtype=np.float32)}),
    ],
)
def test_load_refuses_an_array_that_is_not_the_formats_and_names_it(tmp_path, name, mutate):
    arrays, meta = load(_valid(tmp_path))
    path = _write(tmp_path / "broken.npz", mutate(arrays), meta)
    with pytest.raises(ValueError, match=name):
        load(path)


def test_load_refuses_a_missing_or_unknown_array(tmp_path):
    arrays, meta = load(_valid(tmp_path))
    missing = {key: value for key, value in arrays.items() if key != "served"}
    with pytest.raises(ValueError, match="served"):
        load(_write(tmp_path / "missing.npz", missing, meta))
    with pytest.raises(ValueError, match="extra"):
        load(_write(tmp_path / "extra.npz", {**arrays, "extra": np.zeros(2)}, meta))


def test_load_reads_no_pickle(tmp_path):
    path = tmp_path / "pickle.npz"
    np.savez(path, meta=np.array('{"format_version": 1}', dtype=np.str_), evil=np.array([0], dtype=object))
    with pytest.raises(ValueError, match="allow_pickle|object"):
        load(path)


def test_add_refuses_a_row_that_does_not_fit_and_names_it():
    batch = _batch()
    wrong = {**batch, "observation.images.base_0_rgb": np.zeros((8, 8, 3), dtype=np.float32)}
    with pytest.raises(ValueError, match="image_exterior"):
        _add(Recorder(META), batch=wrong)
    with pytest.raises(ValueError, match="image_wrist"):
        _add(Recorder(META), batch={k: v for k, v in batch.items() if "left_wrist" not in k})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _add(Recorder(META), batch={**batch, "observation.images.base_0_rgb": np.full((3, 224, 224), 2.0)})
    with pytest.raises(ValueError, match="served"):
        _add(Recorder(META), served=np.zeros(6))
    with pytest.raises(ValueError, match="q0"):
        _add(Recorder(META), q0=np.zeros(6))
    with pytest.raises(ValueError, match="episode"):
        _add(Recorder(META), episode=-1)
    with pytest.raises(ValueError, match="tick"):
        _add(Recorder(META), tick=True)


def test_a_refused_row_leaves_the_buffer_untouched(tmp_path):
    """A row is validated before anything is appended: one bad add cannot leave a half-recorded file."""
    recorder = Recorder(META)
    with pytest.raises(ValueError, match="served"):
        _add(recorder, served=np.zeros(6))
    assert len(recorder) == 0
    arrays, _ = load(recorder.save(tmp_path / "states.npz"))
    assert arrays["joint_position"].shape == (0, ARM_DIMS)
    assert arrays["image_exterior"].shape == (0, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_a_second_add_cannot_change_the_sessions_nominal_pose():
    recorder = Recorder(META)
    _add(recorder, q0=Q0)
    with pytest.raises(ValueError, match="q0"):
        _add(recorder, q0=Q0 + 0.5)
    assert len(recorder) == 1
