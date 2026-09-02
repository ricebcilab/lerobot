import numpy as np
import pytest
import teleop
from teleop import (
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    CombinedReader,
    CommandCorruption,
    KeyboardReader,
    NoisyReader,
    RecordingReader,
    TeleopChain,
    build_corruption,
    normalize,
    parse_report,
    read_matrix_spec,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Source:
    def __init__(self, translation=(0.0, 0.0, 0.0), gripper=GRIPPER_OPEN):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.gripper = gripper


def test_parse_report_and_normalize():
    kind, xyz = parse_report(bytes([1]) + (350).to_bytes(2, "little", signed=True) * 3)
    assert kind == "translation" and xyz == (350, 350, 350)
    assert parse_report(bytes([3, 1, 0, 0, 0])) == ("buttons", 1)
    assert parse_report(b"\x09")[0] == "other"
    assert np.all(np.abs(normalize((700, -700, 0))) <= 1.0)


def test_keyboard_reader_speed_stale_and_toggle_parity():
    clock = Clock()
    kb = KeyboardReader(speed=0.5, stale_after=1.0, clock=clock)
    kb.update({"ArrowUp"}, toggles=0)
    forward = kb.translation
    assert np.linalg.norm(forward) == pytest.approx(0.5)
    kb.update({"ArrowUp", "Shift"}, toggles=0)
    assert np.linalg.norm(kb.translation) == pytest.approx(1.0)
    kb.update({"ArrowUp"}, toggles=1)
    assert kb.gripper == GRIPPER_CLOSE
    kb.update({"ArrowUp"}, toggles=2)
    assert kb.gripper == GRIPPER_OPEN
    kb.update({"ArrowUp"}, toggles=0)  # page reload restarts the counter: no toggle
    assert kb.gripper == GRIPPER_OPEN
    clock.now = 2.0
    np.testing.assert_array_equal(kb.translation, 0)  # stale
    assert kb.held == set()
    assert kb.gripper == GRIPPER_OPEN  # gripper keeps its state


def test_combined_reader_priority_and_gripper_parity():
    a, b = Source((0.0, 0.0, 0.0)), Source((0.3, 0.0, 0.0), GRIPPER_CLOSE)
    combined = CombinedReader([a, b])
    np.testing.assert_array_equal(combined.translation, [0.3, 0.0, 0.0])
    assert combined.gripper == GRIPPER_CLOSE
    a.translation = np.array([-0.5, 0.0, 0.0])
    np.testing.assert_array_equal(combined.translation, [-0.5, 0.0, 0.0])  # first deflected wins
    a.gripper = GRIPPER_CLOSE
    assert combined.gripper == GRIPPER_OPEN  # two closed = even parity


def test_command_corruption_applies_only_when_commanding():
    source = Source((1.0, 0.0, 0.0))
    corrupted = CommandCorruption(source, matrix=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], label="rot90")
    np.testing.assert_allclose(corrupted.translation, [0.0, 1.0, 0.0])
    source.translation = np.array([DEADBAND / 2, 0.0, 0.0])
    np.testing.assert_allclose(corrupted.translation, [DEADBAND / 2, 0.0, 0.0])
    corrupted.matrix = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    source.translation = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(corrupted.translation, [1.0, 0.0, 0.0])  # clipped
    assert "rot90" in corrupted.describe()
    corrupted.matrix = None
    assert "off" in corrupted.describe()
    with pytest.raises(ValueError):
        corrupted.matrix = np.eye(2)


def test_build_corruption_and_read_matrix_spec(tmp_path):
    assert build_corruption(None) is None
    np.testing.assert_allclose(build_corruption({"rotation_z_deg": 0}), np.eye(3))
    path = tmp_path / "m.yaml"
    path.write_text("rotation_z_deg: 90\n")
    m = build_corruption(read_matrix_spec(path))
    np.testing.assert_allclose(m @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)
    path.write_text("M:\n  - [1, 0, 0]\n  - [0, 1, 0]\n  - [0, 0, 1]\n")
    np.testing.assert_allclose(build_corruption(read_matrix_spec(path)), np.eye(3))
    path.write_text("- 1\n")
    with pytest.raises(ValueError):
        build_corruption(read_matrix_spec(path))


def test_noisy_reader_is_seeded_and_idle_stays_idle():
    source = Source((0.5, 0.5, 0.5))
    a = NoisyReader(source, std=0.1, rng=np.random.default_rng(1)).translation
    b = NoisyReader(source, std=0.1, rng=np.random.default_rng(1)).translation
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, source.translation)
    source.translation = np.zeros(3)
    np.testing.assert_array_equal(NoisyReader(source, std=0.1).translation, 0)
    with pytest.raises(ValueError):
        NoisyReader(source, std=-1)


def test_recording_reader_counts_reads():
    rec = RecordingReader(Source((0.2, 0.0, 0.0), GRIPPER_CLOSE))
    assert rec.reads == 0
    rec.translation  # noqa: B018 (read for its side effect: bumps rec.reads)
    rec.translation  # noqa: B018
    assert rec.reads == 2
    np.testing.assert_array_equal(rec.last_translation, [0.2, 0.0, 0.0])
    assert rec.gripper == GRIPPER_CLOSE and rec.last_gripper == GRIPPER_CLOSE


def test_teleop_chain_order_and_records():
    kb = KeyboardReader(clock=lambda: 0.0)
    kb.update({"ArrowRight"}, toggles=0)  # full push right at half speed
    chain = TeleopChain(kb, input_noise=0.0)
    chain.corruption.matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    served = chain.reader.translation
    raw = chain.raw.last_translation
    assert chain.served.reads == 1 and chain.raw.reads == 1
    np.testing.assert_allclose(served, chain.corruption.matrix @ raw)
    assert chain.spacemouse is None
    assert chain.combined.sources == [kb]


def test_module_reexports_shared_constants():
    from lerobot.policies.pi05 import steering

    assert teleop.DEADBAND is steering.DEADBAND
    assert teleop.GRIPPER_OPEN == steering.GRIPPER_OPEN
