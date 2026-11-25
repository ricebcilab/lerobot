import numpy as np
import pytest
import teleop
from config import read_matrix_spec
from teleop import (
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    CombinedReader,
    KeyboardReader,
    PolicyOperator,
    PolicyOperatorSettings,
    ReadoutReader,
    RecordingReader,
    TeleopChain,
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
    corrupted = ReadoutReader(source, matrix=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], label="rot90")
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


def test_a_corruption_matrix_from_a_spec_and_from_a_spec_file(tmp_path):
    assert ReadoutReader.matrix_from_spec(None) is None
    np.testing.assert_allclose(ReadoutReader.matrix_from_spec({"rotation_z_deg": 0}), np.eye(3))
    path = tmp_path / "m.yaml"
    path.write_text("rotation_z_deg: 90\n")
    m = ReadoutReader.matrix_from_spec(read_matrix_spec(path))
    np.testing.assert_allclose(m @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)
    path.write_text("M:\n  - [1, 0, 0]\n  - [0, 1, 0]\n  - [0, 0, 1]\n")
    np.testing.assert_allclose(ReadoutReader.matrix_from_spec(read_matrix_spec(path)), np.eye(3))
    path.write_text("- 1\n")
    with pytest.raises(ValueError):
        ReadoutReader.matrix_from_spec(read_matrix_spec(path))


def test_noisy_reader_is_seeded_and_noises_an_idle_push_too():
    source = Source((0.5, 0.5, 0.5))
    a = ReadoutReader(source, translation_noise=0.1, rng=np.random.default_rng(1)).translation
    b = ReadoutReader(source, translation_noise=0.1, rng=np.random.default_rng(1)).translation
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, source.translation)
    source.translation = np.zeros(3)  # the operator rests: the readout still serves its noise
    rest = ReadoutReader(source, translation_noise=0.1, rng=np.random.default_rng(1)).translation
    np.testing.assert_allclose(rest, np.clip(0.1 * np.random.default_rng(1).normal(size=3), -1.0, 1.0))
    with pytest.raises(ValueError):
        ReadoutReader(source, translation_noise=-1)


def test_the_operator_lets_go_while_the_robot_heads_where_it_means():
    """With `let_go_cosine` the operator watches what the robot executed since its last update: heading with its
    new intent at that cosine or more, it rests (zero translation and rotation) until the next update; off course
    it pushes again; `may_rest=False` (teleop: nothing else moves the robot) and the default always push."""
    op = PolicyOperator(10, PolicyOperatorSettings(let_go_cosine=0.5))
    chunk = np.zeros((10, 7))
    chunk[:, 0], chunk[:, 3] = 0.6, 0.1  # the intent: +x, with a little rotation
    op.refresh(chunk, step=0)  # nothing executed yet: it pushes
    np.testing.assert_allclose(op.translation, [0.6, 0.0, 0.0])
    assert not op.resting

    def execute(direction):
        for _ in range(10):
            op.observe(np.array([*direction, 0.0, 0.0, 0.0, -1.0]))

    execute([0.5, 0.2, 0.0])  # on course (cosine 0.93)
    op.refresh(chunk, step=10)
    np.testing.assert_array_equal(op.translation, 0.0)
    np.testing.assert_array_equal(op.rotation, 0.0)
    np.testing.assert_allclose(op.oracle_translation, [0.6, 0.0, 0.0])  # the intent is still what it means
    assert op.resting
    execute([0.1, 0.5, 0.0])  # off course (cosine 0.20)
    op.refresh(chunk, step=20)
    np.testing.assert_allclose(op.translation, [0.6, 0.0, 0.0])
    assert not op.resting
    execute([0.5, 0.0, 0.0])
    op.refresh(chunk, step=30, may_rest=False)
    np.testing.assert_allclose(op.translation, [0.6, 0.0, 0.0])
    default = PolicyOperator(10)
    default.observe(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
    default.refresh(chunk, step=10)
    np.testing.assert_allclose(default.translation, [0.6, 0.0, 0.0])
    with pytest.raises(ValueError, match="let_go_cosine"):
        PolicyOperatorSettings(let_go_cosine=1.5)


def test_recording_reader_counts_reads():
    rec = RecordingReader(Source((0.2, 0.0, 0.0), GRIPPER_CLOSE))
    assert rec.reads == 0
    rec.translation  # noqa: B018 (read for its side effect: bumps rec.reads)
    rec.translation  # noqa: B018
    assert rec.reads == 2
    np.testing.assert_array_equal(rec.last_translation, [0.2, 0.0, 0.0])
    assert rec.gripper == GRIPPER_CLOSE and rec.last_gripper == GRIPPER_CLOSE


def test_rotation_and_gripper_carry_their_own_seeded_noise():
    source = Source((0.5, 0.0, 0.0), gripper=GRIPPER_CLOSE)
    source.rotation = np.array([0.02, 0.0, -0.01])
    quiet = ReadoutReader(source)  # off by default: both served as given
    np.testing.assert_array_equal(quiet.rotation, source.rotation)
    assert quiet.gripper == GRIPPER_CLOSE
    reader = ReadoutReader(source, rotation_noise=0.04, gripper_noise=1.0, rng=np.random.default_rng(0))
    turns = np.array([reader.rotation for _ in range(2000)])
    np.testing.assert_allclose(turns.mean(0), source.rotation, atol=0.004)
    np.testing.assert_allclose(turns.std(0), 0.04, atol=0.004)
    grips = np.array([reader.gripper for _ in range(2000)])
    assert set(grips) == {GRIPPER_OPEN, GRIPPER_CLOSE}  # served as a command: the sign of the noisy one
    assert (grips == GRIPPER_OPEN).mean() == pytest.approx(0.159, abs=0.03)  # P(N(0, 1) < -1)
    np.testing.assert_array_equal(
        reader.translation, source.translation
    )  # the translation's noise is its own
    for bad in ({"rotation_noise": -1}, {"gripper_noise": -1}):
        with pytest.raises(ValueError, match="noise must be >= 0"):
            ReadoutReader(source, **bad)


def test_teleop_chain_order_and_records():
    kb = KeyboardReader(clock=lambda: 0.0)
    kb.update({"ArrowRight"}, toggles=0)  # full push right at half speed
    chain = TeleopChain(kb, translation_noise=0.0)
    chain.readout.matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    served = chain.source.translation
    raw = chain.raw.last_translation
    assert chain.served.reads == 1 and chain.raw.reads == 1
    np.testing.assert_allclose(served, chain.readout.matrix @ raw)
    assert chain.spacemouse is None
    assert chain.combined.sources == [kb]


def test_policy_operator_averages_the_executed_prefix_and_clips():
    op = teleop.PolicyOperator(n_action_steps=2)
    np.testing.assert_array_equal(op.translation, np.zeros(3))
    np.testing.assert_array_equal(op.rotation, np.zeros(3))
    assert op.gripper == GRIPPER_OPEN
    chunk = np.array(
        [
            [0.4, 0.0, 0.0, 0.2, 0, -0.4, 1.0],
            [0.8, 0.0, 0.0, 0.4, 0, -0.2, 1.0],
            [9.0, 9.0, 9.0, 9, 9, 9, -1.0],
        ]
    )
    op.refresh(chunk)
    np.testing.assert_allclose(op.translation, [0.6, 0.0, 0.0])
    np.testing.assert_allclose(op.rotation, [0.3, 0.0, -0.3])  # the whole command: the hand's turn too
    assert op.gripper == GRIPPER_CLOSE
    op.refresh(np.array([[2.0, -3.0, 0.0, 0, 5.0, 0, -1.0]]))
    np.testing.assert_allclose(op.translation, [1.0, -1.0, 0.0])
    np.testing.assert_allclose(op.rotation, [0.0, 1.0, 0.0])
    assert op.gripper == GRIPPER_OPEN


def test_teleop_chain_attach_operator_takes_priority():
    keyboard = KeyboardReader(clock=Clock().now)
    chain = TeleopChain(keyboard)
    op = teleop.PolicyOperator(n_action_steps=1)
    chain.attach_operator(op)
    assert chain.combined.sources[0] is op
    np.testing.assert_array_equal(TeleopChain(keyboard).source.rotation, np.zeros(3))  # the devices give none
    op.refresh(np.array([[0.5, 0.0, 0.0, 0.1, 0, -0.2, -1.0]]))
    np.testing.assert_allclose(chain.source.translation, [0.5, 0.0, 0.0])
    np.testing.assert_allclose(chain.raw.last_translation, [0.5, 0.0, 0.0])
    chain.readout.matrix, chain.readout.translation_noise = [[0, -1, 0], [1, 0, 0], [0, 0, 1]], 0.5
    np.testing.assert_allclose(
        chain.source.rotation, [0.1, 0.0, -0.2]
    )  # a user corrupts the translation only


def test_operator_delays_snapshots_holds_continuous_commands_and_resets():
    op = teleop.PolicyOperator(10, teleop.PolicyOperatorSettings(update_every_steps=20, delay_steps=4))
    assert [step for step in range(41) if op.should_refresh(step)] == [0, 20, 40]
    chunk = np.array([[0.25, -0.37, 0.15, 0, 0, 0, 1]])
    op.refresh(chunk, step=0)
    chunk[:] = 0  # pending commands retain the earlier observation's intent
    op.advance(3)
    np.testing.assert_array_equal(op.translation, 0)
    assert op.gripper == GRIPPER_OPEN and op.command_age_steps == -1
    op.advance(4)
    np.testing.assert_allclose(op.translation, [0.25, -0.37, 0.15])
    assert op.gripper == GRIPPER_CLOSE and op.command_age_steps == 4
    op.refresh(np.array([[-0.5, 0, 0, 0, 0, 0, -1]]), step=20)
    op.advance(23)
    np.testing.assert_allclose(op.translation, [0.25, -0.37, 0.15])
    assert op.command_age_steps == 23
    op.advance(24)
    np.testing.assert_allclose(op.translation, [-0.5, 0, 0])
    assert op.command_age_steps == 4 and op.refreshes == 2
    op.refresh(np.ones((1, 7)), step=40)
    op.reset()
    op.advance(100)
    np.testing.assert_array_equal(op.translation, 0)
    assert op.command_age_steps == -1 and op.refreshes == 0 and op.gripper == GRIPPER_OPEN


def test_operator_delay_longer_than_update_period_preserves_queued_commands():
    op = teleop.PolicyOperator(10, teleop.PolicyOperatorSettings(update_every_steps=2, delay_steps=5))
    for step in [0, 2, 4]:
        op.refresh(np.full((1, 7), (step + 1) / 10), step=step)
    for step, expected in [(5, 0.1), (6, 0.1), (7, 0.3), (9, 0.5)]:
        op.advance(step)
        np.testing.assert_allclose(op.translation, expected)


@pytest.mark.parametrize(
    "settings",
    [
        {"update_every_steps": 0},
        {"delay_steps": 0.5},
    ],
)
def test_operator_rejects_invalid_step_counts(settings):
    with pytest.raises(ValueError, match="policy_operator"):
        teleop.PolicyOperatorSettings(**settings)


# ---------------------------------------------------------------- translation gain: no clipping


def test_translation_gain_keeps_the_direction_and_shortens_the_command():
    source = Source((1.0, 1.0, 0.0), GRIPPER_CLOSE)
    limited = teleop.TranslationGain(source, gain=0.5)
    # two axes at full deflection: same direction, length 1 before the gain, 0.5 after
    np.testing.assert_allclose(limited.translation, [0.35355339, 0.35355339, 0.0])
    assert limited.gripper == GRIPPER_CLOSE
    source.translation = np.array([0.4, 0.0, 0.0])  # already shorter than 1: only the gain applies
    np.testing.assert_allclose(limited.translation, [0.2, 0.0, 0.0])
    limited.gain = None  # off: the command passes through untouched, corner and all
    source.translation = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(limited.translation, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("gain", [0, 1.5, "auto"])
def test_translation_gain_rejects_values_outside_zero_to_one(gain):
    with pytest.raises(ValueError, match="gain"):
        teleop.TranslationGain(Source(), gain=gain)


def test_chain_with_translation_gain_never_clips_a_corruption():
    stick = Source()
    chain = TeleopChain(KeyboardReader(clock=lambda: 0.0))
    chain.combined.sources.insert(0, stick)
    rng = np.random.default_rng(0)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    pushes = rng.uniform(-1, 1, size=(200, 3))
    pushes = pushes[np.linalg.norm(pushes, axis=1) >= 0.5]  # clearly commanding, not idle
    turned = ReadoutReader.matrix_from_spec({"rotation_z_deg": 40}) @ np.diag([1.0, 3.0, 1.0])
    for matrix in (np.diag([4.0, 1.0, 2.0]), turned):
        longest = np.linalg.norm(matrix, 2)  # the most the corruption can lengthen a command
        chain.gain.gain = 1 / longest
        chain.readout.matrix = matrix
        for push in np.vstack([corners, np.eye(3), pushes]):
            stick.translation = push
            served = chain.source.translation
            raw = chain.raw.last_translation
            assert np.max(np.abs(served)) <= 1.0
            # what the policy is served is exactly M @ what was recorded as the operator's command
            np.testing.assert_allclose(served, chain.readout.matrix @ raw, atol=1e-12)
            # and that recorded command is the push itself, only shorter
            assert np.linalg.norm(raw) <= 1 / longest + 1e-12
            assert raw @ push == pytest.approx(np.linalg.norm(raw) * np.linalg.norm(push))


# ---------------------------------------------------------------- curl corruption

from teleop import Curl  # noqa: E402


def test_curl_corruption_integrates_on_the_clock_and_idle_stays_idle():
    source = Source((1.0, 0.0, 0.0))
    curl = Curl(curl=20.0, tau=0.1, dt=0.05)
    corrupted = ReadoutReader(source, curl=curl)
    decay, drive = curl.discretization()
    assert "curl field" in corrupted.describe()
    np.testing.assert_allclose(corrupted.translation, np.zeros(3))  # no clock yet: still at rest
    corrupted.advance(0)
    first = corrupted.translation
    np.testing.assert_allclose(first, drive @ source.translation)
    np.testing.assert_allclose(
        corrupted.translation, first
    )  # a second read at the same clock: no re-integration
    corrupted.advance(9)  # a chunk later: nine more steps with the push held
    v = first.copy()
    for _ in range(9):
        v = decay @ v + drive @ source.translation
    np.testing.assert_allclose(corrupted.translation, np.clip(v, -1, 1))
    source.translation = np.array(
        [DEADBAND / 2, 0.0, 0.0]
    )  # idle push: served as is, the field keeps evolving
    corrupted.advance(10)
    np.testing.assert_allclose(corrupted.translation, [DEADBAND / 2, 0.0, 0.0])
    corrupted.reset()
    corrupted.advance(0)
    source.translation = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(corrupted.translation, first)  # from rest again
    with pytest.raises(ValueError, match="not both"):
        ReadoutReader(source, matrix=np.eye(3), curl=curl)
    corrupted.curl = None
    np.testing.assert_allclose(corrupted.translation, [1.0, 0.0, 0.0])


def test_build_curl_spec():
    curl = Curl.from_spec({"curl": 20, "tau": 0.1})
    assert curl == Curl(20.0, 0.1) and curl.dt == teleop.CONTROL_DT
    assert Curl.from_spec({"curl": 5, "tau": 0.2, "dt": 0.02}).dt == 0.02
    assert Curl.from_spec({"rotation_z_deg": 20}) is None and Curl.from_spec(None) is None
    for bad in (
        {"curl": 20},
        {"curl": 20, "tau": 0.1, "axis": "z"},
        {"curl": 20, "tau": 0},
        {"curl": "x", "tau": 0.1},
    ):
        with pytest.raises(ValueError):
            Curl.from_spec(bad)


def test_teleop_chain_advances_and_resets_the_curl():
    kb = KeyboardReader(clock=lambda: 0.0)
    kb.update({"ArrowRight"}, toggles=0)
    chain = TeleopChain(kb, translation_noise=0.0)
    chain.readout.curl = Curl(curl=20.0, tau=0.1, dt=0.05)
    chain.advance(0)
    served = chain.source.translation
    raw = chain.raw.last_translation
    _, drive = chain.readout.curl.discretization()
    np.testing.assert_allclose(served, drive @ raw)
    chain.reset()
    np.testing.assert_allclose(chain.source.translation, np.zeros(3))  # at rest until the clock advances


def test_a_keypad_readout_snaps_and_a_dropped_axis_goes_idle():
    keypad = ReadoutReader(Source((0.3, -0.7, 0.2)), quantize=True)
    np.testing.assert_allclose(
        keypad.translation, [0.0, -np.sqrt(0.62), 0.0]
    )  # the nearest axis, the push's length
    assert "keypad" in keypad.describe()
    with pytest.raises(ValueError, match="not both"):
        keypad.matrix = np.eye(3)
    keypad.quantize = False
    np.testing.assert_array_equal(keypad.translation, [0.3, -0.7, 0.2])
    stick = ReadoutReader(Source((0.0, 0.0, 0.8)), matrix=np.diag([1.0, 1.0, 0.0]))
    assert np.max(np.abs(stick.translation)) < DEADBAND  # a purely vertical push through a two-axis device


def test_the_target_user_approaches_rests_for_the_grasp_carries_and_recovers():
    """`PolicyOperator.target_push`: a straight push at the task's object (unit length far away, shrinking inside
    TARGET_SLOW); over it (within TARGET_NEAR sideways and TARGET_ABOVE above) it rests and leaves the grasp to the
    policy; once the object has moved from where the operator first saw it and is with the gripper it pushes toward
    a point TARGET_HOVER above the destination (level with the gripper while the object is not lifted: a push
    task) and rests above it; a dropped object is approached again; a reset forgets where the object started."""
    start, dest = np.array([0.3, 0.0, 0.6]), np.array([0.0, 0.4, 0.6])
    user = PolicyOperator(2)
    np.testing.assert_allclose(user.target_push([0.0, 0.0, 1.0], start, dest), [0.6, 0.0, -0.8])
    np.testing.assert_allclose(user.object_start, start)
    closer = start + [0.75 * teleop.TARGET_SLOW, 0.0, 0.0]
    np.testing.assert_allclose(user.target_push(closer, start, dest), [-0.75, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        user.target_push(start + [0.0, 0.0, 0.2], start, dest), [0.0, 0.0, -1.0]
    )  # too high
    over_it = start + [teleop.TARGET_NEAR / 2, 0.0, 0.9 * teleop.TARGET_ABOVE]
    np.testing.assert_allclose(user.target_push(over_it, start, dest), 0.0)  # the grasp is the policy's
    held = np.array([0.3, 0.0, 0.8])  # the object lifted 0.2 with the gripper just above it
    carry = user.target_push(held + [0.0, 0.0, 0.05], held, dest)
    aim = dest + [0.0, 0.0, teleop.TARGET_HOVER] - (held + [0.0, 0.0, 0.05])
    np.testing.assert_allclose(carry, aim / np.linalg.norm(aim))
    slid = start + [0.05, 0.0, 0.0]  # moved along the table, not lifted: a push, level with the gripper
    level = user.target_push(slid + [0.0, 0.0, 0.04], slid, dest)
    assert level[2] == 0.0 and level[1] > 0.0
    over = np.array([0.0, 0.4, 0.9])
    np.testing.assert_allclose(
        user.target_push(over, over - [0.0, 0.0, 0.05], dest), 0.0
    )  # the policy places
    dropped = np.array([0.2, 0.2, 0.6])
    back = user.target_push([0.0, 0.4, 0.9], dropped, dest)
    np.testing.assert_allclose(back, (dropped - [0.0, 0.4, 0.9]) / np.linalg.norm(dropped - [0.0, 0.4, 0.9]))
    user.reset()  # a new trial: the dropped place is now where the object starts, and it is approached, not carried
    assert user.object_start is None
    np.testing.assert_allclose(
        user.target_push(dropped + [0.0, 0.0, 0.03], dropped, None), 0.0
    )  # a button: done
    np.testing.assert_allclose(user.object_start, dropped)
