"""The config loader: DESIGN.md's schema parses to the built-in defaults, unknown keys are named,
the off states are strings, the example's relative paths resolve, and a mode only reads its knobs.

Every test writes its own temporary TOML: the shipped ``configs/franka.toml`` belongs to the
integration slice, and this module must not depend on it existing.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import config as config_module
import pytest
from config import Config, load_config

EXAMPLE_DIR = Path(config_module.__file__).parent

SHIPPED = """
[station]
host = "10.42.0.1"
port = 9090
arm = "follower"
rate = 15.0

[cameras]
exterior = "scene_zed"
wrist = "wrist_zed"

[policy]
path = "lerobot/pi05_droid"
stats = "assets/droid/norm_stats.json"
device = "cuda"
n_action_steps = 8
dtype = "bfloat16"

[command]
translation_gain = 0.11
rotation_gain = 0.5
twist_length = 0.2
deadband = 0.02
gain = "auto"
q0 = []

[readout]
corruption = {rotation_z_deg = 20}
noise = 0.35
seed = 0

[steering]
n_reversal_steps = 10
pinned_steps = 8

[modes.fc]
mode = "flow_control"
tau = 8
lam = 1
ic_gain = 15.0

[modes.frs]
mode = "frs"

[modes.mars-f]
mode = "frs"
adapter_matrix = "corruption"

[modes.mars-u]
mode = "frs"
adapter_user = "assets/user/u/user.pt"

[modes.mars-t]
mode = "frs"
adapter_task = "assets/task/t/task.pt"

[modes.mars]
mode = "frs"
adapter_user = "assets/user/u/user.pt"
adapter_task = "assets/task/t/task.pt"

[recording]
dir = "outputs/mars_franka"
"""


def _load(tmp_path, text: str) -> Config:
    path = tmp_path / "franka.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_the_shipped_schema_parses_to_the_documented_tree(tmp_path):
    cfg = _load(tmp_path, SHIPPED)
    assert cfg.station.host == "10.42.0.1" and cfg.station.port == 9090
    assert cfg.station.arm == "follower" and cfg.station.rate == 15.0
    assert cfg.cameras.exterior == "scene_zed" and cfg.cameras.wrist == "wrist_zed"
    assert cfg.cameras.third is None  # absent: the third image key is omitted
    assert cfg.policy.path == "lerobot/pi05_droid" and cfg.policy.device == "cuda"
    assert cfg.policy.n_action_steps == 8 and cfg.policy.dtype == "bfloat16"
    assert cfg.command.translation_gain == 0.11 and cfg.command.rotation_gain == 0.5
    assert cfg.command.twist_length == 0.2 and cfg.command.deadband == 0.02
    assert cfg.command.gain == "auto" and cfg.command.q0 == ()
    assert cfg.readout.corruption == {"rotation_z_deg": 20} and cfg.readout.noise == 0.35
    assert cfg.readout.seed == 0
    assert cfg.steering.mode == "none" and cfg.steering.n_reversal_steps == 10
    assert cfg.steering.pinned_steps == 8
    assert set(cfg.modes) == {"fc", "frs", "mars-f", "mars-u", "mars-t", "mars"}
    assert cfg.modes["fc"].mode == "flow_control"
    assert (cfg.modes["fc"].tau, cfg.modes["fc"].lam, cfg.modes["fc"].ic_gain) == (8, 1, 15.0)
    assert cfg.modes["frs"].mode == "frs"
    assert cfg.modes["mars-f"].adapter_matrix == "corruption"
    assert cfg.modes["mars-u"].adapter_user == EXAMPLE_DIR / "assets/user/u/user.pt"
    assert cfg.modes["mars-t"].adapter_task == EXAMPLE_DIR / "assets/task/t/task.pt"
    assert cfg.modes["mars"].adapter_user == EXAMPLE_DIR / "assets/user/u/user.pt"
    assert cfg.modes["mars"].adapter_task == EXAMPLE_DIR / "assets/task/t/task.pt"
    assert cfg.recording.dir == EXAMPLE_DIR / "outputs/mars_franka"
    assert cfg.policy.stats == EXAMPLE_DIR / "assets/droid/norm_stats.json"


def test_an_empty_or_minimal_file_has_the_neutral_readout(tmp_path):
    """A library default must be the harmless one: no corruption, no noise. The shipped study
    config (a later slice) spells its corruption and noise explicitly."""
    empty = _load(tmp_path, "")
    assert empty == Config()
    assert (empty.readout.corruption, empty.readout.noise, empty.readout.seed) == (None, 0.0, 0)
    assert empty.command.gain == "auto" and empty.modes == {}
    minimal = _load(tmp_path, '[station]\nhost = "10.42.0.1"\n')
    assert (minimal.readout.corruption, minimal.readout.noise) == (None, 0.0)


def test_the_modes_inherit_the_steering_defaults(tmp_path):
    """A mode table overrides the [steering] section key by key; it does not restate it."""
    cfg = _load(
        tmp_path,
        '[steering]\nn_reversal_steps = 12\n\n[modes.frs]\nmode = "frs"\n\n[modes.x]\nmode = "frs"\npinned_steps = 4\n',
    )
    assert cfg.modes["frs"].n_reversal_steps == 12 and cfg.modes["frs"].pinned_steps == 8
    assert cfg.modes["x"].n_reversal_steps == 12 and cfg.modes["x"].pinned_steps == 4


@pytest.mark.parametrize(
    ("text", "where", "key"),
    [
        ("[station]\nvelocity = 1.0\n", "station", "velocity"),
        ('[cameras]\nrealsense = "x"\n', "cameras", "realsense"),
        ('[modes.fc]\nmode = "flow_control"\ntaus = 8\n', "modes.fc", "taus"),
        ('[steering]\nmode = "frs"\n', "steering", "mode"),
        ('[recording]\nout = "x"\n', "recording", "out"),
    ],
)
def test_an_unknown_key_names_its_section_and_key(tmp_path, text, where, key):
    with pytest.raises(ValueError, match=key) as error:
        _load(tmp_path, text)
    assert where in str(error.value)


def test_an_unknown_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="stations"):
        _load(tmp_path, '[stations]\nhost = "10.42.0.1"\n')


@pytest.mark.parametrize(
    ("text", "knob"),
    [
        ('[modes.frs]\nmode = "frs"\ntau = 4\n', "tau"),
        ('[modes.fc]\nmode = "flow_control"\npinned_steps = 8\n', "pinned_steps"),
        ('[modes.clean]\nmode = "none"\nadapter_user = "assets/user/u.pt"\n', "adapter_user"),
        ('[modes.fc]\nmode = "flow_control"\nadapter_task = "assets/task/t.pt"\n', "adapter_task"),
    ],
)
def test_a_mode_rejects_a_knob_of_another_mode(tmp_path, text, knob):
    with pytest.raises(ValueError, match=knob):
        _load(tmp_path, text)


def test_a_mode_must_name_one_of_the_three_modes(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        _load(tmp_path, "[modes.x]\ntau = 4\n")
    with pytest.raises(ValueError, match="flow_control"):
        _load(tmp_path, '[modes.x]\nmode = "flow"\n')


def test_gain_is_auto_or_a_fraction(tmp_path):
    assert _load(tmp_path, '[command]\ngain = "auto"\n').command.gain == "auto"
    assert _load(tmp_path, "[command]\ngain = 0.5\n").command.gain == 0.5
    assert _load(tmp_path, "[command]\ngain = 1\n").command.gain == 1.0
    for bad in ("\ngain = 0\n", "\ngain = 1.5\n", '\ngain = "fast"\n', "\ngain = true\n"):
        with pytest.raises(ValueError, match="gain"):
            _load(tmp_path, "[command]" + bad)


def test_q0_is_empty_or_the_seven_joint_reset_pose(tmp_path):
    assert _load(tmp_path, "[command]\nq0 = []\n").command.q0 == ()
    angles = [0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0]
    assert _load(tmp_path, f"[command]\nq0 = {angles}\n").command.q0 == tuple(angles)
    with pytest.raises(ValueError, match="q0"):
        _load(tmp_path, "[command]\nq0 = [0.0, 0.0]\n")
    with pytest.raises(ValueError, match="q0"):
        _load(tmp_path, '[command]\nq0 = ["up"]\n')


def test_relative_paths_resolve_against_the_example_and_absolute_paths_pass_through(tmp_path):
    absolute = tmp_path / "stats.json"
    cfg = _load(
        tmp_path,
        f'[policy]\nstats = "assets/droid/norm_stats.json"\n'
        f'[recording]\ndir = "outputs/run"\n'
        f'[modes.x]\nmode = "frs"\nadapter_user = "{absolute}"\n',
    )
    assert cfg.policy.stats == EXAMPLE_DIR / "assets/droid/norm_stats.json"
    assert cfg.recording.dir == EXAMPLE_DIR / "outputs/run"
    assert cfg.modes["x"].adapter_user == absolute


def test_the_corruption_off_state_is_the_string_none(tmp_path):
    """TOML has no null: absent is the neutral default (off), and the explicit off is "none", which
    parses to the same None so a variant file can turn off what it inherits."""
    absent = _load(tmp_path, "")
    explicit = _load(tmp_path, '[readout]\ncorruption = "none"\n')
    assert explicit.readout.corruption is None
    assert explicit.readout == absent.readout
    with pytest.raises(ValueError, match="corruption"):
        _load(tmp_path, '[readout]\ncorruption = "curl"\n')
    with pytest.raises(ValueError, match="corruption"):
        _load(tmp_path, "[readout]\ncorruption = {}\n")


def test_adapter_matrix_accepts_the_two_names_or_a_literal_7x7(tmp_path):
    rows = [[float(i == j) for j in range(7)] for i in range(7)]
    cfg = _load(tmp_path, f'[modes.x]\nmode = "frs"\nadapter_matrix = {rows}\n')
    assert cfg.modes["x"].adapter_matrix == tuple(tuple(row) for row in rows)
    assert (
        _load(tmp_path, '[modes.x]\nmode = "frs"\nadapter_matrix = "corruption_inverse"\n')
        .modes["x"]
        .adapter_matrix
        == "corruption_inverse"
    )
    with pytest.raises(ValueError, match="adapter_matrix"):
        _load(tmp_path, f'[modes.x]\nmode = "frs"\nadapter_matrix = {rows[:6]}\n')
    with pytest.raises(ValueError, match="adapter_matrix"):
        _load(tmp_path, '[modes.x]\nmode = "frs"\nadapter_matrix = [1, 2, 3, 4, 5, 6, 7]\n')
    with pytest.raises(ValueError, match="adapter_matrix"):
        _load(tmp_path, '[modes.x]\nmode = "frs"\nadapter_matrix = "other"\n')


def test_a_syntax_error_names_the_file(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("[station\n", encoding="utf-8")
    with pytest.raises(ValueError, match=str(path)):
        load_config(path)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[station]\nrate = 0.0\n", "rate"),
        ("[station]\nport = 0\n", "port"),
        ("[policy]\nn_action_steps = 0\n", "n_action_steps"),
        ("[readout]\nnoise = -0.1\n", "noise"),
        ("[readout]\nseed = -1\n", "seed"),
        ("[steering]\ntau = 0\n", "tau"),
        ('[modes.x]\nmode = "frs"\npinned_steps = 0\n', "pinned_steps"),
    ],
)
def test_values_are_checked_and_named(tmp_path, text, key):
    with pytest.raises(ValueError, match=key):
        _load(tmp_path, text)


def test_the_config_is_frozen(tmp_path):
    cfg = _load(tmp_path, SHIPPED)
    with pytest.raises(FrozenInstanceError):
        cfg.station.host = "other"
    with pytest.raises(TypeError):
        cfg.modes["fc"] = cfg.modes["frs"]
