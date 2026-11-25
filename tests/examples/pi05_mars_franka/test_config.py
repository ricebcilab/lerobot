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

[states]
dir = "assets/states"

[study]
arms = ["frs", "mars-t"]
trials = 10
order = "shuffled"
max_steps = 600
out = "outputs/mars_franka/study"
seed = 0
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
    assert cfg.states.dir == EXAMPLE_DIR / "assets/states"
    assert cfg.policy.stats == EXAMPLE_DIR / "assets/droid/norm_stats.json"
    assert cfg.study.arms == ("frs", "mars-t")
    assert (cfg.study.trials, cfg.study.order, cfg.study.max_steps) == (10, "shuffled", 600)
    assert cfg.study.out == EXAMPLE_DIR / "outputs/mars_franka/study"
    assert cfg.study.seed == 0


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
        ('[states]\nout = "x"\n', "states", "out"),
        ('[user]\nstates_dir = "x"\n', "user", "states_dir"),
        ("[task]\nepochs = 3\n", "task", "epochs"),
        ("[training]\nepochs = 3\n", "training", "epochs"),
        ("[study]\nblocks = 2\n", "study", "blocks"),
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


def test_the_task_arm_tables_parse_with_their_adapter_and_inherit_the_steering_defaults(tmp_path):
    """``mars-t`` and the composed ``mars`` are ``frs`` tables whose adapters are the Phase B/C
    checkpoints; they inherit ``[steering]`` and resolve the adapter paths beside the example."""
    cfg = _load(
        tmp_path,
        "[steering]\nn_reversal_steps = 12\n\n"
        '[modes.mars-t]\nmode = "frs"\nadapter_task = "assets/task/pilot/task.pt"\n\n'
        '[modes.mars]\nmode = "frs"\nadapter_user = "assets/user/rotz20/user.pt"\n'
        'adapter_task = "assets/task/pilot/task.pt"\n',
    )
    assert set(cfg.modes) == {"mars-t", "mars"}
    assert cfg.modes["mars-t"].n_reversal_steps == 12 and cfg.modes["mars-t"].pinned_steps == 8
    assert cfg.modes["mars-t"].adapter_task == EXAMPLE_DIR / "assets/task/pilot/task.pt"
    assert cfg.modes["mars-t"].adapter_user is None
    assert cfg.modes["mars"].adapter_user == EXAMPLE_DIR / "assets/user/rotz20/user.pt"
    assert cfg.modes["mars"].adapter_task == EXAMPLE_DIR / "assets/task/pilot/task.pt"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ('[modes.mars-t]\nmode = "frs"\n', "adapter_task"),
        ('[modes.mars]\nmode = "frs"\n', "adapter_user"),
        ('[modes.mars]\nmode = "frs"\nadapter_task = "assets/task/t.pt"\n', "adapter_user"),
    ],
)
def test_the_phase_c_arms_must_name_their_modules(tmp_path, text, key):
    """An arm's name states its composition: `mars-t` is FRS+task and `mars` is user ∘ task, so a table
    without the module would silently report a plain FRS run under the arm's name."""
    with pytest.raises(ValueError, match=key) as error:
        _load(tmp_path, text)
    assert "[modes.mars" in str(error.value) and "train-" in str(error.value)


def test_a_custom_mode_name_needs_no_adapter(tmp_path):
    """Only the shipped arm names assert a composition; a study's own table is free to be plain FRS."""
    cfg = _load(tmp_path, '[modes.slow]\nmode = "frs"\n')
    assert cfg.modes["slow"].adapter_user is None and cfg.modes["slow"].adapter_task is None


def test_a_mode_none_table_may_name_the_task_module(tmp_path):
    """Phase D's generic dispatch runs the S2 "policy + task module, no steering" anchor as a
    ``mode = "none"`` table with ``adapter_task``: the task module runs on the base policy's own chunk.
    ``adapter_user`` (the reversal's correction) still has nothing to act on there."""
    cfg = _load(
        tmp_path,
        '[modes.task-only]\nmode = "none"\nadapter_task = "assets/task/pilot/task.pt"\n',
    )
    assert cfg.modes["task-only"].mode == "none"
    assert cfg.modes["task-only"].adapter_task == EXAMPLE_DIR / "assets/task/pilot/task.pt"
    assert cfg.modes["task-only"].adapter_user is None


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


def test_the_spacemouse_section_has_the_documented_defaults(tmp_path):
    """An absent section is the search defaults, which the probe is what verifies on hardware."""
    spacemouse = _load(tmp_path, "").spacemouse
    assert spacemouse == config_module.SpaceMouseConfig()
    assert (spacemouse.path, spacemouse.hid_id) == ("", "")
    assert (spacemouse.full_scale, spacemouse.rotation_full_scale) == (350.0, 350.0)
    assert spacemouse.source == (0, 1, 2, 3, 4, 5)
    assert spacemouse.sign == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_the_spacemouse_section_parses_a_path_mapping_and_scales(tmp_path):
    cfg = _load(
        tmp_path,
        "[spacemouse]\n"
        'path = "/dev/hidraw9"\n'
        'hid_id = "0000046D:0000C62B"\n'
        "full_scale = 300.0\n"
        "rotation_full_scale = 200.0\n"
        "source = [5, 4, 3, 2, 1, 0]\n"
        "sign = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]\n",
    )
    spacemouse = cfg.spacemouse
    assert (spacemouse.path, spacemouse.hid_id) == ("/dev/hidraw9", "0000046D:0000C62B")
    assert (spacemouse.full_scale, spacemouse.rotation_full_scale) == (300.0, 200.0)
    assert spacemouse.source == (5, 4, 3, 2, 1, 0)
    assert spacemouse.sign == (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[spacemouse]\nsource = [0, 1, 2, 3, 4]\n", "source"),
        ("[spacemouse]\nsource = [0, 1, 2, 3, 4, 6]\n", "source"),
        ("[spacemouse]\nsource = [0, 1, 2, 3, 4, -1]\n", "source"),
        ("[spacemouse]\nsource = [0, 1, 2, 3, 4, 5.0]\n", "source"),
        ("[spacemouse]\nsign = [1.0, 1.0, 1.0]\n", "sign"),
        ('[spacemouse]\nsign = [1.0, 1.0, 1.0, 1.0, 1.0, "up"]\n', "sign"),
        ("[spacemouse]\nfull_scale = 0.0\n", "full_scale"),
        ("[spacemouse]\nrotation_full_scale = -1.0\n", "rotation_full_scale"),
        ('[spacemouse]\nbutton = "left"\n', "button"),
    ],
)
def test_the_spacemouse_section_validates_its_fields(tmp_path, text, key):
    with pytest.raises(ValueError, match=key):
        _load(tmp_path, text)


def test_the_states_section_defaults_to_the_assets_states_dir(tmp_path):
    """A Phase B state file lands under assets/states unless a variant config says otherwise."""
    states = _load(tmp_path, "").states
    assert states == config_module.StatesConfig()
    assert states.dir == EXAMPLE_DIR / "assets" / "states"


def test_the_states_section_parses_its_dir_beside_the_example(tmp_path):
    cfg = _load(tmp_path, '[states]\ndir = "assets/states/pilot"\n')
    assert cfg.states.dir == EXAMPLE_DIR / "assets" / "states" / "pilot"


# ---------------------------------------------------------------- Phase B: the user module's tables


def test_the_user_section_has_the_phase_b_defaults(tmp_path):
    """An absent ``[user]`` is the study's own calibration session: three cues, the measured noise
    range, a 256-wide module on the checkpoint's chunk, and no per-step or cross-step sharing."""
    user = _load(tmp_path, "").user
    assert user == config_module.UserConfig()
    assert user.states == EXAMPLE_DIR / "assets" / "states" / "pilot"
    assert user.out == EXAMPLE_DIR / "assets" / "user" / "rotz20"
    assert user.checkpoint == EXAMPLE_DIR / "assets" / "user" / "rotz20" / "user.pt"
    assert (user.cues, user.sigma) == (3, (0.2, 0.5))
    assert (user.horizon, user.step_rank, user.correct_by) == (None, None, None)
    assert (user.d, user.tdim, user.time_embedding) == (256, 64, "mars")
    assert (user.cross_step_rank, user.additive) == (0, False)


def test_the_user_section_parses_the_trainers_knobs(tmp_path):
    cfg = _load(
        tmp_path,
        "[user]\n"
        'states = "assets/states/pilot"\n'
        'out = "assets/user/rotz20"\n'
        'checkpoint = "assets/user/rotz20/user.pt"\n'
        "cues = 5\n"
        "sigma = [0.1, 0.4]\n"
        "horizon = 10\n"
        "d = 128\n"
        "tdim = 32\n"
        'time_embedding = "pi05"\n'
        "step_rank = 3\n"
        "cross_step_rank = 2\n"
        "additive = true\n"
        "correct_by = 4\n",
    )
    user = cfg.user
    assert user.states == EXAMPLE_DIR / "assets" / "states" / "pilot"
    assert user.out == EXAMPLE_DIR / "assets" / "user" / "rotz20"
    assert user.checkpoint == EXAMPLE_DIR / "assets" / "user" / "rotz20" / "user.pt"
    assert (user.cues, user.sigma) == (5, (0.1, 0.4))
    assert (user.horizon, user.step_rank, user.correct_by) == (10, 3, 4)
    assert (user.d, user.tdim, user.time_embedding) == (128, 32, "pi05")
    assert (user.cross_step_rank, user.additive) == (2, True)


def test_the_training_section_has_the_phase_b_defaults(tmp_path):
    training = _load(tmp_path, "").training
    assert training == config_module.TrainingConfig()
    assert (training.steps, training.batch, training.lr) == (3000, 8, 1e-3)
    assert (training.val_frac, training.seed, training.log_every) == (0.1, 0, 100)
    assert (training.val_depths, training.device) == ((10, 4), "cuda")


def test_the_training_section_parses_the_loop_and_the_device(tmp_path):
    cfg = _load(
        tmp_path,
        "[training]\n"
        "steps = 10\n"
        "batch = 2\n"
        "lr = 0.5\n"
        "val_frac = 0.25\n"
        "seed = 7\n"
        "log_every = 2\n"
        "val_depths = [3, 6]\n"
        'device = "cpu"\n',
    )
    training = cfg.training
    assert (training.steps, training.batch, training.lr) == (10, 2, 0.5)
    assert (training.val_frac, training.seed, training.log_every) == (0.25, 7, 2)
    assert training.val_depths == (3, 6)
    assert training.device == "cpu"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[user]\nsigma = [0.5, 0.2]\n", "sigma"),  # low > high
        ("[user]\nsigma = [0.2]\n", "sigma"),  # not a pair
        ("[user]\nsigma = [-0.1, 0.5]\n", "sigma"),  # negative noise
        ("[user]\nsigma = [0.2, 0.5, 0.7]\n", "sigma"),
        ("[user]\ncues = 0\n", "cues"),
        ("[user]\nstep_rank = 0\n", "step_rank"),
        ("[user]\nhorizon = 0\n", "horizon"),
        ("[user]\ncorrect_by = 0\n", "correct_by"),
        ("[user]\nd = 0\n", "d"),
        ('[user]\ntime_embedding = "siren"\n', "time_embedding"),
        ("[user]\nadditive = 1\n", "additive"),
        ("[user]\ncross_step_rank = -1\n", "cross_step_rank"),
        ("[training]\nsteps = 0\n", "steps"),
        ("[training]\nbatch = 0\n", "batch"),
        ("[training]\nlr = 0.0\n", "lr"),
        ("[training]\nval_frac = 0.0\n", "val_frac"),
        ("[training]\nval_frac = 1.5\n", "val_frac"),
        ("[training]\nseed = -1\n", "seed"),
        ("[training]\nlog_every = 0\n", "log_every"),
        ("[training]\nval_depths = []\n", "val_depths"),
        ("[training]\nval_depths = [0]\n", "val_depths"),
    ],
)
def test_the_phase_b_values_are_checked_and_named(tmp_path, text, key):
    with pytest.raises(ValueError, match=key):
        _load(tmp_path, text)


# ---------------------------------------------------------------- Phase C: the task module's table


def test_the_task_section_has_the_phase_c_defaults(tmp_path):
    """An absent ``[task]`` is the shipped study's module: the pilot's demonstrations, the VLM image
    scene at main's render size, the shared identity pull and the module's default width."""
    task = _load(tmp_path, "").task
    assert task == config_module.TaskConfig()
    assert task.demos == EXAMPLE_DIR / "assets" / "demos" / "pilot"
    assert task.out == EXAMPLE_DIR / "assets" / "task" / "pilot"
    assert task.checkpoint == EXAMPLE_DIR / "assets" / "task" / "pilot" / "task.pt"
    assert (task.token_source, task.render_size, task.scene_rank) == ("vlm_image", 224, None)
    assert (task.d, task.tdim, task.lambda_id) == (128, 64, 1e-3)


def test_the_task_section_parses_the_trainers_knobs(tmp_path):
    cfg = _load(
        tmp_path,
        "[task]\n"
        'demos = "assets/demos/push"\n'
        'out = "assets/task/push"\n'
        'checkpoint = "assets/task/push/task.pt"\n'
        'token_source = "vlm_image"\n'
        "render_size = 256\n"
        "scene_rank = 64\n"
        "d = 64\n"
        "tdim = 32\n"
        "lambda_id = 0.5\n",
    )
    task = cfg.task
    assert task.demos == EXAMPLE_DIR / "assets" / "demos" / "push"
    assert task.out == EXAMPLE_DIR / "assets" / "task" / "push"
    assert task.checkpoint == EXAMPLE_DIR / "assets" / "task" / "push" / "task.pt"
    assert (task.token_source, task.render_size, task.scene_rank) == ("vlm_image", 256, 64)
    assert (task.d, task.tdim, task.lambda_id) == (64, 32, 0.5)


def test_the_task_sections_paths_resolve_beside_the_example(tmp_path):
    cfg = _load(tmp_path, '[task]\ndemos = "/an/absolute/dataset"\n')
    assert cfg.task.demos == Path("/an/absolute/dataset")
    assert cfg.task.out == EXAMPLE_DIR / "assets" / "task" / "pilot"


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[task]\nrender_size = 0\n", "render_size"),  # the VLM resizes it; it must be a positive side
        ("[task]\nrender_size = 2.5\n", "render_size"),
        ("[task]\nscene_rank = 0\n", "scene_rank"),  # the reduction is at least one token wide
        ("[task]\nd = 0\n", "d"),
        ("[task]\ntdim = 0\n", "tdim"),
        ("[task]\nlambda_id = -0.1\n", "lambda_id"),  # a pull cannot push
        ('[task]\ntoken_source = ""\n', "token_source"),
        ("[task]\ntoken_source = 3\n", "token_source"),
    ],
)
def test_the_phase_c_values_are_checked_and_named(tmp_path, text, key):
    with pytest.raises(ValueError, match=key):
        _load(tmp_path, text)


# ---------------------------------------------------------------- Phase D: the study's block


def test_the_study_section_has_the_phase_d_defaults(tmp_path):
    """An absent ``[study]`` is a config with no block to run: the shipped values' round count and cap,
    no arms, and the run directories under ``outputs/mars_franka/study`` beside the example."""
    study = _load(tmp_path, "").study
    assert study == config_module.StudyConfig()
    assert study.arms == ()
    assert (study.trials, study.order, study.max_steps) == (10, "shuffled", 600)
    assert study.out == EXAMPLE_DIR / "outputs" / "mars_franka" / "study"
    assert study.seed == 0


def test_the_study_section_parses_a_block_with_its_arms_and_its_cap(tmp_path):
    """Arms are command names: the fixed session commands and/or this config's own mode tables."""
    cfg = _load(
        tmp_path,
        "[study]\n"
        'arms = ["autonomous", "teleop", "task-only", "frs"]\n'
        "trials = 4\n"
        'order = "blocked"\n'
        "max_steps = 60\n"
        'out = "outputs/s2"\n'
        "seed = 3\n\n"
        '[modes.frs]\nmode = "frs"\n\n'
        '[modes.task-only]\nmode = "none"\nadapter_task = "assets/task/pilot/task.pt"\n',
    )
    study = cfg.study
    assert study.arms == ("autonomous", "teleop", "task-only", "frs")
    assert (study.trials, study.order, study.max_steps, study.seed) == (4, "blocked", 60, 3)
    assert study.out == EXAMPLE_DIR / "outputs" / "s2"


def test_a_study_arm_must_resolve_to_a_command_or_a_mode_table(tmp_path):
    """Each arm is checked when the block is read: the bad one is named, with what the config has."""
    with pytest.raises(ValueError, match="'fly'") as error:
        _load(tmp_path, '[study]\narms = ["fly"]\n')
    message = str(error.value)
    assert "autonomous" in message and "not a command that runs" in message
    assert "[modes" in message

    # a fixed steered command runs only through its own table: the message names the table to add
    with pytest.raises(ValueError, match="'frs'") as error:
        _load(tmp_path, '[study]\narms = ["frs"]\n')
    assert "[modes.frs]" in str(error.value) and "add one" in str(error.value)

    # teleop and autonomous need no table; a mode table resolves under its own name
    assert _load(tmp_path, '[study]\narms = ["autonomous", "teleop"]\n').study.arms == (
        "autonomous",
        "teleop",
    )
    resolved = _load(tmp_path, '[study]\narms = ["task-only"]\n[modes.task-only]\nmode = "none"\n')
    assert resolved.study.arms == ("task-only",)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("[study]\narms = []\n", "arms"),  # non-empty: a block needs somewhere to put its trials
        ("[study]\narms = [3]\n", "arms"),
        ('[study]\narms = [""]\n', "arms"),
        ("[study]\ntrials = 2\n", "arms"),  # arms are not optional when the table is present
        ('[study]\narms = ["autonomous"]\ntrials = 0\n', "trials"),
        ('[study]\narms = ["autonomous"]\norder = "random"\n', "order"),
        ('[study]\narms = ["autonomous"]\nmax_steps = 0\n', "max_steps"),
        ('[study]\narms = ["autonomous"]\nseed = -1\n', "seed"),
    ],
)
def test_the_phase_d_values_are_checked_and_named(tmp_path, text, key):
    with pytest.raises(ValueError, match=key):
        _load(tmp_path, text)
