from pathlib import Path

import numpy as np
import pytest
import yaml
from config import (
    CONFIG_DIR,
    DEFAULT_INTERACTIVE_CONFIG,
    MODES,
    ExperimentSettings,
    SessionSettings,
    deep_merge,
    flatten,
    largest_gain,
    load_experiment_settings,
    load_interactive_settings,
    load_yaml_with_extends,
    parse_set,
    set_label,
    sets_to_tree,
    spec_label,
)

from lerobot.policies.pi05.steering import rotation_about_z

CONDITIONS = sorted(
    p.name for p in (CONFIG_DIR / "experiment").glob("*.yaml") if not p.name.startswith("base")
)


def _load(tmp_path, text, **kwargs):
    """The control settings of an experiment config written from `text`."""
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return load_experiment_settings(path, {}, **kwargs).session.control


def test_deep_merge_overlay_wins_and_null_resets():
    base = {"control": {"mode": "policy", "n_guided_steps": 5}, "prompt": "x"}
    merged = deep_merge(base, {"control": {"n_guided_steps": None, "n_reversal_steps": 3}})
    assert merged == {
        "control": {"mode": "policy", "n_guided_steps": None, "n_reversal_steps": 3},
        "prompt": "x",
    }
    assert base["control"]["n_guided_steps"] == 5  # not mutated


def test_flatten_rejects_unknown_keys_and_skips_null():
    schema = {"control": {"mode": "mode"}, "prompt": "prompt"}
    assert flatten({"control": {"mode": "teleop"}, "prompt": None}, schema, "f") == {"mode": "teleop"}
    with pytest.raises(ValueError, match="unknown key 'controls'"):
        flatten({"controls": {}}, schema, "f")
    with pytest.raises(ValueError, match="control.n_guided_steps"):
        flatten({"control": {"n_guided_steps": 1}}, schema, "f")
    with pytest.raises(ValueError, match="must be a mapping"):
        flatten({"control": 3}, schema, "f")


def test_extends_resolves_relative_to_the_file(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: {c: 2, d: 3}\n")
    (tmp_path / "cond.yaml").write_text("extends: base.yaml\nb: {d: 4}\n")
    assert load_yaml_with_extends(tmp_path / "cond.yaml") == {"a": 1, "b": {"c": 2, "d": 4}}
    (tmp_path / "loop.yaml").write_text("extends: loop.yaml\n")
    with pytest.raises(ValueError, match="extends"):
        load_yaml_with_extends(tmp_path / "loop.yaml")


def test_spec_label():
    assert spec_label(None) is None
    assert spec_label({"rotation_z_deg": 20}) == "rotation_z_deg=20"
    assert spec_label({"translation": "corruption", "orientation": "zero", "gripper": "zero"}) == (
        "translation=corruption, orientation=zero, gripper=zero"
    )
    assert spec_label({"M": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}) == "M=literal"


def test_shipped_interactive_config_loads():
    s = load_interactive_settings(DEFAULT_INTERACTIVE_CONFIG, {})
    assert isinstance(s, SessionSettings)
    assert s.control.mode == "policy" and s.control.corruption_matrix is None
    assert s.output_dir == Path("outputs/pi05_libero_interactive")
    assert s.control.resolved_input_gain is not None  # the REPL cannot clip either


def test_interactive_cli_overrides_win_over_file():
    s = load_interactive_settings(DEFAULT_INTERACTIVE_CONFIG, {"task_id": 3, "mode": "teleop"})
    assert s.task_id == 3 and s.control.mode == "teleop"
    s = load_interactive_settings(None, {})
    assert s.suite == "libero_spatial"


@pytest.mark.parametrize("name", CONDITIONS)
def test_every_shipped_condition_loads(name):
    s = load_experiment_settings(CONFIG_DIR / "experiment" / name, {})
    assert isinstance(s, ExperimentSettings)
    assert s.name == Path(name).stem
    assert s.session.control.mode in MODES
    assert s.session.suite == "libero_goal" and s.seed == 0 and s.prompt == "do something"
    # every shipped condition turns the command down enough that its corruption cannot clip
    control = s.session.control
    assert control.resolved_input_gain is not None
    assert control.resolved_input_gain * largest_gain(control) <= 1.0


def test_arm_files_resolve():
    mars = load_experiment_settings(CONFIG_DIR / "experiment" / "mars_rotz20.yaml", {})
    c = mars.session.control
    assert c.mode == "shared_flow_reversal_steering" and c.n_reversal_steps is None
    np.testing.assert_allclose(c.corruption_matrix, rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[:3, :3], rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[3:, 3:], np.eye(4))  # orientation/gripper: identity
    np.testing.assert_allclose(c.reversal_adapter_matrix[:3, 3:], 0)
    np.testing.assert_allclose(c.reversal_adapter_matrix[3:, :3], 0)
    assert spec_label(c.corruption) == "rotation_z_deg=20"
    native = load_experiment_settings(CONFIG_DIR / "experiment" / "flow_reversal_rotz20.yaml", {})
    assert native.session.control.reversal_adapter_matrix is None
    np.testing.assert_allclose(native.session.control.corruption_matrix, rotation_about_z(20))
    fc = load_experiment_settings(CONFIG_DIR / "experiment" / "flow_control_rotz20.yaml", {})
    assert fc.session.control.mode == "shared_flow_control"
    assert (fc.session.control.tau, fc.session.control.lam, fc.session.control.ic_gain) == (None, 1, "action")
    learned = load_experiment_settings(
        CONFIG_DIR / "experiment" / "mars_learned_rotz20.yaml", {}
    ).session.control
    assert learned.reversal_adapter_user_path.endswith("user.pt") and learned.reversal_adapter_matrix is None


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_flow_control_knobs_and_the_old_name(tmp_path):
    arm = _write(
        tmp_path, "fc.yaml", "control:\n  mode: shared_flow_control\n  tau: 25\n  lambda: 2\n  ic_gain: 2.0\n"
    )
    c = load_experiment_settings(arm, {}).session.control
    assert (c.tau, c.lam, c.ic_gain) == (25, 2, 2.0)
    old = _write(tmp_path, "old.yaml", "control:\n  mode: shared_flow_control\n  n_guided_steps: 3\n")
    assert load_experiment_settings(old, {}).session.control.lam == 3  # the old key is lambda
    assert load_experiment_settings(arm, {}, sets=["control.lambda=0"]).session.control.lam == 0
    for bad, match in (
        ("  tau: 0\n", "control.tau"),
        ("  tau: 2.5\n", "control.tau"),
        ("  lambda: -1\n", "control.lambda"),
        ("  ic_gain: 0\n", "control.ic_gain"),
        ("  ic_gain: noise\n", "control.ic_gain"),
    ):
        with pytest.raises(ValueError, match=match):
            load_experiment_settings(
                _write(tmp_path, "bad.yaml", "control:\n  mode: shared_flow_control\n" + bad), {}
            )


def test_learned_and_composed_adapter_specs(tmp_path):
    from lerobot.policies.pi05.steering import LearnedTaskModule, LearnedUserModule, save_adapter_module

    user = save_adapter_module(LearnedUserModule(horizon=4, d=8, tdim=4), tmp_path / "user.pt", kind="user")
    task = save_adapter_module(LearnedTaskModule(horizon=4, d=8, tdim=4), tmp_path / "task.pt", kind="task")
    base = "control:\n  mode: shared_flow_reversal_steering\n  corruption: {rotation_z_deg: 20}\n"
    c = load_experiment_settings(
        _write(tmp_path, "a.yaml", base + f"  reversal_adapter: {{learned: {user}}}\n"), {}
    ).session.control
    assert c.reversal_adapter_user_path == str(user) and c.reversal_adapter_task_path is None
    assert c.reversal_adapter_matrix is None
    spec = f"  reversal_adapter: {{user: {{translation: corruption}}, task: {task}}}\n"
    c = load_experiment_settings(_write(tmp_path, "b.yaml", base + spec), {}).session.control
    np.testing.assert_allclose(c.reversal_adapter_matrix[:3, :3], rotation_about_z(20))
    assert c.reversal_adapter_task_path == str(task) and c.reversal_adapter_user_path is None
    spec = f"  reversal_adapter: {{user: {{learned: {user}}}, task: {task}}}\n"
    c = load_experiment_settings(_write(tmp_path, "c.yaml", base + spec), {}).session.control
    assert (c.reversal_adapter_user_path, c.reversal_adapter_task_path) == (str(user), str(task))
    for bad in ("  reversal_adapter: {learned: 1}\n", "  reversal_adapter: {user: {learned: u.pt}, F: []}\n"):
        with pytest.raises(ValueError):
            load_experiment_settings(_write(tmp_path, "d.yaml", base + bad), {})


def test_curl_corruption_resolves():
    from teleop import Curl

    text = (
        "control:\n  mode: shared_flow_reversal_steering\n  input_gain: auto\n"
        "  corruption: {curl: 20, tau: 0.1}\n  reversal_adapter: {translation: corruption_inverse}\n"
    )
    path = CONFIG_DIR.parent / "_curl_test.yaml"
    path.write_text(text)
    try:
        c = load_experiment_settings(path, {}).session.control
    finally:
        path.unlink()
    assert (
        c.corruption_curl == Curl(20.0, 0.1) and c.corruption_matrix is None and c.corruption_family is None
    )
    assert c.resolved_input_gain == 1.0  # a curl never lengthens a push
    np.testing.assert_allclose(
        c.reversal_adapter_matrix[:3, :3], np.linalg.inv(Curl(20.0, 0.1).steady_state())
    )
    assert "curl=20" in spec_label(c.corruption)


def test_parse_set_and_tree():
    assert parse_set("control.n_guided_steps=4") == (["control", "n_guided_steps"], 4)
    assert parse_set("control.corruption=null") == (["control", "corruption"], None)
    assert parse_set("control.reversal_adapter={translation: corruption, gripper: zero}") == (
        ["control", "reversal_adapter"],
        {"translation": "corruption", "gripper": "zero"},
    )
    assert sets_to_tree(["control.n_guided_steps=4", "experiment.seed=3", "control.n_guided_steps=6"]) == {
        "control": {"n_guided_steps": 6},
        "experiment": {"seed": 3},
    }
    for bad in ("n_guided_steps", "=4", "control.bad key=1"):
        with pytest.raises(ValueError, match="--set"):
            parse_set(bad)
    assert (
        set_label(["control.n_guided_steps=4", "control.corruption=null"])
        == "n_guided_steps-4_corruption-null"
    )


def test_sets_override_the_file_and_are_validated():
    path = CONFIG_DIR / "experiment" / "flow_control_rotz20.yaml"
    s = load_experiment_settings(path, {}, sets=["control.n_guided_steps=4", "control.corruption=null"])
    assert s.session.control.lam == 4 and s.session.control.corruption_matrix is None  # the old key is lambda
    s = load_experiment_settings(
        path, {}, sets=["control.mode=shared_flow_reversal_steering", "control.n_reversal_steps=5"]
    )
    assert s.session.control.n_reversal_steps == 5
    with pytest.raises(ValueError, match="unknown key"):
        load_experiment_settings(path, {}, sets=["control.depth=4"])
    with pytest.raises(ValueError, match="control.mode"):
        load_experiment_settings(path, {}, sets=["control.mode=nope"])


def test_policy_operator_timing_overrides_and_defaults():
    path = CONFIG_DIR / "experiment" / "flow_control_rotz20.yaml"
    default = load_experiment_settings(path, {}, sets=["operator=policy"])
    assert default.session.policy_operator.update_every_steps is None
    assert default.session.policy_operator.delay_steps == 0
    sets = ["policy_operator.update_every_steps=20", "policy_operator.delay_steps=4"]
    settings = load_experiment_settings(path, {}, sets=["operator=policy", *sets])
    assert settings.session.policy_operator.update_every_steps == 20
    assert settings.session.policy_operator.delay_steps == 4
    with pytest.raises(ValueError, match="require operator=policy"):
        load_experiment_settings(path, {}, sets=sets)
    with pytest.raises(ValueError, match="positive integer"):
        load_experiment_settings(path, {}, sets=["operator=policy", "policy_operator.update_every_steps=0"])


def test_experiment_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="n_trials"):
        _load(tmp_path, "experiment: {n_trials: 0}")
    with pytest.raises(ValueError, match="control.mode"):
        _load(tmp_path, "control: {mode: nope}")
    with pytest.raises(ValueError, match="task_order"):
        _load(tmp_path, "experiment: {task_order: backwards}")
    with pytest.raises(ValueError, match="prompt"):
        _load(tmp_path, "prompt: ''")
    with pytest.raises(ValueError, match="corruption"):
        _load(tmp_path, "control: {reversal_adapter: {translation: corruption}}")
    with pytest.raises(ValueError, match="n_reversal_steps"):
        _load(tmp_path, "control: {n_reversal_steps: 0}")
    for value in (-1, 0.5, True):
        with pytest.raises(ValueError, match="seed"):
            _load(tmp_path, f"experiment: {{seed: {value}}}")


def test_experiment_cli_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("prompt: task\n")
    s = load_experiment_settings(p, {"n_trials": 2, "mode": "teleop", "port": 9000, "output_dir": "o"})
    assert s.n_trials == 2 and s.session.control.mode == "teleop" and s.session.port == 9000
    assert s.output_dir == Path("o") and s.prompt == "task"


def test_base_rotz20_yaml_matches_dataclass_defaults():
    data = yaml.safe_load((CONFIG_DIR / "experiment" / "base_rotz20.yaml").read_text())
    assert set(data) == {"experiment", "scene", "prompt", "policy", "control", "server"}


def test_operator_settings():
    path = CONFIG_DIR / "experiment" / "flow_reversal_rotz20.yaml"
    default = load_experiment_settings(path, {})
    assert default.session.operator == "human"
    assert default.session.control.operator_dims == ["translation"]
    s = load_experiment_settings(
        path, {}, sets=["operator=policy", "control.operator_dims=[translation, gripper]"]
    )
    assert s.session.operator == "policy"
    assert s.session.control.operator_dims == ["translation", "gripper"]
    with pytest.raises(ValueError, match="operator"):
        load_experiment_settings(path, {}, sets=["operator=robot"])
    with pytest.raises(ValueError, match="operator_dims"):
        load_experiment_settings(path, {}, sets=["control.operator_dims=[wrist]"])
    with pytest.raises(ValueError, match="operator_dims"):
        load_experiment_settings(path, {}, sets=["control.operator_dims=translation"])


# ---------------------------------------------------------------- random corruption and input gain


def test_random_corruption_spec_becomes_a_family_and_leaves_the_matrices_to_each_trial(tmp_path):
    from teleop import RandomCorruption

    control = _load(
        tmp_path,
        "control: {mode: shared_flow_reversal_steering, reversal_adapter: {translation: corruption},"
        " corruption: {random: {rotation_deg: [30, 90], scale: [1, 4]}}}",
    )
    assert control.corruption_family == RandomCorruption(rotation_deg=(30, 90), scale=(1, 4))
    assert control.corruption_matrix is None and control.reversal_adapter_matrix is None
    assert spec_label(control.corruption) == "random(rotation_deg=30..90, scale=1..4)"
    # either part may be left out, and a fixed corruption has no family
    only_rotation = _load(tmp_path, "control: {corruption: {random: {rotation_deg: [0, 45]}}}")
    assert only_rotation.corruption_family == RandomCorruption(rotation_deg=(0, 45))
    assert _load(tmp_path, "control: {corruption: {rotation_z_deg: 20}}").corruption_family is None


@pytest.mark.parametrize(
    "corruption,complaint",
    [
        ("{random: {angle_deg: [0, 30]}}", "angle_deg"),  # unknown key, named
        ("{random: {rotation_deg: [90, 30]}}", "rotation_deg must be"),  # reversed range
        ("{random: {scale: [0, 2]}}", "scale must be"),  # a zero gain
        ("{random: {scale: [1, 2]}, rotation_z_deg: 20}", "cannot be combined"),  # mixed with a fixed form
        ("{random: [1, 2]}", "'random' must be a mapping"),
    ],
)
def test_random_corruption_spec_errors_name_the_setting_and_the_problem(tmp_path, corruption, complaint):
    with pytest.raises(ValueError, match=f"control.corruption.*{complaint}"):
        _load(tmp_path, f"control: {{corruption: {corruption}}}")


def test_a_random_corruption_needs_trials_so_the_interactive_runner_rejects_it(tmp_path):
    path = tmp_path / "i.yaml"
    path.write_text("control: {corruption: {random: {scale: [1, 2]}}}")
    with pytest.raises(ValueError, match="control.corruption.*one per trial"):
        load_interactive_settings(path, {})


@pytest.mark.parametrize(
    "corruption,largest_safe_gain",
    [
        ("{random: {rotation_deg: [30, 90], scale: [1, 4]}}", 0.25),
        ("{scale: [4, 1, 1]}", 0.25),
        ("{rotation_z_deg: 40}", 1.0),
        ("null", 1.0),
        ("{scale: [0.5, 0.5, 0.5]}", 1.0),  # a corruption that only shrinks: the gain never exceeds 1
    ],
)
def test_input_gain_auto_is_the_largest_gain_that_cannot_clip(tmp_path, corruption, largest_safe_gain):
    control = _load(tmp_path, f"control: {{corruption: {corruption}, input_gain: auto}}")
    assert control.resolved_input_gain == pytest.approx(largest_safe_gain)


def test_input_gain_is_off_by_default_and_a_number_is_checked_against_the_corruption(tmp_path):
    assert _load(tmp_path, "control: {corruption: {scale: [4, 1, 1]}}").resolved_input_gain is None
    control = _load(tmp_path, "control: {corruption: {scale: [4, 1, 1]}, input_gain: 0.2}")
    assert control.resolved_input_gain == pytest.approx(0.2)
    with pytest.raises(ValueError, match=r"control.input_gain.*0\.25"):  # names the largest safe gain
        _load(tmp_path, "control: {corruption: {scale: [4, 1, 1]}, input_gain: 0.5}")
    for bad in ("0", "-1", "1.5", "fast", "true", "[0.5]"):
        with pytest.raises(ValueError, match="control.input_gain"):
            _load(tmp_path, f"control: {{input_gain: {bad}}}")


@pytest.mark.parametrize("arm", ["flow_control", "flow_reversal", "mars"])
def test_the_shipped_random_family_cannot_clip_and_only_the_adapted_arm_has_an_adapter(arm):
    control = load_experiment_settings(CONFIG_DIR / "experiment" / f"{arm}_random.yaml", {}).session.control
    assert control.corruption_family is not None
    # the gain in force times the most the family can lengthen a command stays inside the unit cube
    assert control.resolved_input_gain * control.corruption_family.max_gain <= 1.0
    assert (control.reversal_adapter is not None) == (arm == "mars")
