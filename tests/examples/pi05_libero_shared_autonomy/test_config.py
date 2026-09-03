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
    load_experiment_settings,
    load_interactive_settings,
    load_yaml_with_extends,
    parse_set,
    set_label,
    sets_to_tree,
    spec_label,
)

from lerobot.policies.pi05.steering import rotation_about_z

CONDITIONS = sorted(p.name for p in (CONFIG_DIR / "experiment").glob("*.yaml") if p.name != "base.yaml")


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


def test_arm_files_resolve():
    adapted = load_experiment_settings(CONFIG_DIR / "experiment" / "flow_reversal_rotz20_adapted.yaml", {})
    c = adapted.session.control
    assert c.mode == "shared_flow_reversal_steering" and c.n_reversal_steps is None
    np.testing.assert_allclose(c.corruption_matrix, rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[:3, :3], rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[3:, :], 0)
    assert spec_label(c.corruption) == "rotation_z_deg=20"
    native = load_experiment_settings(CONFIG_DIR / "experiment" / "flow_reversal_rotz20.yaml", {})
    assert native.session.control.reversal_adapter_matrix is None
    np.testing.assert_allclose(native.session.control.corruption_matrix, rotation_about_z(20))
    fc = load_experiment_settings(CONFIG_DIR / "experiment" / "flow_control_rotz20.yaml", {})
    assert fc.session.control.mode == "shared_flow_control" and fc.session.control.n_guided_steps == 8


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
    assert s.session.control.n_guided_steps == 4 and s.session.control.corruption_matrix is None
    s = load_experiment_settings(
        path, {}, sets=["control.mode=shared_flow_reversal_steering", "control.n_reversal_steps=5"]
    )
    assert s.session.control.n_reversal_steps == 5
    with pytest.raises(ValueError, match="unknown key"):
        load_experiment_settings(path, {}, sets=["control.depth=4"])
    with pytest.raises(ValueError, match="control.mode"):
        load_experiment_settings(path, {}, sets=["control.mode=nope"])


def test_experiment_validation_errors(tmp_path):
    def write(text):
        p = tmp_path / "c.yaml"
        p.write_text(text)
        return p

    with pytest.raises(ValueError, match="n_trials"):
        load_experiment_settings(write("experiment: {n_trials: 0}"), {})
    with pytest.raises(ValueError, match="control.mode"):
        load_experiment_settings(write("control: {mode: nope}"), {})
    with pytest.raises(ValueError, match="task_order"):
        load_experiment_settings(write("experiment: {task_order: backwards}"), {})
    with pytest.raises(ValueError, match="prompt"):
        load_experiment_settings(write("prompt: ''"), {})
    with pytest.raises(ValueError, match="corruption"):
        load_experiment_settings(write("control: {reversal_adapter: {translation: corruption}}"), {})
    with pytest.raises(ValueError, match="n_reversal_steps"):
        load_experiment_settings(write("control: {n_reversal_steps: 0}"), {})


def test_experiment_cli_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("prompt: task\n")
    s = load_experiment_settings(p, {"n_trials": 2, "mode": "teleop", "port": 9000, "output_dir": "o"})
    assert s.n_trials == 2 and s.session.control.mode == "teleop" and s.session.port == 9000
    assert s.output_dir == Path("o") and s.prompt == "task"


def test_base_yaml_matches_dataclass_defaults():
    data = yaml.safe_load((CONFIG_DIR / "experiment" / "base.yaml").read_text())
    assert set(data) == {"experiment", "scene", "prompt", "policy", "control", "server"}
