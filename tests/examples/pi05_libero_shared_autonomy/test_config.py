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
    spec_label,
)

from lerobot.policies.pi05.steering import rotation_about_z

CONDITIONS = sorted(p.name for p in (CONFIG_DIR / "experiment").glob("*.yaml") if p.name != "base.yaml")


def test_deep_merge_overlay_wins_and_null_resets():
    base = {"control": {"mode": "policy", "tau": 5}, "prompt": "x"}
    merged = deep_merge(base, {"control": {"tau": None, "n_reverse_steps": 3}})
    assert merged == {"control": {"mode": "policy", "tau": None, "n_reverse_steps": 3}, "prompt": "x"}
    assert base["control"]["tau"] == 5  # not mutated


def test_flatten_rejects_unknown_keys_and_skips_null():
    schema = {"control": {"mode": "mode"}, "prompt": "prompt"}
    assert flatten({"control": {"mode": "teleop"}, "prompt": None}, schema, "f") == {"mode": "teleop"}
    with pytest.raises(ValueError, match="unknown key 'controls'"):
        flatten({"controls": {}}, schema, "f")
    with pytest.raises(ValueError, match="control.tau"):
        flatten({"control": {"tau": 1}}, schema, "f")
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


def test_condition_matrices_resolve():
    s = load_experiment_settings(CONFIG_DIR / "experiment" / "reverse_flow_5steps_rotz20.yaml", {})
    c = s.session.control
    assert c.n_reverse_steps == 5
    np.testing.assert_allclose(c.corruption_matrix, rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[:3, :3], rotation_about_z(20))
    np.testing.assert_allclose(c.reversal_adapter_matrix[3:, :], 0)
    assert spec_label(c.corruption) == "rotation_z_deg=20"
    clean = load_experiment_settings(CONFIG_DIR / "experiment" / "flow_control_tau8.yaml", {})
    assert clean.session.control.tau == 8 and clean.session.control.corruption_matrix is None


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
    with pytest.raises(ValueError, match="n_reverse_steps"):
        load_experiment_settings(write("control: {n_reverse_steps: 0}"), {})


def test_experiment_cli_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("prompt: task\n")
    s = load_experiment_settings(p, {"n_trials": 2, "mode": "teleop", "port": 9000, "output_dir": "o"})
    assert s.n_trials == 2 and s.session.control.mode == "teleop" and s.session.port == 9000
    assert s.output_dir == Path("o") and s.prompt == "task"


def test_base_yaml_matches_dataclass_defaults():
    data = yaml.safe_load((CONFIG_DIR / "experiment" / "base.yaml").read_text())
    assert set(data) == {"experiment", "scene", "prompt", "policy", "control", "server"}
