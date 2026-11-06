"""The cursor example keeps only its two entry points at the top; tools/ and reports/ hold the rest."""

import importlib
from pathlib import Path

import pytest

CURSOR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "cursor"


def py_files(folder):
    return {p.name for p in (CURSOR / folder).glob("*.py")}


@pytest.mark.parametrize(
    "name", ["diag_layouts", "report_grid_image", "hub", "convert_checkpoints", "analyze_results"]
)
def test_moved_scripts_import_by_bare_name(name):
    assert importlib.import_module(name).__file__


def test_folders_hold_exactly_their_scripts():
    assert py_files(".") == {"run_experiment.py", "run_sweep.py"}
    assert py_files("tools") == {"analyze_results.py", "convert_checkpoints.py", "hub.py", "diag_layouts.py"}
    assert py_files("reports") == {"build_study_report.py", "report_grid_image.py"}


def test_cursor_sa_has_no_env_shim_and_demos_share_the_package_layouts():
    import importlib.util

    import cursor_sa.demos

    import lerobot.envs.cursor

    assert importlib.util.find_spec("cursor_sa.env") is None
    assert cursor_sa.demos.Layouts is lerobot.envs.cursor.Layouts
