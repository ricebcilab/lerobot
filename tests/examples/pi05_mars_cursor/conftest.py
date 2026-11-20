"""Make the MARS cursor example's modules importable (they are scripts, not a package), and keep torch's threads few."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

MARS_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars"
for path in (
    MARS_DIR / "cursor",
    MARS_DIR / "cursor" / "tools",
    MARS_DIR / "cursor" / "reports",
    MARS_DIR,
):  # cursor_sa and the scripts; `paper.figures` lives under MARS_DIR
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    """The tests' tensors are tiny: many threads only contend (on a loaded machine, by two orders of magnitude). Four
    ran the simulated sweeps a fifth faster than sixteen, and faster than one."""
    torch.set_num_threads(int(os.environ.get("MARS_TEST_THREADS", min(4, os.cpu_count() or 4))))


@pytest.fixture
def converter():
    """The converter script (`tools/convert_checkpoints.py`), the one reader of the cursor project's formats."""
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoints", MARS_DIR / "cursor" / "tools" / "convert_checkpoints.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
