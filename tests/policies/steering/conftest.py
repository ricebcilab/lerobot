"""Keep torch's threads few for the steering tests (their tensors are tiny: many threads only contend)."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch


def pytest_configure(config):
    torch.set_num_threads(int(os.environ.get("MARS_TEST_THREADS", min(16, os.cpu_count() or 16))))


@pytest.fixture
def converter():
    """The MARS cursor example's converter script, the one reader of the cursor project's checkpoint formats."""
    path = Path(__file__).resolve().parents[3] / "examples/pi05/mars/cursor/tools/convert_checkpoints.py"
    spec = importlib.util.spec_from_file_location("convert_checkpoints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
