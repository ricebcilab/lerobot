"""Make the MARS cursor example's modules importable (they are scripts, not a package), and keep torch's threads few."""

import os
import sys
from pathlib import Path

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
    """The tests' tensors are tiny: many threads only contend (on a loaded machine, by two orders of magnitude)."""
    torch.set_num_threads(int(os.environ.get("MARS_TEST_THREADS", min(8, os.cpu_count() or 8))))
