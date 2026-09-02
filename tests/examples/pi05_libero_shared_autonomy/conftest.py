"""Make the example's modules importable (they are scripts, not a package)."""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "libero_shared_autonomy"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
