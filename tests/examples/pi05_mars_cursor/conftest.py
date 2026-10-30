"""Make the MARS cursor example's modules importable (they are scripts, not a package)."""

import sys
from pathlib import Path

MARS_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars"
for path in (MARS_DIR / "cursor", MARS_DIR):  # cursor_sa and the scripts; `figures` lives under MARS_DIR
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
