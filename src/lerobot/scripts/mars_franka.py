"""``mars-franka``: the console script's shim into ``examples/pi05/mars/franka/cli.py``.

``examples/`` is not an installed package, so the console script cannot import ``cli`` as a module.
This puts the example's directory on ``sys.path`` and hands the arguments over; all the command surface
lives in ``cli.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    example = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"
    if str(example) not in sys.path:
        sys.path.insert(0, str(example))
    import cli

    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
