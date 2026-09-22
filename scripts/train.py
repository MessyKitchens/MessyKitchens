#!/usr/bin/env python3
"""Source-tree wrapper for the installed mod-train command."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multi_object_decoder.commands.train import main  # noqa: E402


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
