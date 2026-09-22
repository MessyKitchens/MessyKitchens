"""Stable object-ID parsing shared by data loading and validation output."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union


def parse_prefixed_object_id(value: Any, prefix: str = "obj") -> Optional[int]:
    """Parse numeric IDs from names such as ``obj_2`` or ``obj_0002.npz``."""
    stem = Path(str(value)).stem
    match = re.fullmatch(rf"{re.escape(prefix)}_0*(\d+)", stem)
    if match is None:
        return None
    return int(match.group(1))


def object_id_sort_key(
    value: Any,
    prefix: str = "obj",
) -> tuple[int, Union[int, str]]:
    """Sort recognized IDs numerically, followed by unrecognized names."""
    object_id = parse_prefixed_object_id(value, prefix=prefix)
    return (0, object_id) if object_id is not None else (1, str(value))


__all__ = ["object_id_sort_key", "parse_prefixed_object_id"]
