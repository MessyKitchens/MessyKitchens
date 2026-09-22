#!/usr/bin/env python3
"""Validate and materialize the bundled synthetic cache for an offline CPU smoke.

This copies precomputed numerical fixtures. It does not run image-to-SAM3D
processing; use the default real demo for that stage.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multi_object_decoder.data import SceneDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.expanduser().resolve()
    if output == source or output in source.parents or output.is_relative_to(ROOT / "assets/demo"):
        parser.error("output must be outside the bundled inputs")
    samples = list(SceneDataset(source / "prepared_data.json", require_target=True))
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error("output is not empty; choose a new directory or use --overwrite")
    shutil.copytree(source, output, dirs_exist_ok=True)
    reloaded = list(SceneDataset(output / "prepared_data.json", require_target=True))
    if len(samples) != len(reloaded):
        raise RuntimeError("materialized sample count changed")
    summary = {"kind": "synthetic_direct_cache_fixture", "scenes": len(reloaded),
               "object_ids": [list(sample.object_ids) for sample in reloaded]}
    (output / "demo_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
