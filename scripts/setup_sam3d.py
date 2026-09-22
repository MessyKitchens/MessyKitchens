#!/usr/bin/env python3
"""Verify the pinned external SAM3D source needed by MOD's runtime integration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REVISION = "afdf6a31522d038c44c68a0bb57aa68827380797"
SOURCE_FILES = (
    "sam3d_objects/model/backbone/generator/flow_matching/solver.py",
    "sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py",
    "sam3d_objects/pipeline/inference_pipeline.py",
    "sam3d_objects/pipeline/inference_pipeline_pointmap.py",
    "sam3d_objects/data/dataset/tdfy/pose_target.py",
    "sam3d_objects/data/dataset/tdfy/transforms_3d.py",
)


def inspect_source(root: Path) -> dict:
    root = root.expanduser().resolve()
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
    revision = git("rev-parse", "HEAD").decode().strip()
    if revision != REVISION:
        raise ValueError("SAM3D HEAD must equal " + REVISION + "; found " + revision)
    hashes = {}
    for name in SOURCE_FILES:
        current = (root / name).read_bytes()
        pinned = git("show", REVISION + ":" + name)
        if current != pinned:
            raise ValueError("SAM3D integration source differs from pinned upstream: " + name)
        hashes[name] = hashlib.sha256(current).hexdigest()
    return {
        "upstream": "https://github.com/facebookresearch/sam-3d-objects",
        "revision": revision,
        "integration": "scoped_runtime_cache_observer",
        "source_files_sha256": hashes,
        "gpu_execution": "not_tested_by_this_source_check",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3d-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect_source(args.sam3d_root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, "SAM3D source check failed: " + str(error) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
