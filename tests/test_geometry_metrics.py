"""Geometry metric formula and command-line coverage."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import trimesh

from multi_object_decoder.geometry_metrics import (
    compute_geometry_metrics,
    compute_iou,
)


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_aligned_mesh_metrics_distinguish_identical_and_disjoint_boxes() -> None:
    gt_mesh = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    identical_mesh = gt_mesh.copy()
    disjoint_mesh = gt_mesh.copy()
    disjoint_mesh.apply_translation((2.0, 0.0, 0.0))

    metrics = compute_geometry_metrics(
        gt_mesh,
        identical_mesh,
        num_grids=32,
        num_samples=512,
        f_score_threshold=0.01,
        seed=7,
    )

    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["chamfer_distance"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["f_score"] == pytest.approx(1.0)
    assert compute_iou(gt_mesh, disjoint_mesh, num_grids=32) == 0.0


def test_metrics_source_wrapper_writes_json(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.ply"
    pred_path = tmp_path / "pred.ply"
    output_path = tmp_path / "metrics.json"
    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    mesh.export(gt_path)
    mesh.export(pred_path)

    result = subprocess.run(
        [
            sys.executable,
            str(RELEASE_ROOT / "scripts" / "metrics.py"),
            "--gt-mesh",
            str(gt_path),
            "--pred-mesh",
            str(pred_path),
            "--output-json",
            str(output_path),
            "--num-grids",
            "24",
            "--num-samples",
            "256",
        ],
        cwd=RELEASE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["gt_mesh"] == str(gt_path)
    assert payload["pred_mesh"] == str(pred_path)
    assert payload["coordinate_frame"] == "input_meshes_already_aligned"
    assert payload["iou"] == pytest.approx(1.0)
    assert payload["chamfer_distance"] == pytest.approx(0.0, abs=1e-12)
