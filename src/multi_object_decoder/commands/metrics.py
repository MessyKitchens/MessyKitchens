"""Compute local IoU/CD/F-score for two already-aligned meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_object_decoder.geometry_metrics import compute_geometry_metrics, load_mesh


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-mesh", type=Path, required=True)
    parser.add_argument("--pred-mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--num-grids", type=int, default=64)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--f-score-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    gt_path = args.gt_mesh.expanduser().resolve()
    pred_path = args.pred_mesh.expanduser().resolve()
    result = {
        "gt_mesh": str(gt_path),
        "pred_mesh": str(pred_path),
        **compute_geometry_metrics(
            load_mesh(gt_path),
            load_mesh(pred_path),
            num_grids=args.num_grids,
            scale=args.scale,
            num_samples=args.num_samples,
            f_score_threshold=args.f_score_threshold,
            seed=args.seed,
            iou_only=args.iou_only,
        ),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
