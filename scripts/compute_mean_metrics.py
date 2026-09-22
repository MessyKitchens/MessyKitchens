#!/usr/bin/env python3
"""
Compute mean metrics (CD, IoU) from a config JSON, write results to Excel, and
write a filtered config JSON containing only entries that have computed metrics.

Usage:
  python scripts/compute_mean_metrics.py --config CONFIG_JSON

Requires: openpyxl (pip install openpyxl)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def safe_mean(xs):
    return None if not xs else statistics.mean(xs)


def compute_mean_metrics_from_config(config_path: Path):
    """
    Load config JSON (list of items with optional ours_results_path), scan each
    result dir for metrics_*.json, aggregate scene-level and object-level CD/IoU.

    Returns:
        dict with keys:
            num_items, num_with_results, num_missing_dir, num_missing_metrics, num_parse_error,
            scene_cd_all, scene_iou_all, obj_cd_all, obj_iou_all (lists),
            scene_mean_cd, scene_mean_iou, obj_mean_cd, obj_mean_iou (floats or None),
            rows: list of dicts for per-file rows,
            filtered_items: list of original config items that have at least one valid metrics_*.json (for writing *_filtered.json).
    """
    with config_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON list in {config_path}, got {type(items)}")

    scene_cd_all = []
    scene_iou_all = []
    obj_cd_all = []
    obj_iou_all = []
    rows = []
    indices_with_metrics = set()

    num_items = 0
    num_with_results = 0
    num_missing_dir = 0
    num_missing_metrics = 0
    num_parse_error = 0

    for idx, it in enumerate(items):
        num_items += 1
        ours_path = it.get("ours_results_path")
        if not isinstance(ours_path, str) or not ours_path:
            continue

        res_dir = Path(ours_path)
        if not res_dir.exists():
            num_missing_dir += 1
            continue

        metrics_files = sorted(res_dir.glob("metrics_*.json"))
        if not metrics_files:
            num_missing_metrics += 1
            continue

        scene_id = it.get("scene_id") or it.get("mesh_path", "") or str(idx)

        for mf in metrics_files:
            try:
                with mf.open("r", encoding="utf-8") as f:
                    m = json.load(f)
            except Exception:
                num_parse_error += 1
                continue

            scene_cd = scene_iou = obj_cd = obj_iou = None

            try:
                sc = m["scene_level"]["predicted"]
                scene_cd = float(sc["mean_cd"])
                scene_iou = float(sc["mean_iou"])
                scene_cd_all.append(scene_cd)
                scene_iou_all.append(scene_iou)
            except Exception:
                pass

            try:
                obj = m["object_level_predicted"]["summary"]
                obj_cd = float(obj["mean_cd"])
                obj_iou = float(obj["mean_iou"])
                obj_cd_all.append(obj_cd)
                obj_iou_all.append(obj_iou)
            except Exception:
                pass

            rows.append({
                "config_index": idx,
                "scene_id": str(scene_id)[:80],
                "metrics_path": str(mf),
                "scene_cd": scene_cd,
                "scene_iou": scene_iou,
                "obj_cd": obj_cd,
                "obj_iou": obj_iou,
            })
            indices_with_metrics.add(idx)
            num_with_results += 1

    filtered_items = [items[i] for i in sorted(indices_with_metrics)]

    return {
        "num_items": num_items,
        "num_with_results": num_with_results,
        "num_missing_dir": num_missing_dir,
        "num_missing_metrics": num_missing_metrics,
        "num_parse_error": num_parse_error,
        "scene_cd_all": scene_cd_all,
        "scene_iou_all": scene_iou_all,
        "obj_cd_all": obj_cd_all,
        "obj_iou_all": obj_iou_all,
        "scene_mean_cd": safe_mean(scene_cd_all),
        "scene_mean_iou": safe_mean(scene_iou_all),
        "obj_mean_cd": safe_mean(obj_cd_all),
        "obj_mean_iou": safe_mean(obj_iou_all),
        "rows": rows,
        "filtered_items": filtered_items,
    }


def write_metrics_to_excel(stats: dict, output_path: Path) -> None:
    """Write aggregate stats and per-file rows to an Excel file."""
    try:
        import openpyxl
        from openpyxl.styles import Font
    except ImportError:
        raise ImportError("Writing Excel requires openpyxl. Install with: pip install openpyxl") from None

    wb = openpyxl.Workbook()

    # Sheet 1: Summary
    ws_sum = wb.active
    ws_sum.title = "Summary"
    headers_sum = [
        "Metric", "Value",
    ]
    for c, h in enumerate(headers_sum, 1):
        ws_sum.cell(row=1, column=c, value=h)
    ws_sum.row_dimensions[1].font = Font(bold=True)

    summary_rows = [
        ("config items", stats["num_items"]),
        ("dirs with results", stats["num_with_results"]),
        ("missing result dirs", stats["num_missing_dir"]),
        ("missing metrics_*.json", stats["num_missing_metrics"]),
        ("metrics parse errors", stats["num_parse_error"]),
        ("scene-level samples", len(stats["scene_cd_all"])),
        ("object-level samples", len(stats["obj_cd_all"])),
        ("", ""),
        ("Scene-level mean CD", stats["scene_mean_cd"] if stats["scene_mean_cd"] is not None else "(no data)"),
        ("Scene-level mean IoU", stats["scene_mean_iou"] if stats["scene_mean_iou"] is not None else "(no data)"),
        ("Object-level mean CD", stats["obj_mean_cd"] if stats["obj_mean_cd"] is not None else "(no data)"),
        ("Object-level mean IoU", stats["obj_mean_iou"] if stats["obj_mean_iou"] is not None else "(no data)"),
    ]
    for r, (label, val) in enumerate(summary_rows, 2):
        ws_sum.cell(row=r, column=1, value=label)
        if isinstance(val, (int, float)):
            ws_sum.cell(row=r, column=2, value=val)
        else:
            ws_sum.cell(row=r, column=2, value=val)

    # Sheet 2: Per-file
    ws_per = wb.create_sheet("Per_file", 1)
    row_headers = ["config_index", "scene_id", "metrics_path", "scene_cd", "scene_iou", "obj_cd", "obj_iou"]
    for c, h in enumerate(row_headers, 1):
        ws_per.cell(row=1, column=c, value=h)
    ws_per.row_dimensions[1].font = Font(bold=True)

    for r, row in enumerate(stats["rows"], 2):
        for c, key in enumerate(row_headers, 1):
            val = row.get(key)
            if val is not None and isinstance(val, float):
                ws_per.cell(row=r, column=c, value=round(val, 6))
            else:
                ws_per.cell(row=r, column=c, value=val)

    wb.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, default=None)
    parser.add_argument("--output-filtered-json", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()

    if not config_path.exists():
        print(f"Error: config json not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if args.output_xlsx is None:
        output_xlsx = Path(__file__).resolve().parent / f"mean_metrics_{config_path.stem}.xlsx"
    else:
        output_xlsx = args.output_xlsx

    if args.output_filtered_json is None:
        output_filtered_json = config_path.parent / f"{config_path.stem}_filtered.json"
    else:
        output_filtered_json = args.output_filtered_json

    output_xlsx = output_xlsx.resolve()
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_filtered_json = output_filtered_json.resolve()
    output_filtered_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using config: {config_path}")
    stats = compute_mean_metrics_from_config(config_path)

    print("")
    print("=== Aggregate stats from metrics_*.json ===")
    print(f"config items:           {stats['num_items']}")
    print(f"dirs with results:      {stats['num_with_results']}")
    print(f"missing result dirs:    {stats['num_missing_dir']}")
    print(f"missing metrics_*.json: {stats['num_missing_metrics']}")
    print(f"metrics parse errors:   {stats['num_parse_error']}")
    print("")
    print(f"scene-level samples:    {len(stats['scene_cd_all'])}")
    print(f"object-level samples:   {len(stats['obj_cd_all'])}")
    print("")
    if stats["scene_cd_all"]:
        print(f"Scene-level  mean CD : {stats['scene_mean_cd']:.6f}")
        print(f"Scene-level  mean IoU: {stats['scene_mean_iou']:.6f}")
    else:
        print("Scene-level  mean CD / IoU: (no data)")
    if stats["obj_cd_all"]:
        print(f"Object-level mean CD : {stats['obj_mean_cd']:.6f}")
        print(f"Object-level mean IoU: {stats['obj_mean_iou']:.6f}")
    else:
        print("Object-level mean CD / IoU: (no data)")

    write_metrics_to_excel(stats, output_xlsx)
    with output_filtered_json.open("w", encoding="utf-8") as f:
        json.dump(stats["filtered_items"], f, indent=2, ensure_ascii=False)
    print("")
    print(f"Excel written: {output_xlsx}")
    print(f"Filtered config ({len(stats['filtered_items'])} items with metrics): {output_filtered_json}")


if __name__ == "__main__":
    main()
