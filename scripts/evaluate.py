#!/usr/bin/env python3
"""Evaluate SAM3D poses per instance and save metrics.txt outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Per-worker validator (set in child by init_worker)
_validate_fn: Any = None
_validator_script: Optional[Path] = None


def init_worker(validator_script: str) -> None:
    """Load validation function in each worker process (required for multiprocessing)."""
    global _validate_fn, _validator_script
    _validator_script = Path(validator_script)
    _validate_fn = load_validate_predicted_poses(_validator_script)


def load_validate_predicted_poses(validator_script: Optional[Path] = None) -> Any:
    """Load validate_predicted_poses from the SAM3D validation script."""
    if validator_script is None:
        release_root = Path(__file__).resolve().parents[1]
        sam3d_root = Path(
            os.environ.get("SAM3D_ROOT", release_root.parent / "sam-3d-objects")
        ).expanduser()
        script_path = sam3d_root / "scripts" / "validate_predicted_poses_quick.py"
    else:
        script_path = validator_script.expanduser()
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Missing validation script: {script_path}")
    spec = importlib.util.spec_from_file_location("validate_predicted_poses_quick", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "validate_predicted_poses"):
        raise AttributeError("validate_predicted_poses not found in validation script")
    return module.validate_predicted_poses


def load_config_entries(config_path: Path) -> List[Dict[str, Any]]:
    """Load list of config entries from JSON."""
    with config_path.open("r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("items", "data", "samples"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError("Config JSON must be a list or contain a list under items/data/samples")


def iter_entries(
    entries: Iterable[Dict[str, Any]],
    include_invalid: bool,
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    """Yield entries with optional validity filtering and limit."""
    count = 0
    for entry in entries:
        if not include_invalid and entry.get("valid") is False:
            continue
        yield entry
        count += 1
        if limit is not None and count >= limit:
            break


def resolve_pose_json(entry: Dict[str, Any], results_key: str = "sam3d_results_path") -> Optional[Path]:
    """Find the predicted pose JSON for a config entry. Also looks in results_key dir (e.g. partcrafter_results_path)."""
    direct_keys = (
        "predicted_poses_json",
        "predicted_poses_path",
        "sam3d_pose_json",
        "pose_json_path",
    )
    for key in direct_keys:
        path_value = entry.get(key)
        if path_value:
            path = Path(path_value)
            if path.exists():
                return path

    candidate_dirs: List[Path] = []
    for key, use_parent in (
        (results_key, False),
        ("sam3d_results_path", False),
        ("midi_results_path", False),
        ("partcrafter_results_path", False),
        ("sam3d_canonical_dir", True),
        ("tokens_path", True),
    ):
        value = entry.get(key)
        if value:
            base = Path(value)
            candidate_dirs.append(base.parent if use_parent else base)

    seen = set()
    deduped_dirs: List[Path] = []
    for directory in candidate_dirs:
        if directory in seen:
            continue
        seen.add(directory)
        deduped_dirs.append(directory)

    candidate_files = (
        "pose_inference.json",
        "pose.json",
        "poses.json",
        "predicted_poses.json",
    )
    for directory in deduped_dirs:
        for filename in candidate_files:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


def resolve_sam3d_paths(entry: Dict[str, Any]) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve SAM3D canonical directory and optional output mesh."""
    value = entry.get("sam3d_canonical_dir")
    if not value:
        return None, None
    path = Path(value)
    output_mesh = None
    canonical_dir = path
    if path.suffix.lower() in {".glb", ".gltf", ".ply", ".obj"}:
        output_mesh = path
        canonical_dir = path.parent / "local_meshes"
    else:
        candidate = path.parent / "complete_multi_object_mesh.glb"
        if candidate.exists():
            output_mesh = candidate
    return canonical_dir, output_mesh


def resolve_refined_metrics(entry: Dict[str, Any]) -> Optional[Path]:
    """Resolve refined metrics.txt path when available."""
    metrics_path = entry.get("metrics_path")
    if metrics_path:
        path = Path(metrics_path)
        if path.exists():
            return path
    gt_poses_path = entry.get("gt_poses_path")
    image_id = entry.get("image_id")
    if gt_poses_path and image_id:
        base = Path(gt_poses_path).parent
        derived = base / f"{image_id}_glb_debug" / "metrics.txt"
        if derived.exists():
            return derived
    return None


def should_use_hungarian_matching(entry: Dict[str, Any]) -> bool:
    """Return True for datasets without GT correspondence (use Hungarian matching, not GT order)."""
    dataset = str(entry.get("dataset", "")).lower()
    scene_type = str(entry.get("type", "")).lower()
    return dataset in {
        "messy_kitchen_26_feb_filtered",
        "realworld",
        "housecat6d",
        "housecat6d_test",
        "graspnet1b",
        "graspnet1b_test",
        "graspclutter6d",
    } or scene_type in {
        "messy_kitchen_26_feb_filtered",
        "realworld",
        "housecat6d",
        "graspnet1b",
        "graspnet1b_test",
        "graspclutter6d",
    }


def write_metrics_txt(output_dir: Path, metrics: Dict[str, Any]) -> Path:
    """Write metrics dict to metrics.txt as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.txt"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path


def _output_dir_for_entry(entry: Dict[str, Any], results_key: str) -> Optional[Path]:
    """Resolve output directory: from results_key, or from sam3d_canonical_dir/sam3d_pose_json parent (HouseCat6D)."""
    output_path = entry.get(results_key)
    if output_path:
        return Path(output_path)
    # HouseCat6D-style: no sam3d_results_path; dir containing local_meshes and pose_inference.json
    canon = entry.get("sam3d_canonical_dir")
    if canon:
        return Path(canon).parent
    pose = entry.get("sam3d_pose_json") or entry.get("pose_json_path")
    if pose:
        return Path(pose).parent
    return None


def evaluate_entry(
    entry: Dict[str, Any],
    validate_predicted_poses: Any,
    num_samples: int,
    skip_existing: bool,
    results_key: str = "sam3d_results_path",
) -> bool:
    """Evaluate one config entry and write metrics.txt. results_key: where to write (e.g. sam3d_results_path). If missing, output_dir derived from sam3d_canonical_dir parent (HouseCat6D)."""
    output_dir = _output_dir_for_entry(entry, results_key)
    if output_dir is None:
        logger.warning("Missing output dir (no %s / sam3d_canonical_dir / sam3d_pose_json) for entry: %s", results_key, entry.get("id"))
        return False
    output_dir = Path(output_dir)
    if skip_existing and (output_dir / "metrics.txt").exists():
        logger.info("Skipping (metrics.txt exists): %s", output_dir)
        return True

    gt_glb_path = entry.get("mesh_path")
    if not gt_glb_path:
        logger.warning("Missing mesh_path (GT GLB) for entry: %s", entry.get("id"))
        return False

    # MIDI and PartCrafter: use completed_meshes.glb in results dir (no pose JSON / canonical)
    use_completed_meshes_glb = results_key in ("midi_results_path", "partcrafter_results_path")
    predicted_mesh_glb = (output_dir / "completed_meshes.glb") if use_completed_meshes_glb else None

    if predicted_mesh_glb is not None and predicted_mesh_glb.exists():
        result = validate_predicted_poses(
            gt_glb_path=Path(gt_glb_path),
            sam3d_canonical_dir=None,
            predicted_poses_json=None,
            output_dir=output_dir,
            num_samples=num_samples,
            refined_metrics_txt=resolve_refined_metrics(entry),
            scene_id=None,
            log_metrics_to_wandb=False,
            use_hungarian_matching=should_use_hungarian_matching(entry),
            predicted_mesh_glb=predicted_mesh_glb,
        )
    else:
        # SAM3D etc.: use pose JSON + canonical meshes
        if use_completed_meshes_glb:
            logger.warning("Missing completed_meshes.glb for entry: %s (%s)", entry.get("id"), output_dir)
            return False
        predicted_poses_json = resolve_pose_json(entry, results_key=results_key)
        if predicted_poses_json is None:
            logger.warning("Missing pose JSON for entry: %s", entry.get("id"))
            return False
        sam3d_canonical_dir, sam3d_output_mesh = resolve_sam3d_paths(entry)
        if sam3d_output_mesh is not None:
            logger.info("Detected SAM3D output mesh: %s", sam3d_output_mesh)
        if not sam3d_canonical_dir:
            logger.warning("Missing sam3d_canonical_dir for entry: %s", entry.get("id"))
            return False
        result = validate_predicted_poses(
            gt_glb_path=Path(gt_glb_path),
            sam3d_canonical_dir=Path(sam3d_canonical_dir),
            predicted_poses_json=predicted_poses_json,
            output_dir=output_dir,
            num_samples=num_samples,
            refined_metrics_txt=resolve_refined_metrics(entry),
            scene_id=None,
            log_metrics_to_wandb=False,
            use_hungarian_matching=should_use_hungarian_matching(entry),
        )
    if result.get("status") != "success":
        logger.warning("Evaluation failed for entry: %s", entry.get("id"))
        return False

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        logger.warning("No metrics in result for entry: %s", entry.get("id"))
        return False

    metrics_txt = write_metrics_txt(output_dir, metrics)
    logger.info("Saved metrics.txt: %s", metrics_txt)
    return True


def _worker_evaluate(args: tuple) -> bool:
    """Worker: evaluate one entry using process-local _validate_fn."""
    entry, num_samples, skip_existing, results_key = args
    assert _validate_fn is not None, "init_worker() must be used with Pool"
    return evaluate_entry(entry, _validate_fn, num_samples, skip_existing, results_key=results_key)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SAM3D poses per instance.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to SAM3D config JSON.",
    )
    parser.add_argument(
        "--validator-script",
        type=Path,
        default=None,
        help=(
            "Path to validate_predicted_poses_quick.py. Defaults to "
            "$SAM3D_ROOT/scripts/validate_predicted_poses_quick.py."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default 4). Use 4-8 for faster evaluation.",
    )
    parser.add_argument(
        "--results-key",
        type=str,
        default="sam3d_results_path",
        help="Key in each record for output directory (default: sam3d_results_path). Use partcrafter_results_path for PartCrafter.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    entries = load_config_entries(args.config)
    work_list = list(iter_entries(entries, args.include_invalid, args.limit))
    total = len(work_list)
    if total == 0:
        logger.error("No valid evaluation entries were selected from %s", args.config)
        return 2
    work_tuples = [(entry, args.num_samples, args.skip_existing, args.results_key) for entry in work_list]

    validator_script = args.validator_script
    if validator_script is None:
        release_root = Path(__file__).resolve().parents[1]
        sam3d_root = Path(
            os.environ.get("SAM3D_ROOT", release_root.parent / "sam-3d-objects")
        ).expanduser()
        validator_script = sam3d_root / "scripts" / "validate_predicted_poses_quick.py"

    if args.workers <= 1:
        validate_predicted_poses = load_validate_predicted_poses(validator_script)
        success = sum(
            1
            for entry in work_list
            if evaluate_entry(entry, validate_predicted_poses, args.num_samples, args.skip_existing, results_key=args.results_key)
        )
    else:
        with Pool(
            args.workers,
            initializer=init_worker,
            initargs=(str(validator_script),),
        ) as pool:
            success = sum(pool.imap_unordered(_worker_evaluate, work_tuples, chunksize=1))

    logger.info("Finished: %d/%d entries evaluated", success, total)
    if success != total:
        logger.error("Evaluation failed for %d/%d entries", total - success, total)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
