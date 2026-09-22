#!/usr/bin/env python3
"""Run MOD refinement from cached SAM3D artifacts and a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from multi_object_decoder.checkpoints import load_checkpoint
from multi_object_decoder.cli import (
    load_config,
    scene_directory_name,
    select_device,
    validated_scene_directory_names,
)
from multi_object_decoder.data import SceneDataset
from multi_object_decoder.export import export_posed_meshes
from multi_object_decoder.poses import pose_loss, save_pose_json
from multi_object_decoder.workflow import build_system


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a MOD checkpoint, refine all objects in each cached scene together, "
            "and write mod_pose.json plus an optional posed GLB."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-json", "--config", dest="data_json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backend",
        choices=("direct", "sam3d"),
        help="Required only for a legacy model.pth checkpoint.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="YAML containing a model section; required only for a legacy checkpoint.",
    )
    parser.add_argument("--sam3d-root", type=Path, default=os.environ.get("SAM3D_ROOT"))
    parser.add_argument(
        "--sam3d-config",
        type=Path,
        default=os.environ.get("SAM3D_PIPELINE_CONFIG"),
    )
    parser.add_argument("--export-meshes", action="store_true")
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help=(
            "Trust legacy tokens/obj_*.npz caches containing pickle-backed object "
            "arrays. Portable scene NPZ caches do not need this flag."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _checkpoint_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "model.pth" if path.is_dir() else path


def _legacy_model_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("--model-config is required for a legacy model.pth checkpoint")
    config = load_config(path)
    model_config = config.get("model")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"{path}: expected a non-empty model section")
    return dict(model_config)


def _decoder(args: argparse.Namespace, backend: str):
    if backend != "sam3d":
        return None
    if args.sam3d_root is None or args.sam3d_config is None:
        raise ValueError(
            "backend=sam3d requires --sam3d-root and --sam3d-config "
            "(or SAM3D_ROOT and SAM3D_PIPELINE_CONFIG)"
        )
    from multi_object_decoder.sam3d_backend import SAM3DBackend

    return SAM3DBackend.from_paths(args.sam3d_root, args.sam3d_config)


def main() -> int:
    args = _parse_args()
    dataset = SceneDataset(
        args.data_json.expanduser().resolve(),
        require_target=False,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    validated_scene_directory_names(
        [str(record["scene_id"]) for record in dataset.records]
    )
    checkpoint_path = _checkpoint_file(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["version"] == 0:
        if args.backend is None:
            raise ValueError("--backend is required for a legacy model.pth checkpoint")
        backend = args.backend
        model_config = _legacy_model_config(args.model_config)
    else:
        backend = checkpoint["backend"]
        model_config = dict(checkpoint["model_config"])
        if args.backend is not None and args.backend != backend:
            raise ValueError(
                f"--backend={args.backend} does not match checkpoint backend={backend}"
            )

    device = select_device(args.device)
    system = build_system(backend, model_config, decoder=_decoder(args, backend))
    system.model.load_state_dict(checkpoint["state_dict"], strict=True)
    system.model.to(device).eval()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    had_error = False

    for sample in dataset:
        scene_dir = output_root / scene_directory_name(sample.scene_id)
        pose_path = scene_dir / "mod_pose.json"
        if pose_path.exists() and not args.overwrite:
            summaries.append(
                {
                    "scene_id": sample.scene_id,
                    "status": "skipped",
                    "pose_path": str(pose_path),
                }
            )
            continue
        try:
            with torch.inference_mode():
                predicted = system.predict(sample, device).float()
            if predicted.shape != (sample.num_objects, 10):
                raise RuntimeError(
                    f"Model returned {tuple(predicted.shape)} for "
                    f"{sample.num_objects} objects"
                )
            save_pose_json(
                pose_path,
                sample.object_names,
                predicted,
                metadata={
                    "backend": backend,
                    "checkpoint": checkpoint_path.name,
                    "scene_id": sample.scene_id,
                },
            )
            outputs = [str(pose_path)]
            if args.export_meshes:
                outputs.extend(
                    str(path)
                    for path in export_posed_meshes(
                        sample.record,
                        sample.object_names,
                        predicted,
                        scene_dir / "posed_meshes",
                    )
                )
            scene_summary: dict[str, Any] = {
                "scene_id": sample.scene_id,
                "status": "success",
                "num_objects": sample.num_objects,
                "outputs": outputs,
            }
            if sample.target_poses is not None:
                _, components = pose_loss(
                    predicted.to(device),
                    sample.target_poses.to(device),
                )
                scene_summary["metrics"] = {
                    key: float(value.cpu()) for key, value in components.items()
                }
            summaries.append(scene_summary)
        except Exception as error:
            had_error = True
            summaries.append(
                {
                    "scene_id": sample.scene_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if not args.continue_on_error:
                break

    report = {
        "backend": backend,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "scenes": summaries,
        "successful": sum(item["status"] == "success" for item in summaries),
        "failed": sum(item["status"] == "failed" for item in summaries),
        "skipped": sum(item["status"] == "skipped" for item in summaries),
    }
    report_path = output_root / "inference_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if had_error else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
