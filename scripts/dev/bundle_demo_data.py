#!/usr/bin/env python3
"""Bundle one complete, trusted SAM3D training sample without changing its tensors.

Source locations are supplied through an existing manifest and are never embedded
in the resulting public bundle. Legacy object arrays are accepted only with the
explicit --trust-legacy-cache conversion flag. Runtime demos use numeric NPZs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from multi_object_decoder.data import SceneDataset  # noqa: E402

GSO_SOURCE = "https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/"
SAM3D_SOURCE = "https://github.com/facebookresearch/sam-3d-objects"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(mapping: dict) -> dict[int, dict]:
    result = {}
    for name, pose in mapping.items():
        match = re.fullmatch(r"obj_0*(\d+)", name)
        if match is None or int(match.group(1)) in result:
            raise ValueError(f"Invalid or duplicate object name: {name}")
        result[int(match.group(1))] = pose
    return result


def bundle(manifest: Path, record_index: int, output: Path, trust: bool) -> dict:
    if not trust:
        raise ValueError("Converting legacy cache object arrays requires --trust-legacy-cache")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    record = records[record_index]
    if not str(record.get("dataset", "")).startswith("GSO"):
        raise ValueError(
            "This curated bundler currently supports attributed GSO-derived scenes only"
        )

    def source(field: str) -> Path:
        path = Path(record[field]).expanduser()
        return path if path.is_absolute() else manifest.parent / path

    image_source = source("image_path")
    mask_source = source("instance_mask_path")
    token_source = source("tokens_path")
    canonical_source = source("sam3d_canonical_dir")
    target_source = source("gt_poses_path")
    base_source = token_source.parent / "pose_inference.json"
    targets = json.loads(target_source.read_text())
    bases = json.loads(base_source.read_text())
    target_ids = _ids(targets["objects"])
    base_ids = _ids(bases["objects"])
    mask = np.load(mask_source, allow_pickle=False)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer-label mask")
    with Image.open(image_source) as image:
        image_size = image.size
    if mask.shape != image_size[::-1]:
        raise ValueError("Image and mask dimensions must match; the bundle never resizes them")
    labels = sorted(int(value) for value in np.unique(mask) if int(value) != 0)
    ids = [label - 1 for label in labels]
    if not labels or min(labels) < 1 or max(labels) > 255:
        raise ValueError("The curated PNG mask requires labels in 1..255 and background 0")
    token_files = {}
    for path in token_source.glob("obj_*.npz"):
        match = re.fullmatch(r"obj_0*(\d+)\.npz", path.name)
        if match is None or int(match.group(1)) in token_files:
            raise ValueError(f"Invalid or duplicate token file: {path.name}")
        token_files[int(match.group(1))] = path
    if not (set(ids) == set(token_files) == set(target_ids) == set(base_ids)):
        raise ValueError(
            "Mask labels, all cached objects, base poses and GT poses must match exactly"
        )
    if targets.get("metadata", {}).get("num_matched", len(ids)) != len(ids):
        raise ValueError("Original target metadata does not report a fully matched scene")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be absent or empty")
    raw = output / "raw"
    prepared = output / "prepared"
    meshes = prepared / "local_meshes"
    raw.mkdir(parents=True, exist_ok=True)
    meshes.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_source, raw / "image.png")
    Image.fromarray(mask.astype(np.uint8)).save(raw / "instance_mask.png", optimize=True)
    with Image.open(raw / "instance_mask.png") as image:
        if not np.array_equal(np.asarray(image), mask):
            raise AssertionError("Instance-label PNG conversion was not lossless")
    safe_metadata = {
        key: value
        for key, value in targets.get("metadata", {}).items()
        if key
        in {"num_gt_objects", "num_sam3d_objects", "num_matched", "scene_transform", "scene_scale"}
    }
    _write_json(
        prepared / "target_poses.json", {"objects": targets["objects"], "metadata": safe_metadata}
    )
    _write_json(prepared / "base_poses.json", {"objects": bases["objects"]})
    sample_id = image_source.stem
    provenance = {
        "sample_id": sample_id,
        "dataset": "GSO synthetic training render",
        "real_camera_capture": False,
        "source_dataset": GSO_SOURCE,
        "source_dataset_license": "CC-BY-4.0",
        "sam3d_source": SAM3D_SOURCE,
        "sam3d_software_and_weight_terms": "SAM License; see THIRD_PARTY_NOTICES.md",
        "source_record_index": record_index,
        "source_manifest_sha256": _sha256(manifest),
        "source_image_sha256": _sha256(image_source),
        "source_mask_npy_sha256": _sha256(mask_source),
        "source_target_json_sha256": _sha256(target_source),
        "source_base_json_sha256": _sha256(base_source),
        "image_resolution": list(image_size),
        "objects": [],
        "transformations": [
            "Copied original RGB bytes without resizing or recompression.",
            "Converted the original integer instance labels to lossless uint8 PNG.",
            "Unboxed trusted cached mappings once; saved every original numeric array and dtype in per-object compressed NPZs.",
            "Retained original GT/base object pose values; removed private filesystem metadata.",
            "Exported all canonical vertices and faces as binary PLY; omitted visual attributes without geometry sampling.",
        ],
    }
    shard_names = []
    for object_id in ids:
        path = token_files[object_id]
        with np.load(path, allow_pickle=True) as archive:
            legacy = {
                key: (archive[key].item() if archive[key].dtype.hasobject else archive[key])
                for key in archive
            }
        arrays = {"object_ids": np.asarray([object_id], dtype=np.int64)}
        for prefix in ("pose_latents_raw", "x_t_last_step"):
            for key, value in legacy[prefix].items():
                arrays[f"{prefix}__{key}"] = np.asarray(value)
        for key in ("pointmap_scale", "pointmap_shift"):
            arrays[key] = np.asarray(legacy[key])
        for key, value in legacy["solver_state"].items():
            arrays[f"solver_state__{key}"] = np.asarray([value])
        condition_source = token_source / f"img_cond_{object_id:03d}.npy"
        arrays["image_condition_per_object"] = np.load(condition_source, allow_pickle=False)
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("Portable cache may not contain object arrays")
        shard_name = f"scene_obj_{object_id:03d}.npz"
        shard = prepared / shard_name
        np.savez_compressed(shard, **arrays)
        with np.load(shard, allow_pickle=False) as converted:
            for key, expected in arrays.items():
                actual = converted[key]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise AssertionError(f"Cache conversion changed values or dtype for {key}")
        if shard.stat().st_size > 50 * 1024**2:
            raise ValueError(f"Object shard exceeds 50 MiB: {shard_name}")
        shard_names.append(shard_name)
        mesh_source = canonical_source / f"obj_{object_id}_local.ply"
        original = trimesh.load(mesh_source, force="mesh", process=False)
        geometry = trimesh.Trimesh(
            vertices=original.vertices.copy(), faces=original.faces.copy(), process=False
        )
        mesh_output = meshes / mesh_source.name
        mesh_output.write_bytes(
            trimesh.exchange.ply.export_ply(
                geometry, encoding="binary", vertex_normal=False, include_attributes=False
            )
        )
        restored = trimesh.load(mesh_output, force="mesh", process=False)
        if not np.array_equal(original.vertices, restored.vertices) or not np.array_equal(
            original.faces, restored.faces
        ):
            raise AssertionError("Canonical mesh conversion changed vertices or faces")
        provenance["objects"].append(
            {
                "mask_label": object_id + 1,
                "object_id": object_id,
                "gt_pose_key": next(
                    key for key in targets["objects"] if int(key.split("_")[-1]) == object_id
                ),
                "canonical_file": f"prepared/local_meshes/{mesh_source.name}",
                "cache_shard": f"prepared/{shard_name}",
                "source_token_sha256": _sha256(path),
                "source_image_condition_sha256": _sha256(condition_source),
                "source_canonical_ply_sha256": _sha256(mesh_source),
                "canonical_vertices": len(original.vertices),
                "canonical_faces": len(original.faces),
                "pose_tokens": list(arrays["pose_latents_raw__translation"].shape),
                "shape_tokens": list(arrays["pose_latents_raw__shape"].shape),
                "image_condition": list(arrays["image_condition_per_object"].shape),
            }
        )
    scene_id = f"gso/{sample_id}"
    _write_json(
        raw / "input.json",
        [
            {
                "id": sample_id,
                "scene_id": scene_id,
                "image_id": sample_id,
                "dataset": "GSO_synthetic_demo",
                "num_parts": len(ids),
                "image_path": "image.png",
                "instance_mask_path": "instance_mask.png",
            }
        ],
    )
    _write_json(
        prepared / "prepared_data.json",
        [
            {
                "scene_id": scene_id,
                "dataset": "GSO_synthetic_demo",
                "num_parts": len(ids),
                "scene_npz_paths": shard_names,
                "base_poses_path": "base_poses.json",
                "target_poses_path": "target_poses.json",
                "sam3d_canonical_dir": "local_meshes",
                "image_path": "../raw/image.png",
                "instance_mask_path": "../raw/instance_mask.png",
            }
        ],
    )
    _write_json(output / "PROVENANCE.json", provenance)
    sample = SceneDataset(prepared / "prepared_data.json", require_target=True)[0]
    if tuple(sample.object_ids) != tuple(ids) or sample.target_poses is None:
        raise AssertionError("Bundled complete scene failed loader validation")
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if total_bytes > 170 * 1024**2:
        raise ValueError(f"Bundle exceeds the 170 MiB dataset budget: {total_bytes} bytes")
    return {
        "sample_id": sample_id,
        "object_ids": ids,
        "bytes": total_bytes,
        "pose_tokens": list(sample.pose_tokens.shape),
        "shape_tokens": list(sample.shape_tokens.shape),
        "cache_values_and_dtypes_preserved": True,
        "canonical_geometry_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--record-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-legacy-cache", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            bundle(
                args.source_manifest, args.record_index, args.output_dir, args.trust_legacy_cache
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
