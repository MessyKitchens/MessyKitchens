#!/usr/bin/env python3
"""Generate a tiny, deterministic portable MOD quick-start dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multi_object_decoder.data import SceneDataset  # noqa: E402


POSE_KEYS = (
    "6drotation_normalized",
    "translation",
    "scale",
    "translation_scale",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _scene_archive(
    rng: np.random.Generator,
    object_ids: np.ndarray,
    scene_index: int,
) -> dict[str, np.ndarray]:
    object_count = int(object_ids.size)
    feature_dim = 8
    shape_token_count = 4
    archive: dict[str, np.ndarray] = {"object_ids": object_ids.astype(np.int64)}

    for key in POSE_KEYS:
        archive[f"pose_latents_raw__{key}"] = rng.normal(
            size=(object_count, 1, feature_dim)
        ).astype(np.float32)
    archive["pose_latents_raw__shape"] = rng.normal(
        size=(object_count, shape_token_count, feature_dim)
    ).astype(np.float32)

    x_t_dimensions = {
        "6drotation_normalized": (1, 6),
        "translation": (1, 3),
        "scale": (1, 3),
        "translation_scale": (1, 1),
        "shape": (shape_token_count, 8),
    }
    for key, (token_count, output_dim) in x_t_dimensions.items():
        archive[f"x_t_last_step__{key}"] = rng.normal(
            size=(object_count, token_count, output_dim)
        ).astype(np.float32)

    archive["pointmap_scale"] = np.ones((object_count, 3), dtype=np.float32)
    archive["pointmap_shift"] = np.zeros((object_count, 3), dtype=np.float32)
    archive["solver_state__t"] = np.full(object_count, 0.75, dtype=np.float32)
    archive["solver_state__dt"] = np.full(object_count, 0.05, dtype=np.float32)
    archive["image_condition_per_object"] = rng.normal(
        size=(object_count, 1, feature_dim)
    ).astype(np.float32)

    base_poses = np.zeros((object_count, 10), dtype=np.float32)
    base_poses[:, 0] = object_ids.astype(np.float32) * 0.4
    base_poses[:, 1] = float(scene_index) * 0.2
    base_poses[:, 3] = 1.0  # Unit quaternion in [w, x, y, z] order.
    base_poses[:, 7:] = 1.0
    target_poses = base_poses.copy()
    target_poses[:, :3] += np.asarray(
        [0.15 + 0.02 * scene_index, -0.08, 0.04], dtype=np.float32
    )
    target_poses[:, 7:] *= 1.0 + 0.01 * (scene_index + 1)
    archive["base_poses"] = base_poses
    archive["target_poses"] = target_poses
    return archive


def make_demo_data(output_dir: Path, seed: int = 17) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scene_specs = (
        ("demo/scene_000", np.asarray([2, 0], dtype=np.int64)),
        ("demo/scene_001", np.asarray([4, 1, 0], dtype=np.int64)),
    )
    records = []
    summaries = []
    local_meshes = output_dir / "local_meshes"
    local_meshes.mkdir(parents=True, exist_ok=True)
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            "Demo mesh generation requires trimesh; run bash scripts/install.sh"
        ) from error
    all_object_ids = sorted(
        {int(object_id) for _, object_ids in scene_specs for object_id in object_ids}
    )
    for object_id in all_object_ids:
        mesh = trimesh.creation.box(
            extents=(0.18 + object_id * 0.01, 0.14, 0.12)
        )
        mesh.export(local_meshes / f"obj_{object_id}_local.ply")

    for scene_index, (scene_id, object_ids) in enumerate(scene_specs):
        scene_path = output_dir / f"scene_{scene_index:03d}.npz"
        archive = _scene_archive(rng, object_ids, scene_index)
        np.savez(scene_path, **archive)
        records.append(
            {
                "scene_id": scene_id,
                "dataset": "generated_demo",
                "scene_npz_path": scene_path.name,
                "sam3d_canonical_dir": "local_meshes",
            }
        )
        summaries.append(
            {
                "scene_id": scene_id,
                "object_ids_on_disk": object_ids.tolist(),
                "cache": scene_path.name,
            }
        )

    manifest_path = output_dir / "prepared_data.json"
    manifest_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    dataset = SceneDataset(manifest_path, require_target=True)
    validated = [
        {
            "scene_id": sample.scene_id,
            "object_ids_loaded": list(sample.object_ids),
            "pose_tokens": list(sample.pose_tokens.shape),
            "shape_tokens": list(sample.shape_tokens.shape),
        }
        for sample in dataset
    ]
    summary = {
        "seed": seed,
        "manifest": str(manifest_path),
        "canonical_meshes": [
            f"local_meshes/obj_{object_id}_local.ply" for object_id in all_object_ids
        ],
        "scenes": summaries,
        "validated": validated,
    }
    (output_dir / "demo_data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = _parse_args()
    summary = make_demo_data(args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
