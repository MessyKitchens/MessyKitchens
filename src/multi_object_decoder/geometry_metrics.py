"""Self-contained geometry metrics for meshes already in one coordinate frame.

Parts are adapted from PartCrafter; see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _trimesh_module():
    try:
        import trimesh
    except ImportError as error:  # pragma: no cover - exercised by packaging only
        raise RuntimeError("Geometry metrics require trimesh; install .[iou]") from error
    return trimesh


def load_mesh(path: str | Path) -> Any:
    """Load one mesh or concatenate a scene with node transforms applied."""

    trimesh = _trimesh_module()
    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh does not exist: {mesh_path}")
    geometry = trimesh.load(mesh_path, process=False)
    if isinstance(geometry, trimesh.Scene):
        if not geometry.geometry:
            raise ValueError(f"Mesh scene contains no geometry: {mesh_path}")
        if hasattr(geometry, "to_geometry"):
            geometry = geometry.to_geometry()
        else:  # pragma: no cover - compatibility with older trimesh releases
            geometry = geometry.dump(concatenate=True)
    if not isinstance(geometry, trimesh.Trimesh):
        raise ValueError(f"Expected triangle mesh data at {mesh_path}")
    if len(geometry.vertices) == 0 or len(geometry.faces) == 0:
        raise ValueError(f"Mesh is empty: {mesh_path}")
    return geometry


def get_voxel_set(
    mesh: Any,
    num_grids: int = 64,
    scale: float = 2.0,
) -> set[tuple[int, int, int]]:
    """Return filled voxel centers using the camera-ready IoU convention."""

    trimesh = _trimesh_module()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if num_grids <= 0:
        raise ValueError("num_grids must be positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    pitch = float(scale) / int(num_grids)
    voxel_grid = mesh.voxelized(pitch=pitch).fill()
    indices = np.round(np.asarray(voxel_grid.points) / pitch).astype(np.int64)
    return {tuple(int(value) for value in row) for row in indices}


def compute_iou(
    mesh1: Any,
    mesh2: Any,
    num_grids: int = 64,
    scale: float = 2.0,
    return_counts: bool = False,
) -> float | tuple[float, int, int]:
    """Compute filled-voxel intersection over union for two aligned meshes."""

    voxels1 = get_voxel_set(mesh1, num_grids=num_grids, scale=scale)
    voxels2 = get_voxel_set(mesh2, num_grids=num_grids, scale=scale)
    intersection_count = len(voxels1.intersection(voxels2))
    union_count = len(voxels1.union(voxels2))
    iou = intersection_count / union_count if union_count else 0.0
    if return_counts:
        return iou, intersection_count, union_count
    return iou


def compute_IoU(
    mesh1: Any,
    mesh2: Any,
    num_grids: int = 64,
    scale: float = 2.0,
    return_counts: bool = False,
) -> float | tuple[float, int, int]:
    """Compatibility alias for the original camera-ready function name."""

    return compute_iou(mesh1, mesh2, num_grids, scale, return_counts)


def _sample_surface(mesh: Any, count: int, seed: int) -> np.ndarray:
    trimesh = _trimesh_module()
    if count <= 0:
        raise ValueError("num_samples must be positive")
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float64)


def _mutual_nearest_distances(
    points1: np.ndarray,
    points2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - exercised by packaging only
        raise RuntimeError("Chamfer distance requires scipy; install .[iou]") from error
    distance_1_to_2 = cKDTree(points2).query(points1, k=1)[0]
    distance_2_to_1 = cKDTree(points1).query(points2, k=1)[0]
    return distance_1_to_2, distance_2_to_1


def compute_chamfer_distance(
    mesh1: Any,
    mesh2: Any,
    num_samples: int = 10_000,
    seed: int = 0,
) -> float:
    """Return the non-squared symmetric surface Chamfer distance."""

    points1 = _sample_surface(mesh1, num_samples, seed)
    points2 = _sample_surface(mesh2, num_samples, seed)
    distance_1_to_2, distance_2_to_1 = _mutual_nearest_distances(points1, points2)
    return float(distance_1_to_2.mean() + distance_2_to_1.mean())


def compute_f_score(
    mesh1: Any,
    mesh2: Any,
    num_samples: int = 10_000,
    threshold: float = 0.1,
    seed: int = 0,
) -> float:
    """Return the symmetric surface F-score at a distance threshold."""

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    points1 = _sample_surface(mesh1, num_samples, seed)
    points2 = _sample_surface(mesh2, num_samples, seed)
    distance_1_to_2, distance_2_to_1 = _mutual_nearest_distances(points1, points2)
    precision1 = float(np.mean(distance_1_to_2 < threshold))
    precision2 = float(np.mean(distance_2_to_1 < threshold))
    denominator = precision1 + precision2
    return 2.0 * precision1 * precision2 / denominator if denominator else 0.0


def compute_geometry_metrics(
    gt_mesh: Any,
    pred_mesh: Any,
    *,
    num_grids: int = 64,
    scale: float = 2.0,
    num_samples: int = 10_000,
    f_score_threshold: float = 0.1,
    seed: int = 0,
    iou_only: bool = False,
) -> dict[str, Any]:
    """Compute deterministic metrics without performing global alignment."""

    iou, intersection, union = compute_iou(
        gt_mesh,
        pred_mesh,
        num_grids=num_grids,
        scale=scale,
        return_counts=True,
    )
    result: dict[str, Any] = {
        "coordinate_frame": "input_meshes_already_aligned",
        "num_grids": int(num_grids),
        "scale": float(scale),
        "pitch": float(scale) / int(num_grids),
        "iou": float(iou),
        "intersection_voxels": int(intersection),
        "union_voxels": int(union),
    }
    if not iou_only:
        result.update(
            {
                "num_samples": int(num_samples),
                "chamfer_distance": compute_chamfer_distance(
                    gt_mesh, pred_mesh, num_samples=num_samples, seed=seed
                ),
                "f_score_threshold": float(f_score_threshold),
                "f_score": compute_f_score(
                    gt_mesh,
                    pred_mesh,
                    num_samples=num_samples,
                    threshold=f_score_threshold,
                    seed=seed,
                ),
            }
        )
    return result


__all__ = [
    "compute_IoU",
    "compute_chamfer_distance",
    "compute_f_score",
    "compute_geometry_metrics",
    "compute_iou",
    "get_voxel_set",
    "load_mesh",
]
