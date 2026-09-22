"""Small, checked mesh helpers used by MOD preparation and paper training.

The functions in this module deliberately cover only the release's public
contracts.  They do not import SAM3D internals and they raise on invalid
geometry or failed export instead of silently returning an unposed mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .poses import normalize_quaternion


def _trimesh_module():
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Mesh processing requires trimesh; install .[train]") from exc
    return trimesh


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def transform_points_by_pose(points: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Apply WXYZ quaternion poses as local scale, rotation, then translation."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, P, 3], got {tuple(points.shape)}")
    if poses.shape != (points.shape[0], 10):
        raise ValueError(
            f"poses must have shape ({points.shape[0]}, 10), got {tuple(poses.shape)}"
        )
    if not torch.isfinite(points).all() or not torch.isfinite(poses).all():
        raise ValueError("points and poses must contain only finite values")
    if (poses[:, 7:10] <= 0).any():
        raise ValueError("pose scales must be positive")

    quaternion = normalize_quaternion(poses[:, 3:7])
    w, x, y, z = quaternion.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    # SAM3D composes PyTorch3D transforms as scale -> rotate -> translate.
    # PyTorch3D applies those transforms to row vectors, hence multiplication
    # by ``rotation`` itself rather than the column-vector ``rotation.T`` form.
    scaled = points * poses[:, None, 7:10]
    return torch.matmul(scaled, rotation) + poses[:, None, :3]


def mesh_result_to_trimesh(mesh_result: Any):
    """Convert one SAM3D mesh result into a validated Trimesh."""

    trimesh = _trimesh_module()
    if not hasattr(mesh_result, "vertices") or not hasattr(mesh_result, "faces"):
        raise TypeError("SAM3D mesh result must expose vertices and faces")
    vertices = np.asarray(_numpy(mesh_result.vertices), dtype=np.float64)
    faces = _numpy(mesh_result.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"mesh vertices must have shape [V, 3], got {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices contain NaN or infinity")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"mesh faces must have shape [F, 3], got {faces.shape}")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("mesh face indices must be integers")
    faces = np.asarray(faces, dtype=np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("mesh face indices are outside the vertex array")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    attributes = getattr(mesh_result, "vertex_attrs", None)
    if attributes is not None:
        colors = np.asarray(_numpy(attributes))
        if colors.ndim != 2 or colors.shape[0] != len(vertices) or colors.shape[1] < 3:
            raise ValueError("vertex_attrs must have shape [V, C] with C >= 3")
        colors = np.asarray(colors[:, :4], dtype=np.float64)
        if not np.isfinite(colors).all():
            raise ValueError("vertex colors contain NaN or infinity")
        if colors.min() < 0 or colors.max() > 255:
            raise ValueError("vertex colors must lie in [0, 1] or [0, 255]")
        if colors.max() <= 1.0:
            colors = colors * 255.0
        colors = np.rint(colors).astype(np.uint8)
        if colors.shape[1] == 3:
            alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
            colors = np.concatenate((colors, alpha), axis=1)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


def transform_mesh(
    mesh: Any,
    rotation: Any,
    translation: Any,
    scale: Any,
):
    """Return a copy of ``mesh`` transformed with one SAM3D-format pose."""

    trimesh = _trimesh_module()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh must contain vertices with shape [V, 3]")

    tensor_values = (rotation, translation, scale)
    device = next(
        (value.device for value in tensor_values if isinstance(value, torch.Tensor)),
        torch.device("cpu"),
    )
    dtype = next(
        (
            value.dtype
            for value in tensor_values
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ),
        torch.float32,
    )
    rotation_tensor = torch.as_tensor(rotation, device=device, dtype=dtype).reshape(-1)
    translation_tensor = torch.as_tensor(translation, device=device, dtype=dtype).reshape(-1)
    scale_tensor = torch.as_tensor(scale, device=device, dtype=dtype).reshape(-1)
    if rotation_tensor.numel() != 4:
        raise ValueError("rotation must contain one WXYZ quaternion")
    if translation_tensor.numel() != 3 or scale_tensor.numel() != 3:
        raise ValueError("translation and scale must each contain three values")
    pose = torch.cat((translation_tensor, rotation_tensor, scale_tensor)).reshape(1, 10)
    vertex_tensor = torch.as_tensor(vertices, device=device, dtype=dtype).unsqueeze(0)
    transformed_vertices = transform_points_by_pose(vertex_tensor, pose)[0]

    transformed = mesh.copy()
    transformed.vertices = transformed_vertices.detach().cpu().numpy()
    return transformed


def export_mesh_scene(
    meshes: Sequence[Any],
    names: Sequence[str],
    output_path: Path | str,
) -> Path:
    """Export named meshes to one GLB and fail if the result is not materialized."""

    trimesh = _trimesh_module()
    mesh_rows = list(meshes)
    name_rows = [str(name) for name in names]
    if not mesh_rows or len(mesh_rows) != len(name_rows):
        raise ValueError("meshes and names must have the same non-zero length")
    if any(not name.strip() for name in name_rows) or len(set(name_rows)) != len(name_rows):
        raise ValueError("mesh names must be non-empty and unique")

    scene = trimesh.Scene()
    for mesh, name in zip(mesh_rows, name_rows):
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"{name}: expected a trimesh.Trimesh")
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError(f"{name}: cannot export invalid or empty triangle geometry")
        scene.add_geometry(mesh.copy(), geom_name=name, node_name=name)

    destination = Path(output_path)
    if destination.suffix.lower() != ".glb":
        raise ValueError("combined scene output must use the .glb extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(destination, file_type="glb", include_normals=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Trimesh did not create a non-empty GLB: {destination}")
    return destination


__all__ = [
    "export_mesh_scene",
    "mesh_result_to_trimesh",
    "transform_mesh",
    "transform_points_by_pose",
]
