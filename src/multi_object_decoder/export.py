"""Export refined MOD poses as posed object meshes and a combined GLB scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .mesh_io import transform_mesh
from .object_ids import parse_prefixed_object_id


def _canonical_directory(record: Mapping[str, Any]) -> Path:
    explicit = record.get("sam3d_canonical_dir")
    if explicit:
        return Path(explicit)
    tokens_path = record.get("tokens_path")
    if tokens_path:
        token_path = Path(tokens_path)
        parent = token_path if token_path.is_dir() else token_path.parent
        for candidate in (parent.parent / "local_meshes", parent / "local_meshes", parent.parent):
            if candidate.is_dir():
                return candidate
    raise ValueError("Record does not identify a SAM3D canonical mesh directory")


def export_posed_meshes(
    record: Mapping[str, Any],
    object_names: Sequence[str],
    poses: torch.Tensor,
    output_dir: Path | str,
) -> list[Path]:
    """Write one posed PLY per object plus scene.glb.

    trimesh is imported lazily so pose-only inference has no mesh dependency.
    """
    try:
        import trimesh
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Mesh export requires the 'trimesh' package") from error

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    canonical_dir = _canonical_directory(record)
    scene = trimesh.Scene()
    outputs: list[Path] = []
    for index, (name, pose) in enumerate(zip(object_names, poses.detach().cpu())):
        object_id = parse_prefixed_object_id(name)
        candidates = []
        if object_id is not None:
            candidates.extend(
                (
                    canonical_dir / f"obj_{object_id:03d}_local.ply",
                    canonical_dir / f"obj_{object_id}_local.ply",
                )
            )
        candidates.append(canonical_dir / f"obj_{index}_local.ply")
        mesh_path = next((path for path in candidates if path.is_file()), None)
        if mesh_path is None:
            raise FileNotFoundError(
                f"No canonical mesh for {name} in {canonical_dir}; tried "
                + ", ".join(path.name for path in candidates)
            )
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{mesh_path}: expected a single triangle mesh")
        transformed = transform_mesh(
            mesh,
            rotation=pose[3:7],
            translation=pose[:3],
            scale=pose[7:10],
        )
        output_path = destination / f"{name}.ply"
        transformed.export(output_path)
        outputs.append(output_path)
        scene.add_geometry(transformed, geom_name=name, node_name=name)
    scene_path = destination / "scene.glb"
    scene.export(scene_path)
    outputs.append(scene_path)
    return outputs
