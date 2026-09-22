"""Generate supervised SAM3D-frame poses by registering annotated GT geometry.

This is a checked, release-owned wrapper around an explicitly supplied GAPs
``mshalign`` executable. RGB and instance masks alone are not supervision:
callers must provide a posed GT scene and an explicit object-to-scene-node map.
No failed registration is ever replaced with an identity transformation.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch

from .mesh_io import _trimesh_module, transform_mesh


_OBJECT_ID = re.compile(r"(?:obj_)?0*(\d+)")
_CANONICAL_NAME = re.compile(r"obj_0*(\d+)_local\.ply", re.IGNORECASE)
_MATRIX_ROW = re.compile(r"Matrix\[([0-3])\]\[0-3\]\s*=\s*(.*)")


def _object_id(value: Any) -> int:
    match = _OBJECT_ID.fullmatch(str(value))
    if match is None:
        raise ValueError(f"Invalid object ID {value!r}; use a numeric ID or obj_<ID>")
    return int(match.group(1))


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _checked_mesh(mesh: Any, label: str):
    trimesh = _trimesh_module()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{label}: expected triangle geometry")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError(f"{label}: invalid or empty triangle geometry")
    if not math.isfinite(float(mesh.area)) or mesh.area <= 0:
        raise ValueError(f"{label}: geometry must have positive finite surface area")
    return mesh


def _checked_affine(matrix: Any, label: str) -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label}: expected a finite 4x4 transformation")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError(f"{label}: invalid homogeneous matrix row")
    determinant = float(np.linalg.det(transform[:3, :3]))
    if not math.isfinite(determinant) or determinant == 0:
        raise ValueError(f"{label}: singular transformation")
    return transform


def _checked_similarity(matrix: Any, label: str) -> tuple[np.ndarray, float]:
    transform = _checked_affine(matrix, label)
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0:
        raise ValueError(f"{label}: reflections and nonpositive scales are not supported")
    scale = float(np.cbrt(determinant))
    rotation = linear / scale
    # GAPs prints six significant digits using %g; tolerate that quantization,
    # not anisotropic scaling or shearing in the registration transform.
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=2e-5, atol=2e-5):
        raise ValueError(f"{label}: expected a rotation and uniform positive scale")
    return transform, scale


def _parse_gaps_output(stdout: str) -> tuple[np.ndarray, dict[str, Any]]:
    marker = "Computed alignment transformation ..."
    if marker not in stdout:
        raise ValueError("GAPs stdout is missing the final alignment transformation")
    # Debug builds can print trial transforms. Only the final, explicitly
    # labelled alignment block is authoritative.
    block = stdout.rsplit(marker, 1)[1]
    rows: dict[int, list[float]] = {}
    values: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        row = _MATRIX_ROW.fullmatch(line)
        if row is not None:
            index = int(row.group(1))
            if index in rows:
                raise ValueError("GAPs stdout has duplicate transformation rows")
            try:
                numbers = [float(value) for value in row.group(2).split()]
            except ValueError as exc:
                raise ValueError("GAPs stdout contains an invalid matrix row") from exc
            if len(numbers) != 4:
                raise ValueError("GAPs stdout must contain four values per matrix row")
            rows[index] = numbers
        for field in ("Scale", "Converged", "RMSD"):
            match = re.fullmatch(rf"{field}\s*=\s*(\S+)", line)
            if match:
                if field in values:
                    raise ValueError(f"GAPs stdout has duplicate {field} diagnostics")
                values[field] = match.group(1)
    if set(rows) != {0, 1, 2, 3}:
        raise ValueError("GAPs stdout does not contain a complete 4x4 transformation")
    transform, scale = _checked_similarity([rows[i] for i in range(4)], "GAPs")
    if set(values) != {"Scale", "Converged", "RMSD"}:
        raise ValueError("GAPs stdout is missing Scale, Converged or RMSD diagnostics")
    reported_scale = float(values["Scale"])
    rmsd = float(values["RMSD"])
    if not math.isfinite(reported_scale) or reported_scale <= 0:
        raise ValueError("GAPs reported an invalid scale")
    if not math.isclose(reported_scale, scale, rel_tol=2e-5, abs_tol=1e-10):
        raise ValueError("GAPs scale diagnostic disagrees with its transformation")
    if values["Converged"] != "1":
        raise ValueError("GAPs registration did not converge")
    if not math.isfinite(rmsd) or rmsd < 0:
        raise ValueError("GAPs reported an invalid RMSD")
    return transform, {"scale": scale, "converged": True, "rmsd": rmsd}


def _register_meshes(
    source: Any,
    target: Any,
    *,
    binary: Path,
    timeout: float,
    work_dir: Path,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_path = work_dir / f"{label}_source.ply"
    target_path = work_dir / f"{label}_target.ply"
    aligned_path = work_dir / f"{label}_aligned.ply"
    _checked_mesh(source, f"{label} source").export(source_path)
    _checked_mesh(target, f"{label} target").export(target_path)
    command = [str(binary), str(source_path), str(target_path), str(aligned_path), "-v"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"GAPs {label} registration timed out after {timeout:g}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"GAPs {label} registration failed ({exc.returncode}): {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot execute GAPs binary {binary}: {exc}") from exc
    transform, diagnostics = _parse_gaps_output(result.stdout)
    if not aligned_path.is_file() or aligned_path.stat().st_size == 0:
        raise RuntimeError(f"GAPs {label} did not produce an aligned mesh")
    aligned = _checked_mesh(
        _trimesh_module().load(aligned_path, process=False), f"{label} aligned mesh"
    )
    expected = np.asarray(source.vertices) @ transform[:3, :3].T + transform[:3, 3]
    tolerance = max(1.0, float(np.abs(expected).max())) * 5e-5
    if aligned.vertices.shape != expected.shape or not np.allclose(
        aligned.vertices, expected, rtol=5e-5, atol=tolerance
    ):
        raise ValueError(f"GAPs {label} aligned mesh disagrees with its transformation")
    return transform, diagnostics


def _pose_matrix(pose: Any, label: str) -> np.ndarray:
    if not isinstance(pose, dict):
        raise ValueError(f"{label}: expected a pose mapping")
    fields = {}
    for field, size in (("translation", 3), ("rotation", 4), ("scale", 3)):
        value = np.asarray(pose.get(field), dtype=np.float64)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{label}: {field} must contain {size} finite numbers")
        fields[field] = value
    if np.any(fields["scale"] <= 0):
        raise ValueError(f"{label}: scale must be positive")
    quaternion_norm = float(np.linalg.norm(fields["rotation"]))
    if not math.isfinite(quaternion_norm) or quaternion_norm <= 1e-8:
        raise ValueError(f"{label}: rotation quaternion must be nonzero")
    rotation = _trimesh_module().transformations.quaternion_matrix(
        fields["rotation"] / quaternion_norm
    )[:3, :3]
    result = np.eye(4)
    # SAM3D applies scale @ R(q) to row points. Its corresponding column
    # matrix is R(q).T @ scale, not R(q) @ scale.
    result[:3, :3] = rotation.T @ np.diag(fields["scale"])
    result[:3, 3] = fields["translation"]
    return result


def _matrix_pose(matrix: np.ndarray) -> dict[str, list[float]]:
    matrix = _checked_affine(matrix, "Final object pose")
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Final object pose has invalid scales")
    rotation = linear / scale
    if np.linalg.det(rotation) <= 0 or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=5e-5, atol=5e-5
    ):
        raise ValueError("Final object pose contains reflection or shear")
    # Project only the tiny rounding error introduced by GAPs' printed matrix
    # to SO(3), then transpose back to SAM3D's row-vector quaternion convention.
    u, _, vh = np.linalg.svd(rotation)
    quaternion_matrix = np.eye(4)
    quaternion_matrix[:3, :3] = (u @ vh).T
    quaternion = _trimesh_module().transformations.quaternion_from_matrix(quaternion_matrix)
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion = -quaternion
    return {
        "translation": matrix[:3, 3].tolist(),
        "rotation": quaternion.tolist(),
        "scale": scale.tolist(),
    }


def generate_training_gt(
    *,
    canonical_dir: Path,
    base_poses_path: Path,
    gt_scene_path: Path,
    object_node_map: dict[str, str],
    output_path: Path,
    gaps_binary: Path,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Register geometry and atomically write targets in the SAM3D prior frame.

    ``object_node_map`` maps every base-pose object ID to exactly one GT scene
    graph node, not to a geometry index or a guessed sorted correspondence.
    The returned dictionary is also the JSON written to ``output_path``.
    Registration failure leaves any existing output unchanged. The generated
    targets are registration-derived labels and still require quality review.
    """
    trimesh = _trimesh_module()
    binary = Path(gaps_binary).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"GAPs binary is missing or not executable: {binary}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("GAPs timeout must be a positive finite number")
    canonical_dir = Path(canonical_dir)
    base_poses_path = Path(base_poses_path)
    gt_scene_path = Path(gt_scene_path)
    output_path = Path(output_path)
    payload = json.loads(base_poses_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    objects = payload.get("objects", payload) if isinstance(payload, dict) else None
    if not isinstance(objects, dict) or not objects:
        raise ValueError("Base poses must contain a non-empty objects mapping")
    poses_by_id = {}
    for name, pose in objects.items():
        object_id = _object_id(name)
        if object_id in poses_by_id:
            raise ValueError(f"Base poses contain duplicate normalized object ID {object_id}")
        poses_by_id[object_id] = (name, pose, _pose_matrix(pose, name))
    if not isinstance(object_node_map, dict) or not object_node_map:
        raise ValueError("An explicit, non-empty object_node_map is required")
    nodes_by_id: dict[int, str] = {}
    for name, node in object_node_map.items():
        object_id = _object_id(name)
        if object_id in nodes_by_id:
            raise ValueError(f"Object map contains duplicate normalized object ID {object_id}")
        if not isinstance(node, str) or not node.strip():
            raise ValueError(f"Object {name}: scene node must be a non-empty string")
        if node in nodes_by_id.values():
            raise ValueError(f"Object map must be one-to-one; duplicate scene node {node!r}")
        nodes_by_id[object_id] = node
    if set(nodes_by_id) != set(poses_by_id):
        raise ValueError(
            "Object map must cover exactly all base objects; "
            f"missing={sorted(set(poses_by_id) - set(nodes_by_id))}, "
            f"extra={sorted(set(nodes_by_id) - set(poses_by_id))}"
        )
    paths_by_id: dict[int, Path] = {}
    for path in sorted(canonical_dir.glob("*.ply")):
        match = _CANONICAL_NAME.fullmatch(path.name)
        if match is None:
            continue
        object_id = int(match.group(1))
        if object_id in paths_by_id:
            raise ValueError(f"Duplicate canonical mesh for object {object_id}")
        paths_by_id[object_id] = path
    missing_meshes = set(poses_by_id) - set(paths_by_id)
    if missing_meshes:
        raise ValueError(f"Missing canonical obj_<ID>_local.ply meshes: {sorted(missing_meshes)}")
    protected = [base_poses_path, gt_scene_path, binary, *paths_by_id.values()]
    if output_path.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Training GT output must not overwrite an input or executable")
    scene = trimesh.load(gt_scene_path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError("GT geometry must be loadable as a scene")
    predicted: dict[int, Any] = {}
    ground_truth: dict[int, Any] = {}
    for object_id in sorted(poses_by_id):
        name, pose, _ = poses_by_id[object_id]
        node = nodes_by_id[object_id]
        if node not in scene.graph.nodes_geometry:
            raise ValueError(f"{name}: GT scene has no geometry-bearing node {node!r}")
        world_transform, geometry_name = scene.graph[node]
        gt_mesh = _checked_mesh(scene.geometry[geometry_name].copy(), f"GT node {node}")
        gt_mesh.apply_transform(_checked_affine(world_transform, f"GT node {node}"))
        ground_truth[object_id] = _checked_mesh(gt_mesh, f"GT node {node} in world frame")
        local = _checked_mesh(
            trimesh.load(paths_by_id[object_id], process=False), f"{name} canonical mesh"
        )
        predicted[object_id] = transform_mesh(
            local,
            rotation=torch.as_tensor(pose["rotation"], dtype=torch.float64),
            translation=torch.as_tensor(pose["translation"], dtype=torch.float64),
            scale=torch.as_tensor(pose["scale"], dtype=torch.float64),
        )
    targets = {}
    object_diagnostics = {}
    with tempfile.TemporaryDirectory(prefix="mod-training-gt-") as directory:
        work_dir = Path(directory)
        scene_transform, scene_diagnostics = _register_meshes(
            trimesh.util.concatenate(list(predicted.values())),
            trimesh.util.concatenate(list(ground_truth.values())),
            binary=binary, timeout=timeout, work_dir=work_dir, label="scene",
        )
        scene_inverse = np.linalg.inv(scene_transform)
        for object_id in sorted(poses_by_id):
            name, _, base_matrix = poses_by_id[object_id]
            gt_in_prediction_frame = ground_truth[object_id].copy()
            gt_in_prediction_frame.apply_transform(scene_inverse)
            object_transform, diagnostics = _register_meshes(
                predicted[object_id], gt_in_prediction_frame,
                binary=binary, timeout=timeout, work_dir=work_dir, label=f"object_{object_id}",
            )
            targets[name] = _matrix_pose(object_transform @ base_matrix)
            object_diagnostics[name] = {
                **diagnostics,
                "gt_scene_node": nodes_by_id[object_id],
                "object_transform_sam3d_pred": object_transform.tolist(),
            }
    result = {
        "objects": targets,
        "metadata": {
            "output_pose_space": "sam3d_pred",
            "method": "gaps_mshalign_scene_then_object",
            "num_matched": len(targets),
            "scene_transform_sam3d_to_gt": scene_transform.tolist(),
            "scene_transform_gt_to_sam3d": scene_inverse.tolist(),
            "scene_alignment": scene_diagnostics,
            "object_alignments": object_diagnostics,
            "source_path_base": "target_json_directory",
            "gt_scene_path": os.path.relpath(gt_scene_path.resolve(), output_path.parent.resolve()),
            "base_poses_path": os.path.relpath(base_poses_path.resolve(), output_path.parent.resolve()),
            "gaps_binary": binary.name,
            "gaps_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return result


__all__ = ["generate_training_gt"]
