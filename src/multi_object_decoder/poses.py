"""Pose I/O and differentiable pose utilities used by training and inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .object_ids import object_id_sort_key, parse_prefixed_object_id


POSE_SIZE = 10


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Return unit quaternions in ``[w, x, y, z]`` order."""
    normalized = F.normalize(quaternion, dim=-1, eps=1e-8)
    identity = torch.zeros_like(normalized)
    identity[..., 0] = 1.0
    finite = torch.isfinite(normalized).all(dim=-1, keepdim=True)
    return torch.where(finite, normalized, identity)


def apply_pose_delta(base_pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply an identity-centred translation/quaternion/log-scale residual."""
    if base_pose.shape != delta.shape or base_pose.shape[-1] != POSE_SIZE:
        raise ValueError(
            f"base_pose and delta must both have shape (..., {POSE_SIZE}); "
            f"got {tuple(base_pose.shape)} and {tuple(delta.shape)}"
        )
    translation = base_pose[..., :3] + delta[..., :3]
    rotation = normalize_quaternion(base_pose[..., 3:7] + delta[..., 3:7])
    # Log-scale residuals preserve positive scale and make zero exactly identity.
    scale = base_pose[..., 7:10].clamp_min(1e-6) * torch.exp(
        delta[..., 7:10].clamp(min=-4.0, max=4.0)
    )
    return torch.cat((translation, rotation, scale), dim=-1)


def pose_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    scale_weight: float = 1.0,
    scale_loss_mode: str = "log_mse",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the MOD translation, sign-invariant quaternion and scale losses.

    ``log_mse`` is the portable direct-backend default. ``raw_mse`` reproduces
    the scale objective recovered from the paper-working-tree trainer and must
    be selected explicitly by its training configuration.
    """
    if predicted.shape != target.shape or predicted.shape[-1] != POSE_SIZE:
        raise ValueError(
            f"predicted and target poses must match with last dimension {POSE_SIZE}; "
            f"got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    translation = F.mse_loss(predicted[..., :3], target[..., :3])
    pred_quat = normalize_quaternion(predicted[..., 3:7])
    target_quat = normalize_quaternion(target[..., 3:7])
    rotation = (1.0 - (pred_quat * target_quat).sum(dim=-1).abs()).mean()
    if scale_loss_mode == "log_mse":
        # Work in log space so small and large objects receive comparable gradients.
        scale = F.mse_loss(
            predicted[..., 7:10].clamp_min(1e-6).log(),
            target[..., 7:10].clamp_min(1e-6).log(),
        )
    elif scale_loss_mode == "raw_mse":
        scale = F.mse_loss(predicted[..., 7:10], target[..., 7:10])
    else:
        raise ValueError("scale_loss_mode must be 'log_mse' or 'raw_mse'")
    total = (
        float(translation_weight) * translation
        + float(rotation_weight) * rotation
        + float(scale_weight) * scale
    )
    return total, {
        "loss": total.detach(),
        "translation_loss": translation.detach(),
        "rotation_loss": rotation.detach(),
        "scale_loss": scale.detach(),
    }


def _pose_vector(value: Mapping[str, Any], *, source: Path, object_name: str) -> list[float]:
    translation = value.get("translation")
    rotation = value.get("rotation", value.get("quaternion"))
    scale = value.get("scale")
    if translation is None or rotation is None or scale is None:
        raise ValueError(
            f"{source}: {object_name} must contain translation, rotation and scale"
        )
    result = [float(item) for item in (*translation, *rotation, *scale)]
    if len(result) != POSE_SIZE or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{source}: invalid pose for {object_name}")
    return result


def load_pose_json(path: Path | str) -> tuple[list[str], torch.Tensor]:
    """Load ``{"objects": {"obj_000": ...}}`` while preserving numeric ID order."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    objects = payload.get("objects", payload)
    if not isinstance(objects, Mapping) or not objects:
        raise ValueError(f"{source}: expected a non-empty object mapping")
    names = sorted((str(name) for name in objects), key=object_id_sort_key)
    poses = torch.tensor(
        [_pose_vector(objects[name], source=source, object_name=name) for name in names],
        dtype=torch.float32,
    )
    poses[:, 3:7] = normalize_quaternion(poses[:, 3:7])
    if (poses[:, 7:10] <= 0).any():
        raise ValueError(f"{source}: pose scale must be positive")
    return names, poses


def align_pose_tensor(
    source_names: Sequence[str],
    source_poses: torch.Tensor,
    object_names: Sequence[str],
    *,
    source: Path | str,
) -> torch.Tensor:
    """Align a pose tensor to token object IDs and reject ambiguous matches."""
    by_id: dict[int, torch.Tensor] = {}
    by_name = {name: source_poses[index] for index, name in enumerate(source_names)}
    for index, name in enumerate(source_names):
        object_id = parse_prefixed_object_id(name)
        if object_id is not None:
            by_id[object_id] = source_poses[index]

    aligned = []
    missing = []
    for name in object_names:
        if name in by_name:
            aligned.append(by_name[name])
            continue
        object_id = parse_prefixed_object_id(name)
        if object_id is not None and object_id in by_id:
            aligned.append(by_id[object_id])
            continue
        missing.append(name)
    if missing:
        raise ValueError(f"{source}: missing poses for object IDs {missing}")
    return torch.stack(aligned, dim=0)


def poses_to_json(
    object_names: Iterable[str],
    poses: torch.Tensor,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an ``(N, 10)`` tensor into the public MOD pose JSON schema."""
    pose_values = poses.detach().cpu().float()
    names = list(object_names)
    if pose_values.shape != (len(names), POSE_SIZE):
        raise ValueError(
            f"Expected poses with shape ({len(names)}, {POSE_SIZE}), got {tuple(pose_values.shape)}"
        )
    if not bool(torch.isfinite(pose_values).all()):
        raise ValueError("Poses contain NaN or infinity; refusing to write pose JSON")
    objects: dict[str, Any] = {}
    for name, pose in zip(names, pose_values):
        objects[name] = {
            "translation": pose[:3].tolist(),
            "rotation": normalize_quaternion(pose[3:7]).tolist(),
            "scale": pose[7:10].tolist(),
        }
    payload: dict[str, Any] = {"objects": objects}
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def save_pose_json(
    path: Path | str,
    object_names: Iterable[str],
    poses: torch.Tensor,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = poses_to_json(object_names, poses, metadata=metadata)
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
