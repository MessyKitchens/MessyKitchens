"""Checked training objectives for portable and paper MOD workflows.

The portable workflow keeps its historical log-scale pose loss. The paper
objective is an explicit opt-in: it uses raw scale MSE and the PyTorch3D
Chamfer implementation recovered from the paper working tree.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .mesh_io import transform_points_by_pose
from .poses import pose_loss


TensorTransform = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ChamferFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_CANONICAL_NAME = re.compile(r"^obj_0*(\d+)_local\.(npy|ply)$", re.IGNORECASE)


@dataclass(frozen=True)
class ChamferSettings:
    """Configuration for the optional geometric training loss."""

    enabled: bool = False
    backend: str = "none"
    canonical_source: str = "auto"
    canonical_point_count: int = 4096
    num_samples: int = 1024
    sampling_seed: int = 0
    cache_size: int = 16


@dataclass(frozen=True)
class ObjectiveSettings:
    """Resolved MOD objective settings."""

    profile: str = "portable"
    scale_loss: str = "log_mse"
    chamfer: ChamferSettings = ChamferSettings()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        workflow_backend: str,
    ) -> "ObjectiveSettings":
        config = dict(value or {})
        unknown = sorted(set(config) - {"profile", "scale_loss", "chamfer"})
        if unknown:
            raise ValueError(f"Unknown train.objective keys: {unknown}")

        profile = str(config.get("profile", "portable")).lower()
        if profile not in {"portable", "paper"}:
            raise ValueError("train.objective.profile must be 'portable' or 'paper'")
        scale_loss = str(config.get("scale_loss", "log_mse")).lower()
        if scale_loss not in {"log_mse", "raw_mse"}:
            raise ValueError("train.objective.scale_loss must be 'log_mse' or 'raw_mse'")

        raw_chamfer = config.get("chamfer", {})
        if raw_chamfer is None:
            raw_chamfer = {}
        if not isinstance(raw_chamfer, Mapping):
            raise ValueError("train.objective.chamfer must be a mapping")
        chamfer_config = dict(raw_chamfer)
        allowed_chamfer = {
            "enabled",
            "backend",
            "canonical_source",
            "canonical_point_count",
            "num_samples",
            "sampling_seed",
            "cache_size",
        }
        unknown_chamfer = sorted(set(chamfer_config) - allowed_chamfer)
        if unknown_chamfer:
            raise ValueError(f"Unknown train.objective.chamfer keys: {unknown_chamfer}")

        enabled_value = chamfer_config.get("enabled", False)
        if not isinstance(enabled_value, bool):
            raise ValueError("train.objective.chamfer.enabled must be true or false")
        enabled = enabled_value
        backend = str(chamfer_config.get("backend", "none")).lower()
        if backend not in {"none", "pytorch3d", "torch"}:
            raise ValueError(
                "train.objective.chamfer.backend must be 'none', 'pytorch3d', or 'torch'"
            )
        if enabled and backend == "none":
            raise ValueError("Enabled Chamfer loss requires an explicit backend")
        if not enabled and backend != "none":
            raise ValueError("A Chamfer backend may only be set when chamfer.enabled=true")

        source = str(chamfer_config.get("canonical_source", "auto")).lower()
        if source not in {"auto", "points", "mesh"}:
            raise ValueError(
                "train.objective.chamfer.canonical_source must be 'auto', 'points', or 'mesh'"
            )
        point_count = int(chamfer_config.get("canonical_point_count", 4096))
        num_samples = int(chamfer_config.get("num_samples", 1024))
        sampling_seed = int(chamfer_config.get("sampling_seed", 0))
        cache_size = int(chamfer_config.get("cache_size", 16))
        if point_count <= 0 or num_samples <= 0:
            raise ValueError("Chamfer point counts must be positive")
        if num_samples > point_count:
            raise ValueError("Chamfer num_samples cannot exceed canonical_point_count")
        if cache_size < 0:
            raise ValueError("Chamfer cache_size cannot be negative")

        chamfer = ChamferSettings(
            enabled=enabled,
            backend=backend,
            canonical_source=source,
            canonical_point_count=point_count,
            num_samples=num_samples,
            sampling_seed=sampling_seed,
            cache_size=cache_size,
        )
        settings = cls(profile=profile, scale_loss=scale_loss, chamfer=chamfer)
        settings._validate_profile(workflow_backend)
        return settings

    def _validate_profile(self, workflow_backend: str) -> None:
        if self.profile != "paper":
            return
        failures = []
        if workflow_backend.lower() != "sam3d":
            failures.append("workflow.backend=sam3d")
        if self.scale_loss != "raw_mse":
            failures.append("scale_loss=raw_mse")
        if not self.chamfer.enabled:
            failures.append("chamfer.enabled=true")
        if self.chamfer.backend != "pytorch3d":
            failures.append("chamfer.backend=pytorch3d")
        if failures:
            raise ValueError(
                "The paper objective requires " + ", ".join(failures) + "; no fallback is used"
            )


def _canonical_paths(
    directory: Path | str,
    object_ids: Sequence[int],
    *,
    source: str,
) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Canonical geometry directory does not exist: {root}")
    expected = tuple(int(object_id) for object_id in object_ids)
    if not expected or len(set(expected)) != len(expected) or any(value < 0 for value in expected):
        raise ValueError("object_ids must be unique non-negative integers")

    by_kind: dict[str, dict[int, Path]] = {"npy": {}, "ply": {}}
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = _CANONICAL_NAME.fullmatch(path.name)
        if match is None:
            continue
        object_id, kind = int(match.group(1)), match.group(2).lower()
        previous = by_kind[kind].get(object_id)
        if previous is not None:
            raise ValueError(
                f"Ambiguous canonical geometry for object {object_id}: {previous} and {path}"
            )
        by_kind[kind][object_id] = path

    allowed_kinds = {
        "points": ("npy",),
        "mesh": ("ply",),
        "auto": ("npy", "ply"),
    }[source]
    discovered = set().union(*(set(by_kind[kind]) for kind in allowed_kinds))
    expected_set = set(expected)
    if discovered != expected_set:
        raise FileNotFoundError(
            f"Canonical object IDs in {root} do not match cache object_ids; "
            f"missing={sorted(expected_set - discovered)}, "
            f"unexpected={sorted(discovered - expected_set)}"
        )

    result = []
    for object_id in expected:
        # ``auto`` deliberately matches the recovered loader preference without
        # creating the old obj_*_local.npy side-effect.
        if source in {"auto", "points"} and object_id in by_kind["npy"]:
            result.append(by_kind["npy"][object_id])
        else:
            result.append(by_kind["ply"][object_id])
    return result


def _validate_point_array(points: np.ndarray, *, source: Path) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError(f"{source}: canonical points must have shape [P, 3]")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{source}: canonical points must be finite numeric values")
    return np.asarray(values, dtype=np.float32)


def _sample_mesh(path: Path, *, count: int, seed: int) -> np.ndarray:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Loading canonical PLY meshes requires trimesh; install the train extra"
        ) from exc

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if len(geometry.vertices)]
        if len(geometries) != 1:
            raise ValueError(f"{path}: expected exactly one non-empty mesh geometry")
        loaded = geometries[0]
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"{path}: expected a non-empty triangle mesh")

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0:
        raise ValueError(f"{path}: mesh has no finite positive-area faces")

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(triangles), size=count, replace=True, p=areas / total_area)
    chosen = triangles[selected]
    root_u = np.sqrt(rng.random((count, 1)))
    v = rng.random((count, 1))
    points = (
        (1.0 - root_u) * chosen[:, 0]
        + root_u * (1.0 - v) * chosen[:, 1]
        + root_u * v * chosen[:, 2]
    )
    if not np.isfinite(points).all():
        raise ValueError(f"{path}: sampled non-finite canonical points")
    return points.astype(np.float32)


def load_canonical_points(
    directory: Path | str,
    object_ids: Sequence[int],
    *,
    source: str = "auto",
    point_count: int = 4096,
    sampling_seed: int = 0,
) -> torch.Tensor:
    """Load canonical geometry in exact ``object_ids`` order without writing caches."""
    if source not in {"auto", "points", "mesh"}:
        raise ValueError("source must be 'auto', 'points', or 'mesh'")
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    paths = _canonical_paths(directory, object_ids, source=source)
    rows = []
    for object_id, path in zip(object_ids, paths):
        seed = int(sampling_seed) + int(object_id) * 1_000_003
        if path.suffix.lower() == ".npy":
            try:
                values = np.load(path, allow_pickle=False)
            except Exception as exc:
                raise ValueError(f"Could not load canonical points {path}: {exc}") from exc
            # Match the recovered working-tree behavior: pre-sampled NPY points are used
            # unchanged. ``point_count`` applies only to on-the-fly PLY sampling.
            points = _validate_point_array(values, source=path)
        else:
            points = _sample_mesh(path, count=point_count, seed=seed)
        rows.append(torch.from_numpy(points))
    row_shapes = {tuple(row.shape) for row in rows}
    if len(row_shapes) != 1:
        raise ValueError(
            "Canonical point clouds must have one shared shape for batching; "
            f"found {sorted(row_shapes)}"
        )
    return torch.stack(rows, dim=0).contiguous()


def torch_chamfer_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """PyTorch-only equivalent of PyTorch3D's default symmetric squared CD."""
    if first.ndim != 3 or second.ndim != 3 or first.shape[0] != second.shape[0]:
        raise ValueError("Chamfer inputs must have shapes [N, P, 3] and [N, Q, 3]")
    distances = torch.cdist(first, second, p=2).square()
    return (
        distances.min(dim=2).values.mean(dim=1)
        + distances.min(dim=1).values.mean(dim=1)
    ).mean()


def _require_pytorch3d_chamfer() -> Callable[..., Any]:
    try:
        from pytorch3d.loss import chamfer_distance
    except Exception as exc:
        raise RuntimeError(
            "train.objective.chamfer.backend=pytorch3d requires PyTorch3D; install "
            "the revision pinned by the selected SAM3D checkout"
        ) from exc
    return chamfer_distance


def _pytorch3d_chamfer(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    chamfer_distance = _require_pytorch3d_chamfer()
    result, _ = chamfer_distance(first, second, batch_reduction="mean")
    return result


def compute_pose_chamfer_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    canonical_points: torch.Tensor,
    *,
    num_samples: int,
    backend: str,
    transform_fn: TensorTransform | None = None,
    chamfer_fn: ChamferFunction | None = None,
) -> torch.Tensor:
    """Compute differentiable scene-space CD from one canonical cloud per object."""
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 10:
        raise ValueError("predicted and target must have matching shape [N, 10]")
    if (
        canonical_points.ndim != 3
        or canonical_points.shape[0] != predicted.shape[0]
        or canonical_points.shape[-1] != 3
    ):
        raise ValueError("canonical_points must have shape [N, P, 3] matching the poses")
    if not torch.isfinite(canonical_points).all():
        raise ValueError("canonical_points contains NaN or infinity")
    point_count = canonical_points.shape[1]
    if num_samples <= 0 or num_samples > point_count:
        raise ValueError(f"num_samples must be in [1, {point_count}]")
    if num_samples < point_count:
        indices = torch.randperm(point_count, device=canonical_points.device)[:num_samples]
        canonical_points = canonical_points.index_select(1, indices)

    if backend == "pytorch3d":
        transform = transform_fn or transform_points_by_pose
        chamfer = chamfer_fn or _pytorch3d_chamfer
    elif backend == "torch":
        transform = transform_fn or transform_points_by_pose
        chamfer = chamfer_fn or torch_chamfer_distance
    else:
        raise ValueError("backend must be 'pytorch3d' or 'torch'")
    predicted_points = transform(canonical_points, predicted)
    target_points = transform(canonical_points, target)
    result = chamfer(predicted_points, target_points)
    if result.ndim != 0 or not torch.isfinite(result):
        raise FloatingPointError("Chamfer backend returned a non-finite scalar")
    return result


class TrainingObjective:
    """Stateful objective with an in-memory-only canonical point cache."""

    def __init__(
        self,
        settings: ObjectiveSettings,
        loss_weights: Mapping[str, float],
        *,
        transform_fn: TensorTransform | None = None,
        chamfer_fn: ChamferFunction | None = None,
    ) -> None:
        self.settings = settings
        self.weights = {
            name: float(loss_weights.get(name, 1.0))
            for name in ("translation", "rotation", "scale", "cd")
        }
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("Loss weights must be finite and non-negative")
        self.transform_fn = transform_fn
        self.chamfer_fn = chamfer_fn
        self._canonical_cache: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()

        if settings.chamfer.enabled and settings.chamfer.backend == "pytorch3d":
            # Resolve the required implementation at startup, never halfway
            # through a long paper run. Test-only dependency injection can
            # exercise the surrounding logic on CPU without PyTorch3D.
            if self.chamfer_fn is None:
                _require_pytorch3d_chamfer()

    def _canonical_points(self, sample: Any) -> torch.Tensor:
        directory = sample.record.get("sam3d_canonical_dir")
        if directory is None or not str(directory).strip():
            raise ValueError(
                f"{sample.scene_id}: paper Chamfer loss requires sam3d_canonical_dir"
            )
        settings = self.settings.chamfer
        key = (
            str(Path(directory).expanduser().resolve()),
            tuple(sample.object_ids),
            settings.canonical_source,
            settings.canonical_point_count,
            settings.sampling_seed,
        )
        cached = self._canonical_cache.pop(key, None)
        if cached is not None:
            self._canonical_cache[key] = cached
            return cached
        points = load_canonical_points(
            key[0],
            sample.object_ids,
            source=settings.canonical_source,
            point_count=settings.canonical_point_count,
            sampling_seed=settings.sampling_seed,
        )
        if settings.cache_size:
            self._canonical_cache[key] = points
            while len(self._canonical_cache) > settings.cache_size:
                self._canonical_cache.popitem(last=False)
        return points

    def __call__(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        sample: Any,
        *,
        include_chamfer: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, components = pose_loss(
            predicted,
            target,
            translation_weight=self.weights["translation"],
            rotation_weight=self.weights["rotation"],
            scale_weight=self.weights["scale"],
            scale_loss_mode=self.settings.scale_loss,
        )
        if include_chamfer and self.settings.chamfer.enabled:
            canonical = self._canonical_points(sample).to(
                device=predicted.device, dtype=predicted.dtype
            )
            cd_loss = compute_pose_chamfer_loss(
                predicted,
                target,
                canonical,
                num_samples=self.settings.chamfer.num_samples,
                backend=self.settings.chamfer.backend,
                transform_fn=self.transform_fn,
                chamfer_fn=self.chamfer_fn,
            )
            loss = loss + self.weights["cd"] * cd_loss
            components["cd_loss"] = cd_loss.detach()
            components["loss"] = loss.detach()
        return loss, components


__all__ = [
    "ChamferSettings",
    "ObjectiveSettings",
    "TrainingObjective",
    "compute_pose_chamfer_loss",
    "load_canonical_points",
    "torch_chamfer_distance",
    "transform_points_by_pose",
]
