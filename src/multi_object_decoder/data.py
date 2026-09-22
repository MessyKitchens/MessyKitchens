"""Strict cached-scene loading for MOD training and inference.

The original camera-ready scripts loaded several loosely coupled files in the
training loop.  This module defines one checked boundary for those artifacts:

* legacy ``tokens/obj_<id>.npz`` files plus ``img_cond_<id>.npy`` sidecars;
* a portable, pickle-free, scene-level ``.npz`` file; and
* base/target pose JSON files aligned by the numeric object id.

Relative paths in a record are always resolved against the manifest directory,
never against the process working directory.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]

POSE_TOKEN_KEYS: Tuple[str, ...] = (
    "6drotation_normalized",
    "translation",
    "scale",
    "translation_scale",
)
POSE_LATENT_KEYS: Tuple[str, ...] = POSE_TOKEN_KEYS + ("shape",)
_POSE_OUTPUT_DIMS: Mapping[str, int] = {
    "6drotation_normalized": 6,
    "translation": 3,
    "scale": 3,
    "translation_scale": 1,
    "shape": 8,
}
_OBJECT_NAME_RE = re.compile(r"^obj_0*(\d+)$")
_PORTABLE_PREFIX_SEPARATORS = ("__", "/", ".")


class ArtifactSchemaError(ValueError):
    """Raised when a cached MOD artifact does not satisfy the release schema."""


def _move_tree(value: Any, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> Any:
    if isinstance(value, torch.Tensor):
        target_dtype = dtype if dtype is not None and value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, dict):
        return {key: _move_tree(item, device, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device, dtype) for item in value]
    return value


@dataclass(frozen=True)
class DecodeState:
    """SAM3D state required to decode a pose-refiner residual.

    All tensor-valued mappings are stacked along object dimension zero and use
    the same numeric order as :attr:`object_ids`.
    """

    object_ids: Tuple[int, ...]
    pose_latents_raw: Mapping[str, torch.Tensor]
    pointmap_scale: torch.Tensor
    pointmap_shift: torch.Tensor
    solver_state: Tuple[Mapping[str, Any], ...]
    x_t_last_step: Mapping[str, torch.Tensor]
    image_condition_per_object: torch.Tensor
    source_path: Path

    @property
    def image_condition(self) -> torch.Tensor:
        """Short alias used by backend adapters."""

        return self.image_condition_per_object

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DecodeState":
        target_device = torch.device(device) if device is not None else None
        return replace(
            self,
            pose_latents_raw=_move_tree(dict(self.pose_latents_raw), target_device, dtype),
            pointmap_scale=_move_tree(self.pointmap_scale, target_device, dtype),
            pointmap_shift=_move_tree(self.pointmap_shift, target_device, dtype),
            solver_state=_move_tree(tuple(self.solver_state), target_device, dtype),
            x_t_last_step=_move_tree(dict(self.x_t_last_step), target_device, dtype),
            image_condition_per_object=_move_tree(
                self.image_condition_per_object, target_device, dtype
            ),
        )


@dataclass(frozen=True)
class SceneSample:
    """One multi-object MOD scene, ready for training or refinement."""

    scene_id: str
    object_names: Tuple[str, ...]
    pose_tokens: torch.Tensor
    shape_tokens: torch.Tensor
    base_poses: torch.Tensor
    target_poses: Optional[torch.Tensor]
    decode_state: DecodeState
    record: Mapping[str, Any]

    @property
    def object_ids(self) -> Tuple[int, ...]:
        return self.decode_state.object_ids

    @property
    def num_objects(self) -> int:
        return len(self.object_names)

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SceneSample":
        target_device = torch.device(device) if device is not None else None
        return replace(
            self,
            pose_tokens=_move_tree(self.pose_tokens, target_device, dtype),
            shape_tokens=_move_tree(self.shape_tokens, target_device, dtype),
            base_poses=_move_tree(self.base_poses, target_device, dtype),
            target_poses=(
                None
                if self.target_poses is None
                else _move_tree(self.target_poses, target_device, dtype)
            ),
            decode_state=self.decode_state.to(target_device, dtype),
        )


@dataclass(frozen=True)
class _LoadedArtifacts:
    object_ids: Tuple[int, ...]
    pose_latents_raw: Mapping[str, torch.Tensor]
    pointmap_scale: torch.Tensor
    pointmap_shift: torch.Tensor
    solver_state: Tuple[Mapping[str, Any], ...]
    x_t_last_step: Mapping[str, torch.Tensor]
    image_condition_per_object: torch.Tensor
    source_path: Path
    embedded_base_poses: Optional[torch.Tensor] = None
    embedded_target_poses: Optional[torch.Tensor] = None


def _schema_error(location: Union[str, Path], message: str) -> ArtifactSchemaError:
    return ArtifactSchemaError(f"{location}: {message}")


def _manifest_base(manifest_dir: PathLike) -> Path:
    path = Path(manifest_dir).expanduser()
    # Accept either the requested manifest directory or the manifest path itself.
    if path.suffix.lower() == ".json" or path.is_file():
        path = path.parent
    return path.resolve()


def _resolve_record_path(value: Any, manifest_dir: Path, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise _schema_error(field, "expected a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _numeric_object_id(value: Union[str, Path]) -> int:
    stem = Path(str(value)).stem
    match = _OBJECT_NAME_RE.fullmatch(stem)
    if match is None:
        raise _schema_error(value, "object name must match obj_<numeric-id>")
    return int(match.group(1))


def _as_tensor(value: Any, location: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        array = np.asarray(value)
        if array.dtype == object:
            raise _schema_error(location, "expected a numeric array, found object dtype")
        if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
            raise _schema_error(location, f"expected numeric dtype, found {array.dtype}")
        tensor = torch.from_numpy(np.array(array, copy=True))
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor).all()):
            raise _schema_error(location, "contains NaN or infinity")
        tensor = tensor.float()
    return tensor.contiguous()


def _unbox_object(value: Any, location: str) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size != 1:
            raise _schema_error(location, "object array must contain exactly one mapping")
        return value.reshape(()).item()
    return value


def _tensor_tree(value: Any, location: str) -> Any:
    value = _unbox_object(value, location)
    if isinstance(value, Mapping):
        return {str(key): _tensor_tree(item, f"{location}.{key}") for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_tree(item, f"{location}[{index}]") for index, item in enumerate(value))
    if isinstance(value, list):
        return [_tensor_tree(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise _schema_error(location, "contains NaN or infinity")
        return value
    return _as_tensor(value, location)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    value = _unbox_object(value, location)
    if not isinstance(value, Mapping):
        raise _schema_error(location, f"expected mapping, found {type(value).__name__}")
    return value


def _validate_raw_latents(
    latents: Mapping[str, torch.Tensor], location: str, expected_objects: int
) -> None:
    missing = [key for key in POSE_LATENT_KEYS if key not in latents]
    if missing:
        raise _schema_error(location, f"missing pose_latents_raw keys: {missing}")
    unexpected = sorted(set(latents) - set(POSE_LATENT_KEYS))
    if unexpected:
        raise _schema_error(location, f"unexpected pose_latents_raw keys: {unexpected}")

    feature_dim: Optional[int] = None
    for key in POSE_LATENT_KEYS:
        tensor = latents[key]
        if tensor.ndim != 3 or tensor.shape[0] != expected_objects:
            raise _schema_error(
                f"{location}.{key}",
                f"expected [N,T,F] with N={expected_objects}, found {tuple(tensor.shape)}",
            )
        if key != "shape" and tensor.shape[1] != 1:
            raise _schema_error(
                f"{location}.{key}", f"pose latent must have one token, found {tensor.shape[1]}"
            )
        if tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
            raise _schema_error(
                f"{location}.{key}", "token and feature dimensions must be positive"
            )
        if feature_dim is None:
            feature_dim = tensor.shape[2]
        elif tensor.shape[2] != feature_dim:
            raise _schema_error(
                f"{location}.{key}",
                f"feature dimension {tensor.shape[2]} does not match {feature_dim}",
            )


def _validate_x_t(
    x_t: Mapping[str, torch.Tensor],
    location: str,
    expected_objects: int,
    expected_shape_tokens: int,
) -> None:
    missing = [key for key in POSE_LATENT_KEYS if key not in x_t]
    if missing:
        raise _schema_error(location, f"missing x_t_last_step keys: {missing}")
    unexpected = sorted(set(x_t) - set(POSE_LATENT_KEYS))
    if unexpected:
        raise _schema_error(location, f"unexpected x_t_last_step keys: {unexpected}")
    for key in POSE_LATENT_KEYS:
        tensor = x_t[key]
        expected_tokens = expected_shape_tokens if key == "shape" else 1
        expected_dim = _POSE_OUTPUT_DIMS[key]
        expected = (expected_objects, expected_tokens, expected_dim)
        if tuple(tensor.shape) != expected:
            raise _schema_error(
                f"{location}.{key}", f"expected shape {expected}, found {tuple(tensor.shape)}"
            )


def _validate_solver_state(states: Sequence[Mapping[str, Any]], location: str, count: int) -> None:
    if len(states) != count:
        raise _schema_error(location, f"expected {count} solver states, found {len(states)}")
    for index, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise _schema_error(f"{location}[{index}]", "expected mapping")
        missing = [key for key in ("t", "dt") if key not in state]
        if missing:
            raise _schema_error(f"{location}[{index}]", f"missing keys: {missing}")
        if state["t"] is None or state["dt"] is None:
            raise _schema_error(f"{location}[{index}]", "t and dt must not be null")


def _validate_loaded(artifacts: _LoadedArtifacts) -> None:
    count = len(artifacts.object_ids)
    if count == 0:
        raise _schema_error(artifacts.source_path, "scene contains no objects")
    if len(set(artifacts.object_ids)) != count:
        raise _schema_error(artifacts.source_path, "numeric object ids are not unique")
    if tuple(sorted(artifacts.object_ids)) != artifacts.object_ids:
        raise _schema_error(artifacts.source_path, "object ids are not in numeric order")

    _validate_raw_latents(artifacts.pose_latents_raw, "pose_latents_raw", count)
    shape_tokens = artifacts.pose_latents_raw["shape"].shape[1]
    _validate_x_t(artifacts.x_t_last_step, "x_t_last_step", count, shape_tokens)
    _validate_solver_state(artifacts.solver_state, "solver_state", count)

    if tuple(artifacts.pointmap_scale.shape) != (count, 3):
        raise _schema_error(
            "pointmap_scale",
            f"expected shape ({count}, 3), found {tuple(artifacts.pointmap_scale.shape)}",
        )
    if tuple(artifacts.pointmap_shift.shape) != (count, 3):
        raise _schema_error(
            "pointmap_shift",
            f"expected shape ({count}, 3), found {tuple(artifacts.pointmap_shift.shape)}",
        )
    image_condition = artifacts.image_condition_per_object
    feature_dim = artifacts.pose_latents_raw[POSE_TOKEN_KEYS[0]].shape[-1]
    if (
        image_condition.ndim != 3
        or image_condition.shape[0] != count
        or image_condition.shape[1] <= 0
        or image_condition.shape[2] != feature_dim
    ):
        raise _schema_error(
            "image_condition_per_object",
            f"expected shape ({count}, C, {feature_dim}) with C>0, "
            f"found {tuple(image_condition.shape)}",
        )
    for name, poses in (
        ("base_poses", artifacts.embedded_base_poses),
        ("target_poses", artifacts.embedded_target_poses),
    ):
        if poses is not None and tuple(poses.shape) != (count, 10):
            raise _schema_error(name, f"expected shape ({count}, 10), found {tuple(poses.shape)}")


def _legacy_item(path: Path, *, allow_legacy_pickle: bool) -> Dict[str, Any]:
    if not allow_legacy_pickle:
        raise _schema_error(
            path,
            "legacy obj_*.npz caches contain pickle-backed object arrays; only load "
            "a trusted cache and explicitly set allow_legacy_pickle=True (CLI: "
            "--allow-legacy-pickle)",
        )
    try:
        archive = np.load(path, allow_pickle=allow_legacy_pickle)
    except Exception as exc:  # pragma: no cover - exact NumPy exception is format dependent
        raise _schema_error(path, f"could not read NPZ: {exc}") from exc
    with archive:
        required = (
            "pose_latents_raw",
            "pointmap_scale",
            "pointmap_shift",
            "solver_state",
            "x_t_last_step",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise _schema_error(path, f"missing legacy NPZ keys: {missing}")

        raw_mapping = _require_mapping(archive["pose_latents_raw"], f"{path}:pose_latents_raw")
        raw = {
            str(key): _as_tensor(value, f"{path}:pose_latents_raw.{key}")
            for key, value in raw_mapping.items()
        }
        _validate_raw_latents(raw, f"{path}:pose_latents_raw", 1)

        x_t_mapping = _require_mapping(archive["x_t_last_step"], f"{path}:x_t_last_step")
        x_t = {
            str(key): _as_tensor(value, f"{path}:x_t_last_step.{key}")
            for key, value in x_t_mapping.items()
        }
        _validate_x_t(x_t, f"{path}:x_t_last_step", 1, raw["shape"].shape[1])

        solver = _tensor_tree(archive["solver_state"], f"{path}:solver_state")
        if not isinstance(solver, Mapping):
            raise _schema_error(f"{path}:solver_state", "expected mapping")
        _validate_solver_state((solver,), f"{path}:solver_state", 1)

        scale = _as_tensor(archive["pointmap_scale"], f"{path}:pointmap_scale").reshape(-1)
        if scale.numel() != 3:
            raise _schema_error(f"{path}:pointmap_scale", "expected exactly three values")
        shift = _as_tensor(archive["pointmap_shift"], f"{path}:pointmap_shift").reshape(-1)
        if shift.numel() != 3:
            raise _schema_error(f"{path}:pointmap_shift", "expected exactly three values")

        inline_condition = None
        for key in ("image_condition_per_object", "image_condition", "img_cond"):
            if key in archive:
                inline_condition = _as_tensor(archive[key], f"{path}:{key}")
                break
    return {
        "raw": raw,
        "scale": scale,
        "shift": shift,
        "solver": dict(solver),
        "x_t": x_t,
        "condition": inline_condition,
    }


def _load_legacy_directory(
    tokens_dir: Path,
    *,
    allow_legacy_pickle: bool,
) -> _LoadedArtifacts:
    if not tokens_dir.is_dir():
        raise _schema_error(tokens_dir, "legacy tokens_path must be a directory")
    token_files = list(tokens_dir.glob("obj_*.npz"))
    if not token_files:
        raise _schema_error(tokens_dir, "contains no obj_<numeric-id>.npz files")
    if not allow_legacy_pickle:
        raise _schema_error(
            tokens_dir,
            "legacy obj_*.npz caches contain pickle-backed object arrays and are "
            "disabled by default; only load a trusted cache and explicitly set "
            "allow_legacy_pickle=True (CLI: --allow-legacy-pickle)",
        )

    indexed: Dict[int, Path] = {}
    for path in token_files:
        object_id = _numeric_object_id(path)
        if object_id in indexed:
            raise _schema_error(
                tokens_dir,
                f"duplicate numeric id {object_id}: {indexed[object_id].name} and {path.name}",
            )
        indexed[object_id] = path

    object_ids = tuple(sorted(indexed))
    loaded = []
    conditions = []
    for object_id in object_ids:
        path = indexed[object_id]
        item = _legacy_item(path, allow_legacy_pickle=allow_legacy_pickle)
        condition = item["condition"]
        if condition is None:
            condition_path = tokens_dir / f"img_cond_{object_id}.npy"
            padded_candidates = sorted(tokens_dir.glob(f"img_cond_*{object_id}.npy"))
            exact_numeric_candidates = [
                candidate
                for candidate in padded_candidates
                if _numeric_object_id(candidate.stem.replace("img_cond_", "obj_")) == object_id
            ]
            if condition_path.exists():
                exact_numeric_candidates.append(condition_path)
            exact_numeric_candidates = list(dict.fromkeys(exact_numeric_candidates))
            if len(exact_numeric_candidates) != 1:
                raise _schema_error(
                    tokens_dir,
                    f"expected one img_cond_<id>.npy for object {object_id}, "
                    f"found {[candidate.name for candidate in exact_numeric_candidates]}",
                )
            condition = _as_tensor(
                np.load(exact_numeric_candidates[0], allow_pickle=False),
                str(exact_numeric_candidates[0]),
            )
        if condition.ndim == 2:
            condition = condition.unsqueeze(0)
        loaded.append(item)
        conditions.append(condition)

    raw = {key: torch.cat([item["raw"][key] for item in loaded], dim=0) for key in POSE_LATENT_KEYS}
    x_t = {key: torch.cat([item["x_t"][key] for item in loaded], dim=0) for key in POSE_LATENT_KEYS}
    artifacts = _LoadedArtifacts(
        object_ids=object_ids,
        pose_latents_raw=raw,
        pointmap_scale=torch.stack([item["scale"] for item in loaded], dim=0),
        pointmap_shift=torch.stack([item["shift"] for item in loaded], dim=0),
        solver_state=tuple(item["solver"] for item in loaded),
        x_t_last_step=x_t,
        image_condition_per_object=torch.cat(conditions, dim=0),
        source_path=tokens_dir,
    )
    _validate_loaded(artifacts)
    return artifacts


def _portable_key(archive: Mapping[str, Any], prefix: str, key: str) -> Optional[str]:
    for separator in _PORTABLE_PREFIX_SEPARATORS:
        candidate = f"{prefix}{separator}{key}"
        if candidate in archive:
            return candidate
    return None


def _portable_mapping(
    archive: Mapping[str, Any], prefix: str, required_keys: Sequence[str], path: Path
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for key in required_keys:
        archive_key = _portable_key(archive, prefix, key)
        if archive_key is None:
            raise _schema_error(path, f"missing portable key {prefix}__{key}")
        result[key] = _as_tensor(archive[archive_key], f"{path}:{archive_key}")
    return result


def _portable_optional(archive: Mapping[str, Any], names: Sequence[str]) -> Optional[Any]:
    for name in names:
        if name in archive:
            return archive[name]
    return None


def _load_portable_scene(path: Path) -> _LoadedArtifacts:
    if not path.is_file():
        raise _schema_error(path, "portable scene cache must be a file")
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - exact NumPy exception is format dependent
        raise _schema_error(path, f"could not read portable NPZ without pickle: {exc}") from exc
    with archive:
        if "object_ids" not in archive:
            raise _schema_error(path, "missing portable key object_ids")
        ids_array = np.asarray(archive["object_ids"])
        if ids_array.ndim != 1 or ids_array.size == 0:
            raise _schema_error(path, "object_ids must be a non-empty one-dimensional array")
        if not np.issubdtype(ids_array.dtype, np.integer):
            raise _schema_error(
                path, f"object_ids must use an integer dtype, found {ids_array.dtype}"
            )
        ids_in_file = tuple(int(item) for item in ids_array.tolist())
        if len(set(ids_in_file)) != len(ids_in_file) or any(item < 0 for item in ids_in_file):
            raise _schema_error(path, "object_ids must be unique non-negative integers")
        order = tuple(sorted(range(len(ids_in_file)), key=lambda index: ids_in_file[index]))
        object_ids = tuple(ids_in_file[index] for index in order)
        order_tensor = torch.tensor(order, dtype=torch.long)
        count = len(object_ids)

        raw = _portable_mapping(archive, "pose_latents_raw", POSE_LATENT_KEYS, path)
        x_t = _portable_mapping(archive, "x_t_last_step", POSE_LATENT_KEYS, path)
        for mapping_name, mapping in (("pose_latents_raw", raw), ("x_t_last_step", x_t)):
            for key, tensor in mapping.items():
                if tensor.ndim == 0 or tensor.shape[0] != count:
                    raise _schema_error(
                        f"{path}:{mapping_name}__{key}",
                        f"first dimension must equal object count {count}, found {tuple(tensor.shape)}",
                    )
                mapping[key] = tensor.index_select(0, order_tensor)

        if "pointmap_scale" not in archive or "pointmap_shift" not in archive:
            raise _schema_error(path, "portable cache requires pointmap_scale and pointmap_shift")
        pointmap_scale = _as_tensor(archive["pointmap_scale"], f"{path}:pointmap_scale")
        if tuple(pointmap_scale.shape) != (count, 3):
            raise _schema_error(path, f"pointmap_scale must have shape ({count}, 3)")
        pointmap_scale = pointmap_scale.index_select(0, order_tensor)
        pointmap_shift = _as_tensor(archive["pointmap_shift"], f"{path}:pointmap_shift")
        if tuple(pointmap_shift.shape) != (count, 3):
            raise _schema_error(path, f"pointmap_shift must have shape ({count}, 3)")
        pointmap_shift = pointmap_shift.index_select(0, order_tensor)

        condition_value = _portable_optional(
            archive, ("image_condition_per_object", "image_condition", "img_cond")
        )
        if condition_value is None:
            raise _schema_error(path, "portable cache requires image_condition_per_object")
        image_condition = _as_tensor(condition_value, f"{path}:image_condition_per_object")
        if image_condition.ndim == 2:
            image_condition = image_condition.unsqueeze(1)
        if image_condition.shape[0] != count:
            raise _schema_error(path, "image condition first dimension does not match object_ids")
        image_condition = image_condition.index_select(0, order_tensor)

        solver_keys = []
        for archive_key in archive.files:
            for separator in _PORTABLE_PREFIX_SEPARATORS:
                prefix = f"solver_state{separator}"
                if archive_key.startswith(prefix):
                    solver_keys.append((archive_key[len(prefix) :], archive_key))
                    break
        if not solver_keys:
            raise _schema_error(path, "portable cache has no solver_state__* arrays")
        solver_columns: Dict[str, torch.Tensor] = {}
        for logical_key, archive_key in solver_keys:
            if not logical_key:
                raise _schema_error(path, f"invalid empty solver key in {archive_key}")
            tensor = _as_tensor(archive[archive_key], f"{path}:{archive_key}")
            if tensor.ndim == 0 or tensor.shape[0] != count:
                raise _schema_error(
                    path, f"{archive_key} first dimension must equal object count {count}"
                )
            solver_columns[logical_key] = tensor.index_select(0, order_tensor)
        solver_state = tuple(
            {key: value[index] for key, value in solver_columns.items()} for index in range(count)
        )

        embedded_base_value = _portable_optional(archive, ("base_poses", "sam3d_poses"))
        embedded_target_value = _portable_optional(archive, ("target_poses", "gt_poses"))
        embedded_base = None
        if embedded_base_value is not None:
            embedded_base_in_file = _as_tensor(embedded_base_value, f"{path}:base_poses")
            if tuple(embedded_base_in_file.shape) != (count, 10):
                raise _schema_error(path, f"base_poses must have shape ({count}, 10)")
            embedded_base = embedded_base_in_file.index_select(0, order_tensor)
        embedded_target = None
        if embedded_target_value is not None:
            embedded_target_in_file = _as_tensor(embedded_target_value, f"{path}:target_poses")
            if tuple(embedded_target_in_file.shape) != (count, 10):
                raise _schema_error(path, f"target_poses must have shape ({count}, 10)")
            embedded_target = embedded_target_in_file.index_select(0, order_tensor)

    artifacts = _LoadedArtifacts(
        object_ids=object_ids,
        pose_latents_raw=raw,
        pointmap_scale=pointmap_scale,
        pointmap_shift=pointmap_shift,
        solver_state=solver_state,
        x_t_last_step=x_t,
        image_condition_per_object=image_condition,
        source_path=path,
        embedded_base_poses=embedded_base,
        embedded_target_poses=embedded_target,
    )
    _validate_loaded(artifacts)
    return artifacts


def _pose_mapping(path: Path) -> Dict[int, torch.Tensor]:
    if not path.is_file():
        raise _schema_error(path, "pose JSON file does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _schema_error(path, f"could not parse pose JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("objects"), Mapping):
        raise _schema_error(path, "expected top-level {'objects': {...}} mapping")

    result: Dict[int, torch.Tensor] = {}
    names_by_id: Dict[int, str] = {}
    for object_name, pose in payload["objects"].items():
        object_id = _numeric_object_id(str(object_name))
        if object_id in result:
            raise _schema_error(
                path,
                f"duplicate numeric id {object_id}: {names_by_id[object_id]} and {object_name}",
            )
        if not isinstance(pose, Mapping):
            raise _schema_error(path, f"{object_name} pose must be a mapping")
        missing = [key for key in ("translation", "rotation", "scale") if key not in pose]
        if missing:
            raise _schema_error(path, f"{object_name} pose missing keys: {missing}")
        translation = (
            _as_tensor(pose["translation"], f"{path}:{object_name}.translation").reshape(-1).float()
        )
        rotation = (
            _as_tensor(pose["rotation"], f"{path}:{object_name}.rotation").reshape(-1).float()
        )
        scale = _as_tensor(pose["scale"], f"{path}:{object_name}.scale").reshape(-1).float()
        if translation.numel() != 3 or rotation.numel() != 4 or scale.numel() != 3:
            raise _schema_error(
                path,
                f"{object_name} requires translation[3], rotation quaternion wxyz[4], scale[3]",
            )
        if float(torch.linalg.vector_norm(rotation)) <= 1.0e-8:
            raise _schema_error(path, f"{object_name} rotation quaternion has zero norm")
        if not bool((scale > 0).all()):
            raise _schema_error(path, f"{object_name} scale values must be positive")
        result[object_id] = torch.cat((translation, rotation, scale), dim=0).float()
        names_by_id[object_id] = str(object_name)
    if not result:
        raise _schema_error(path, "pose JSON contains no objects")
    return result


def _aligned_pose_tensor(path: Path, object_ids: Tuple[int, ...]) -> torch.Tensor:
    poses = _pose_mapping(path)
    expected = set(object_ids)
    actual = set(poses)
    if actual != expected:
        raise _schema_error(
            path,
            f"numeric object ids do not match cache; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}",
        )
    return torch.stack([poses[object_id] for object_id in object_ids], dim=0)


def _record_artifact_path(record: Mapping[str, Any], manifest_dir: Path) -> Path:
    candidates = []
    for field in ("scene_npz_path", "cache_path", "tokens_path"):
        value = record.get(field)
        if value is not None and str(value).strip():
            candidates.append((field, _resolve_record_path(value, manifest_dir, field)))
    if not candidates:
        raise _schema_error("record", "requires one of scene_npz_path, cache_path, or tokens_path")
    unique_paths = {path for _, path in candidates}
    if len(unique_paths) != 1:
        raise _schema_error(
            "record",
            f"conflicting cache paths: {[(field, str(path)) for field, path in candidates]}",
        )
    return candidates[0][1]


def _record_artifact_paths(record: Mapping[str, Any], manifest_dir: Path) -> Tuple[Path, ...]:
    """Resolve a single cache or object shards belonging to one complete scene."""
    shards = record.get("scene_npz_paths")
    if shards is None:
        return (_record_artifact_path(record, manifest_dir),)
    if not isinstance(shards, list) or not shards:
        raise _schema_error("scene_npz_paths", "expected a non-empty list of NPZ paths")
    if any(
        record.get(field) is not None for field in ("scene_npz_path", "cache_path", "tokens_path")
    ):
        raise _schema_error("record", "scene_npz_paths conflicts with a single cache path")
    paths = tuple(_resolve_record_path(value, manifest_dir, "scene_npz_paths") for value in shards)
    if len(set(paths)) != len(paths):
        raise _schema_error("scene_npz_paths", "duplicate shard paths")
    return paths


def _load_portable_shards(paths: Tuple[Path, ...]) -> _LoadedArtifacts:
    """Join lossless object shards, checking their schema before numeric ID sorting."""
    signatures = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as archive:
                signature = {}
                for key in archive.files:
                    value = archive[key]
                    if value.dtype.hasobject or value.ndim == 0:
                        raise _schema_error(
                            path, f"shard field {key} must be a numeric object array"
                        )
                    signature[key] = (value.dtype.str, tuple(value.shape[1:]))
                signatures.append(signature)
        except ArtifactSchemaError:
            raise
        except Exception as exc:
            raise _schema_error(
                path, f"could not read portable shard without pickle: {exc}"
            ) from exc
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise _schema_error(
            paths[0], "shards must have identical fields, dtypes and non-object dimensions"
        )
    shards = [_load_portable_scene(path) for path in paths]
    ids = tuple(object_id for shard in shards for object_id in shard.object_ids)
    if len(set(ids)) != len(ids):
        raise _schema_error(paths[0], "duplicate object IDs across scene shards")
    order = tuple(sorted(range(len(ids)), key=lambda index: ids[index]))
    index = torch.tensor(order, dtype=torch.long)

    def joined(values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tuple(values), dim=0).index_select(0, index)

    def optional(name: str) -> Optional[torch.Tensor]:
        values = [getattr(shard, name) for shard in shards]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise _schema_error(paths[0], f"inconsistent embedded {name} across shards")
        return joined(values)

    states = tuple(state for shard in shards for state in shard.solver_state)
    artifacts = _LoadedArtifacts(
        object_ids=tuple(ids[position] for position in order),
        pose_latents_raw={
            key: joined([shard.pose_latents_raw[key] for shard in shards])
            for key in POSE_LATENT_KEYS
        },
        pointmap_scale=joined([shard.pointmap_scale for shard in shards]),
        pointmap_shift=joined([shard.pointmap_shift for shard in shards]),
        solver_state=tuple(states[position] for position in order),
        x_t_last_step={
            key: joined([shard.x_t_last_step[key] for shard in shards]) for key in POSE_LATENT_KEYS
        },
        image_condition_per_object=joined([shard.image_condition_per_object for shard in shards]),
        source_path=paths[0],
        embedded_base_poses=optional("embedded_base_poses"),
        embedded_target_poses=optional("embedded_target_poses"),
    )
    _validate_loaded(artifacts)
    return artifacts


def _explicit_pose_path(
    record: Mapping[str, Any], manifest_dir: Path, fields: Sequence[str]
) -> Optional[Path]:
    candidates = []
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            candidates.append((field, _resolve_record_path(value, manifest_dir, field)))
    unique_paths = {path for _, path in candidates}
    if len(unique_paths) > 1:
        raise _schema_error(
            "record",
            f"conflicting pose paths: {[(field, str(path)) for field, path in candidates]}",
        )
    return candidates[0][1] if candidates else None


def load_manifest(path: PathLike) -> list[dict]:
    """Load and structurally validate a JSON scene manifest.

    Record paths remain unchanged; :func:`load_scene` resolves them relative to
    this manifest's directory.
    """

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise _schema_error(manifest_path, "manifest does not exist")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _schema_error(manifest_path, f"could not parse JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise _schema_error(manifest_path, "top level must be a list of scene records")
    records = []
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise _schema_error(manifest_path, f"record {index} must be a mapping")
        scene_id = record.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise _schema_error(manifest_path, f"record {index} requires non-empty scene_id")
        records.append(dict(record))
    if not records:
        raise _schema_error(manifest_path, "manifest contains no records")
    return records


def load_scene(
    record: Mapping[str, Any],
    manifest_dir: PathLike,
    require_target: bool = True,
    *,
    allow_legacy_pickle: bool = False,
) -> SceneSample:
    """Load one checked scene from a manifest record.

    Args:
        record: One item returned by :func:`load_manifest`.
        manifest_dir: Directory containing the manifest (a manifest path is also
            accepted for convenience).
        require_target: Require GT/target poses. Set false for inference-only
            records.
        allow_legacy_pickle: Explicitly trust and load legacy ``obj_*.npz``
            object arrays. Keep false for untrusted data and portable caches.
    """

    if not isinstance(record, Mapping):
        raise _schema_error("record", "expected a mapping")
    scene_id = record.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise _schema_error("record", "requires a non-empty scene_id")
    base_dir = _manifest_base(manifest_dir)
    artifact_paths = _record_artifact_paths(record, base_dir)
    artifact_path = artifact_paths[0]
    if record.get("scene_npz_paths") is not None:
        artifacts = _load_portable_shards(artifact_paths)
    else:
        artifacts = (
            _load_legacy_directory(artifact_path, allow_legacy_pickle=allow_legacy_pickle)
            if artifact_path.is_dir()
            else _load_portable_scene(artifact_path)
        )

    base_path = _explicit_pose_path(record, base_dir, ("base_poses_path", "sam3d_pose_path"))
    if base_path is not None:
        base_poses = _aligned_pose_tensor(base_path, artifacts.object_ids)
    elif artifacts.embedded_base_poses is not None:
        base_poses = artifacts.embedded_base_poses
    else:
        fallback = (
            artifact_path.parent / "pose_inference.json"
            if artifact_path.is_dir()
            else artifact_path.parent / "pose_inference.json"
        )
        if not fallback.is_file():
            raise _schema_error(
                "record",
                "base poses are required; set base_poses_path/sam3d_pose_path, embed "
                f"base_poses in a portable cache, or provide {fallback}",
            )
        base_poses = _aligned_pose_tensor(fallback, artifacts.object_ids)

    target_path = _explicit_pose_path(record, base_dir, ("target_poses_path", "gt_poses_path"))
    if target_path is not None:
        target_poses: Optional[torch.Tensor] = _aligned_pose_tensor(
            target_path, artifacts.object_ids
        )
    else:
        target_poses = artifacts.embedded_target_poses
    if require_target and target_poses is None:
        raise _schema_error(
            "record", "target poses are required; set gt_poses_path/target_poses_path"
        )

    count = len(artifacts.object_ids)
    if tuple(base_poses.shape) != (count, 10):
        raise _schema_error("base_poses", f"expected shape ({count}, 10)")
    if target_poses is not None and tuple(target_poses.shape) != (count, 10):
        raise _schema_error("target_poses", f"expected shape ({count}, 10)")

    pose_tokens = torch.cat([artifacts.pose_latents_raw[key] for key in POSE_TOKEN_KEYS], dim=1)
    shape_tokens = artifacts.pose_latents_raw["shape"]
    decode_state = DecodeState(
        object_ids=artifacts.object_ids,
        pose_latents_raw=dict(artifacts.pose_latents_raw),
        pointmap_scale=artifacts.pointmap_scale,
        pointmap_shift=artifacts.pointmap_shift,
        solver_state=tuple(dict(item) for item in artifacts.solver_state),
        x_t_last_step=dict(artifacts.x_t_last_step),
        image_condition_per_object=artifacts.image_condition_per_object,
        source_path=artifacts.source_path,
    )
    resolved_record = dict(record)
    if record.get("scene_npz_paths") is not None:
        resolved_record["scene_npz_paths"] = [str(path) for path in artifact_paths]
    for field in (
        "tokens_path",
        "scene_npz_path",
        "cache_path",
        "sam3d_canonical_dir",
        "base_poses_path",
        "sam3d_pose_path",
        "target_poses_path",
        "gt_poses_path",
        "image_path",
        "instance_mask_path",
        "mesh_path",
    ):
        value = resolved_record.get(field)
        if value is not None and str(value).strip():
            resolved_record[field] = str(_resolve_record_path(value, base_dir, field))

    return SceneSample(
        scene_id=scene_id,
        object_names=tuple(f"obj_{object_id}" for object_id in artifacts.object_ids),
        pose_tokens=pose_tokens,
        shape_tokens=shape_tokens,
        base_poses=base_poses.float(),
        target_poses=None if target_poses is None else target_poses.float(),
        decode_state=decode_state,
        record=resolved_record,
    )


class SceneDataset(Dataset[SceneSample]):
    """Lazy strict dataset over a MOD JSON manifest."""

    def __init__(
        self,
        manifest: Union[PathLike, Sequence[Mapping[str, Any]]],
        require_target: bool = True,
        *,
        manifest_dir: Optional[PathLike] = None,
        allow_legacy_pickle: bool = False,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest).expanduser().resolve()
            self.records = load_manifest(manifest_path)
            self.manifest_dir = manifest_path.parent
        else:
            if manifest_dir is None:
                raise ValueError("manifest_dir is required when manifest is an in-memory sequence")
            self.records = [dict(record) for record in manifest]
            if not self.records:
                raise ArtifactSchemaError("manifest contains no records")
            self.manifest_dir = _manifest_base(manifest_dir)
        self.require_target = bool(require_target)
        self.allow_legacy_pickle = bool(allow_legacy_pickle)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SceneSample:
        return load_scene(
            self.records[index],
            self.manifest_dir,
            require_target=self.require_target,
            allow_legacy_pickle=self.allow_legacy_pickle,
        )

    def __iter__(self) -> Iterator[SceneSample]:
        for index in range(len(self)):
            yield self[index]


__all__ = [
    "ArtifactSchemaError",
    "DecodeState",
    "POSE_LATENT_KEYS",
    "POSE_TOKEN_KEYS",
    "SceneDataset",
    "SceneSample",
    "load_manifest",
    "load_scene",
]
