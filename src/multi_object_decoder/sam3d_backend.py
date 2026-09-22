"""Explicit, fail-fast bridge to the external SAM3D pose decoder.

The release package must remain importable without SAM3D.  Consequently all
SAM3D, Hydra and OmegaConf imports live inside :func:`load_sam3d_pipeline`.
The functions in this module intentionally use a small, checked subset of the
legacy pipeline API needed by camera-ready MOD checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
from torch import nn

if TYPE_CHECKING:  # Avoid importing the data module (or SAM3D) at runtime.
    from .data import SceneSample


class SAM3DBackendError(RuntimeError):
    """Base exception for an unavailable or incompatible SAM3D backend."""


class SAM3DUnavailableError(SAM3DBackendError):
    """Raised when the explicitly selected SAM3D installation cannot load."""


class SAM3DConfigurationError(SAM3DBackendError):
    """Raised when SAM3D paths or its runtime object graph are incompatible."""


class SAM3DDecodeError(SAM3DBackendError):
    """Raised when a legacy raw-cache object cannot be decoded."""


_MISSING = object()


def _member(value: Any, name: str, default: Any = _MISSING) -> Any:
    """Read a field from either a dataclass-like object or a mapping."""
    if isinstance(value, Mapping) and name in value:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise SAM3DDecodeError(f"Legacy decode state is missing required field {name!r}")


def _resolved_file(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SAM3DConfigurationError(f"{label} is not a file: {resolved}")
    return resolved


def _resolved_root(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise SAM3DConfigurationError(f"SAM3D_ROOT is not a directory: {resolved}")
    if not (resolved / "sam3d_objects").is_dir():
        raise SAM3DConfigurationError(
            f"SAM3D_ROOT does not contain the sam3d_objects package: {resolved}"
        )
    return resolved


def _ensure_selected_sam3d_import(sam3d_root: Path) -> None:
    """Import SAM3D from ``sam3d_root`` and reject an already-loaded mismatch."""
    existing = sys.modules.get("sam3d_objects")
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        if module_file is None:
            raise SAM3DConfigurationError(
                "An unlocatable sam3d_objects module is already loaded; start a clean process"
            )
        loaded_path = Path(module_file).resolve()
        if not loaded_path.is_relative_to(sam3d_root):
            raise SAM3DConfigurationError(
                "sam3d_objects is already loaded from a different checkout: "
                f"{loaded_path} (requested SAM3D_ROOT={sam3d_root})"
            )

    root_string = str(sam3d_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        imported = importlib.import_module("sam3d_objects")
    except Exception as exc:  # SAM3D can fail through one of many compiled deps.
        raise SAM3DUnavailableError(
            f"Failed to import sam3d_objects from SAM3D_ROOT={sam3d_root}: {exc}"
        ) from exc

    module_file = getattr(imported, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(sam3d_root):
        raise SAM3DConfigurationError(
            f"Imported sam3d_objects does not belong to SAM3D_ROOT={sam3d_root}"
        )


def load_sam3d_pipeline(
    sam3d_root: Path | str,
    config_path: Path | str,
    *,
    compile_model: bool = False,
    rendering_engine: str = "pytorch3d",
) -> Any:
    """Instantiate SAM3D from explicit source and pipeline-config paths.

    No repository-relative or environment-variable fallback is used.  This is
    deliberate: callers must record the exact checkout and checkpoint config
    that created a MOD checkpoint.
    """
    root = _resolved_root(sam3d_root)
    config_file = _resolved_file(config_path, label="SAM3D pipeline config")
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    _ensure_selected_sam3d_import(root)

    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except Exception as exc:
        raise SAM3DUnavailableError(
            "Loading SAM3D requires hydra-core and omegaconf"
        ) from exc

    try:
        config = OmegaConf.load(str(config_file))
        OmegaConf.update(config, "rendering_engine", rendering_engine, force_add=True)
        OmegaConf.update(config, "compile_model", bool(compile_model), force_add=True)
        OmegaConf.update(config, "workspace_dir", str(config_file.parent), force_add=True)
        pipeline = instantiate(config)
    except Exception as exc:
        raise SAM3DConfigurationError(
            f"Failed to instantiate SAM3D pipeline from {config_file}: {exc}"
        ) from exc

    # Preserve tensor leaves in upstream's dataclass conversion. Its default
    # deepcopy rejects non-leaf tensors during differentiable MOD training.
    # This changes serialization only; the upstream pose calculation is kept.
    if type(pipeline).__module__.startswith("sam3d_objects.pipeline."):
        from .sam3d_capture import tensor_preserving_asdict

        pose_target_module = importlib.import_module(
            "sam3d_objects.data.dataset.tdfy.pose_target"
        )
        pose_target_module.asdict = tensor_preserving_asdict

    # Validate the private API once at construction instead of failing halfway
    # through a training iteration.
    get_sam3d_backbone(pipeline)
    if not callable(getattr(pipeline, "pose_decoder", None)):
        raise SAM3DConfigurationError("SAM3D pipeline has no callable pose_decoder")
    return pipeline


def _unwrap_pipeline(pipeline: Any) -> Any:
    """Accept either the raw pipeline or SAM3D notebook's Inference wrapper."""
    return getattr(pipeline, "_pipeline", pipeline)


def _pipeline_model(pipeline: Any, name: str) -> Any:
    pipeline = _unwrap_pipeline(pipeline)
    models = getattr(pipeline, "models", None)
    if models is None:
        raise SAM3DConfigurationError("SAM3D pipeline has no models collection")
    try:
        model = models[name]
    except (KeyError, TypeError) as exc:
        raise SAM3DConfigurationError(f"SAM3D pipeline has no {name!r} model") from exc
    if model is None:
        raise SAM3DConfigurationError(f"SAM3D pipeline model {name!r} is None")
    return model


def get_sam3d_backbone(pipeline: Any) -> Any:
    """Return the sparse-structure backbone required by legacy MOD decoding."""
    generator = _pipeline_model(pipeline, "ss_generator")
    reverse_fn = getattr(generator, "reverse_fn", None)
    backbone = getattr(reverse_fn, "backbone", None)
    if backbone is None:
        raise SAM3DConfigurationError(
            "SAM3D ss_generator.reverse_fn.backbone is unavailable"
        )
    latent_mapping = getattr(backbone, "latent_mapping", None)
    if latent_mapping is None or len(latent_mapping) == 0:
        raise SAM3DConfigurationError("SAM3D backbone has no latent_mapping")
    return backbone


def _refiner_with_latent_mapping(model: nn.Module) -> nn.Module:
    """Resolve a raw legacy refiner through common training wrappers."""
    candidate: nn.Module = model
    if hasattr(candidate, "module") and isinstance(candidate.module, nn.Module):
        candidate = candidate.module
    if callable(getattr(candidate, "load_latent_mapping_from_backbone", None)):
        return candidate
    nested = getattr(candidate, "refiner", None)
    if isinstance(nested, nn.Module) and callable(
        getattr(nested, "load_latent_mapping_from_backbone", None)
    ):
        return nested
    raise SAM3DConfigurationError(
        "Refiner does not implement load_latent_mapping_from_backbone(backbone)"
    )


def initialize_refiner_latent_mapping(model: nn.Module, pipeline: Any) -> nn.Module:
    """Attach SAM3D latent heads before optimizer creation or checkpoint load.

    Legacy ``model.pth`` files contain parameters for dynamically-created
    ``latent_mapping`` and ``_adapter`` modules.  Therefore this function must
    run before a strict ``load_state_dict`` call.
    """
    target = _refiner_with_latent_mapping(model)
    target.load_latent_mapping_from_backbone(get_sam3d_backbone(pipeline))
    keys = list(getattr(target, "_pose_latent_keys", ()))
    adapters = getattr(target, "_adapter", None)
    if not keys or adapters is None or len(adapters) == 0:
        raise SAM3DConfigurationError(
            "Refiner latent mapping initialization produced no pose residual heads"
        )
    return target


def _object_count(sample: Any, pose_latents_raw: Mapping[str, Any]) -> int:
    object_names = _member(sample, "object_names", None)
    if object_names is not None:
        count = len(object_names)
        if count <= 0:
            raise SAM3DDecodeError("SceneSample.object_names must not be empty")
        return count
    for value in pose_latents_raw.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] <= 0:
                break
            return int(value.shape[0])
    raise SAM3DDecodeError("Cannot determine object count from SceneSample")


def _slice_tensor(value: Any, index: int, count: int, *, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SAM3DDecodeError(f"{field} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim == 0 or value.shape[0] != count:
        raise SAM3DDecodeError(
            f"{field} must have leading object dimension {count}, got {tuple(value.shape)}"
        )
    return value[index : index + 1]


def _slice_tree(value: Any, index: int, count: int, *, field: str) -> Any:
    if isinstance(value, torch.Tensor):
        return _slice_tensor(value, index, count, field=field)
    if isinstance(value, Mapping):
        return {
            key: _slice_tree(item, index, count, field=f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != count:
            raise SAM3DDecodeError(
                f"{field} must contain {count} per-object entries, got {len(value)}"
            )
        return value[index]
    # Scalar solver metadata (for example t/dt) is shared by all objects.
    return value


def _per_object_state(value: Any, index: int, count: int) -> Mapping[str, Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != count:
            raise SAM3DDecodeError(
                f"solver_state must contain {count} entries, got {len(value)}"
            )
        state = value[index]
    elif isinstance(value, Mapping):
        state = {
            key: _slice_tree(item, index, count, field=f"solver_state.{key}")
            if isinstance(item, (torch.Tensor, Mapping, Sequence))
            and not isinstance(item, (str, bytes))
            else item
            for key, item in value.items()
        }
    else:
        raise SAM3DDecodeError(
            f"solver_state must be a mapping or sequence, got {type(value).__name__}"
        )
    if not isinstance(state, Mapping):
        raise SAM3DDecodeError(
            f"solver_state[{index}] must be a mapping, got {type(state).__name__}"
        )
    return state


def _latent_to_output(latent_module: Any, latent: torch.Tensor, *, key: str) -> torch.Tensor:
    to_output = getattr(latent_module, "to_output", None)
    if not callable(to_output):
        raise SAM3DConfigurationError(f"SAM3D latent_mapping[{key!r}] has no to_output")
    out_layer = getattr(latent_module, "out_layer", None)
    weight = getattr(out_layer, "weight", None)
    if isinstance(weight, torch.Tensor):
        latent = latent.to(device=weight.device, dtype=weight.dtype)
    output = to_output(latent)
    if not isinstance(output, torch.Tensor):
        raise SAM3DDecodeError(f"Decoded latent {key!r} is not a tensor")
    return output


def _compatible_dt(dt: Any, reference: torch.Tensor, *, object_index: int) -> Any:
    if isinstance(dt, torch.Tensor):
        if dt.numel() != 1 or not bool(torch.isfinite(dt).all()):
            raise SAM3DDecodeError(
                f"Object {object_index}: solver_state.dt must be one finite value"
            )
        return dt.to(device=reference.device, dtype=reference.dtype)
    try:
        scalar = float(dt)
    except (TypeError, ValueError) as exc:
        raise SAM3DDecodeError(
            f"Object {object_index}: invalid solver_state.dt={dt!r}"
        ) from exc
    if not torch.isfinite(torch.tensor(scalar)):
        raise SAM3DDecodeError(f"Object {object_index}: solver_state.dt is not finite")
    return scalar


def _compatible_alpha(
    alpha: float | torch.Tensor, reference: torch.Tensor, *, object_index: int
) -> float | torch.Tensor:
    if isinstance(alpha, torch.Tensor):
        if alpha.numel() != 1 or not bool(torch.isfinite(alpha).all()):
            raise SAM3DDecodeError(
                f"Object {object_index}: delta_alpha must be one finite value"
            )
        return alpha.to(device=reference.device, dtype=reference.dtype)
    try:
        scalar = float(alpha)
    except (TypeError, ValueError) as exc:
        raise SAM3DDecodeError(
            f"Object {object_index}: invalid delta_alpha={alpha!r}"
        ) from exc
    if not bool(torch.isfinite(torch.tensor(scalar))):
        raise SAM3DDecodeError(f"Object {object_index}: delta_alpha is not finite")
    return scalar


def _checked_pose_component(
    pose: Mapping[str, Any], name: str, size: int, *, object_index: int
) -> torch.Tensor:
    value = pose.get(name)
    if not isinstance(value, torch.Tensor):
        raise SAM3DDecodeError(
            f"Object {object_index}: pose_decoder did not return tensor {name!r}"
        )
    flat = value.reshape(-1)
    if flat.numel() != size:
        raise SAM3DDecodeError(
            f"Object {object_index}: pose {name!r} has {flat.numel()} values, expected {size}"
        )
    if not bool(torch.isfinite(flat).all()):
        raise SAM3DDecodeError(f"Object {object_index}: pose {name!r} contains NaN/Inf")
    return flat.reshape(1, size)


def decode_legacy_scene_sample(
    pipeline: Any,
    sample: "SceneSample | Any",
    refined_pose_delta: Mapping[str, torch.Tensor] | None,
    *,
    delta_alpha: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Differentiably decode one ``SceneSample`` legacy raw cache to ``(N, 10)``.

    ``sample.decode_state`` may be either a dataclass or mapping with
    ``pose_latents_raw``, ``solver_state``, ``pointmap_scale``,
    ``pointmap_shift``, ``x_t_last_step`` and optional
    ``image_condition_per_object``.  Unlike the historical trainer, this
    function never substitutes an all-zero pose when one object fails.
    """
    raw_state = _member(sample, "decode_state", None)
    if raw_state is None:
        # Transitional compatibility for records that placed legacy fields on
        # the sample itself.  New data loaders should always use decode_state.
        raw_state = sample

    pose_latents_raw = _member(raw_state, "pose_latents_raw")
    if not isinstance(pose_latents_raw, Mapping) or not pose_latents_raw:
        raise SAM3DDecodeError("pose_latents_raw must be a non-empty mapping")
    solver_states = _member(raw_state, "solver_state")
    pointmap_scale = _member(raw_state, "pointmap_scale")
    pointmap_shift = _member(raw_state, "pointmap_shift")
    x_t_last_step = _member(raw_state, "x_t_last_step")
    image_conditions = _member(raw_state, "image_condition_per_object", None)

    count = _object_count(sample, pose_latents_raw)
    pipeline = _unwrap_pipeline(pipeline)
    backbone = get_sam3d_backbone(pipeline)
    latent_mapping = backbone.latent_mapping
    pose_decoder = getattr(pipeline, "pose_decoder", None)
    if not callable(pose_decoder):
        raise SAM3DConfigurationError("SAM3D pipeline has no callable pose_decoder")

    if refined_pose_delta is not None:
        if not isinstance(refined_pose_delta, Mapping):
            raise SAM3DDecodeError("refined_pose_delta must be a mapping or None")
        for key, value in refined_pose_delta.items():
            _slice_tensor(value, 0, count, field=f"refined_pose_delta.{key}")

    configured_keys = list(getattr(backbone, "input_latent_mappings", ()) or ())
    latent_keys = configured_keys or [key for key in pose_latents_raw if key in latent_mapping]
    if not latent_keys:
        raise SAM3DConfigurationError(
            "No pose_latents_raw keys match the SAM3D backbone latent_mapping"
        )
    missing_raw = [key for key in latent_keys if key not in pose_latents_raw]
    if missing_raw:
        raise SAM3DDecodeError(f"pose_latents_raw is missing latent keys {missing_raw}")

    decoded: list[torch.Tensor] = []
    for object_index in range(count):
        try:
            state_i = _per_object_state(solver_states, object_index, count)
            dt = state_i.get("dt")
            if dt is None:
                raise SAM3DDecodeError(
                    f"Object {object_index}: solver_state is missing required 'dt'"
                )

            pose_raw_i = {
                key: _slice_tensor(
                    pose_latents_raw[key],
                    object_index,
                    count,
                    field=f"pose_latents_raw.{key}",
                )
                for key in latent_keys
            }
            x_t_i = _slice_tree(
                x_t_last_step,
                object_index,
                count,
                field="x_t_last_step",
            )
            if not isinstance(x_t_i, Mapping) or not x_t_i:
                raise SAM3DDecodeError(
                    f"Object {object_index}: x_t_last_step must be a non-empty mapping"
                )

            velocity = {
                key: _latent_to_output(latent_mapping[key], pose_raw_i[key], key=key)
                for key in latent_keys
            }
            x_keys = list(x_t_i)
            if set(x_keys) != set(velocity):
                raise SAM3DDecodeError(
                    f"Object {object_index}: x_t_last_step keys {sorted(x_keys)} do not "
                    f"match decoded velocity keys {sorted(velocity)}"
                )
            x_tp1: dict[str, torch.Tensor] = {}
            for key in x_keys:
                x_t_value = x_t_i[key]
                if not isinstance(x_t_value, torch.Tensor):
                    raise SAM3DDecodeError(
                        f"Object {object_index}: x_t_last_step.{key} is not a tensor"
                    )
                velocity_value = velocity[key]
                x_t_value = x_t_value.to(
                    device=velocity_value.device, dtype=velocity_value.dtype
                )
                x_tp1[key] = x_t_value + velocity_value * _compatible_dt(
                    dt, velocity_value, object_index=object_index
                )

            if refined_pose_delta:
                for key, values in refined_pose_delta.items():
                    if key not in x_tp1:
                        raise SAM3DDecodeError(
                            f"Object {object_index}: refined delta key {key!r} "
                            "is not in solver state"
                        )
                    delta_i = _slice_tensor(
                        values,
                        object_index,
                        count,
                        field=f"refined_pose_delta.{key}",
                    ).to(device=x_tp1[key].device, dtype=x_tp1[key].dtype)
                    alpha = _compatible_alpha(
                        delta_alpha, delta_i, object_index=object_index
                    )
                    x_tp1[key] = x_tp1[key] + delta_i * alpha

            if image_conditions is not None:
                condition_i = _slice_tensor(
                    image_conditions,
                    object_index,
                    count,
                    field="image_condition_per_object",
                )
                condition_reference = next(iter(x_tp1.values()))
                condition_i = condition_i.to(
                    device=condition_reference.device,
                    dtype=condition_reference.dtype,
                )
                backbone.latest_condition_args = (condition_i,)
                backbone.latest_condition_kwargs = {}

            scale_i = _slice_tensor(
                pointmap_scale,
                object_index,
                count,
                field="pointmap_scale",
            )
            shift_i = _slice_tensor(
                pointmap_shift,
                object_index,
                count,
                field="pointmap_shift",
            )
            reference = next(iter(x_tp1.values()))
            scale_i = scale_i.to(device=reference.device, dtype=reference.dtype)
            shift_i = shift_i.to(device=reference.device, dtype=reference.dtype)
            pose = pose_decoder(x_tp1, scene_scale=scale_i, scene_shift=shift_i)
            if not isinstance(pose, Mapping):
                raise SAM3DDecodeError(
                    f"Object {object_index}: pose_decoder returned {type(pose).__name__}"
                )
            decoded.append(
                torch.cat(
                    (
                        _checked_pose_component(
                            pose, "translation", 3, object_index=object_index
                        ),
                        _checked_pose_component(
                            pose, "rotation", 4, object_index=object_index
                        ),
                        _checked_pose_component(pose, "scale", 3, object_index=object_index),
                    ),
                    dim=-1,
                )
            )
        except SAM3DBackendError:
            raise
        except Exception as exc:
            raise SAM3DDecodeError(
                f"Failed to decode object {object_index}/{count - 1}: {exc}"
            ) from exc

    result = torch.cat(decoded, dim=0)
    if result.shape != (count, 10):
        raise SAM3DDecodeError(
            f"Decoded pose tensor has shape {tuple(result.shape)}, expected ({count}, 10)"
        )
    return result


@dataclass(frozen=True)
class SAM3DBackend:
    """Loaded SAM3D runtime with explicit provenance-bearing paths."""

    sam3d_root: Path
    config_path: Path
    pipeline: Any

    @classmethod
    def from_paths(
        cls,
        sam3d_root: Path | str,
        config_path: Path | str,
        *,
        compile_model: bool = False,
        rendering_engine: str = "pytorch3d",
    ) -> "SAM3DBackend":
        root = _resolved_root(sam3d_root)
        config = _resolved_file(config_path, label="SAM3D pipeline config")
        pipeline = load_sam3d_pipeline(
            root,
            config,
            compile_model=compile_model,
            rendering_engine=rendering_engine,
        )
        return cls(sam3d_root=root, config_path=config, pipeline=pipeline)

    @property
    def backbone(self) -> Any:
        return get_sam3d_backbone(self.pipeline)

    def initialize_refiner(self, model: nn.Module) -> nn.Module:
        return initialize_refiner_latent_mapping(model, self.pipeline)

    def initialize_model(self, model: nn.Module) -> None:
        """Initialize a workflow model from the configured SAM3D backbone.

        This is the :class:`multi_object_decoder.workflow.DecoderBackend` compatibility
        entry point.  Initialization intentionally happens before checkpoint
        loading and optimizer construction because the legacy refiner creates
        its latent residual heads dynamically.
        """
        self.initialize_refiner(model)

    def decode_scene(
        self,
        sample: "SceneSample | Any",
        refined_pose_delta: Mapping[str, torch.Tensor] | None,
        *,
        delta_alpha: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        return decode_legacy_scene_sample(
            self.pipeline,
            sample,
            refined_pose_delta,
            delta_alpha=delta_alpha,
        )

    def decode(
        self,
        sample: "SceneSample | Any",
        refined_pose_delta: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Decode through the shared workflow ``DecoderBackend`` protocol.

        ``Tensor.to`` remains differentiable, so moving model-produced
        residuals here preserves the training gradient.  The exact SAM3D
        latent/output dtypes are selected later from the decoder modules.
        """
        target_device = torch.device(device)
        move_sample = getattr(sample, "to", None)
        if callable(move_sample):
            sample = move_sample(device=target_device)
        if not isinstance(refined_pose_delta, Mapping):
            raise SAM3DDecodeError("refined_pose_delta must be a mapping")
        moved_delta = {
            key: value.to(device=target_device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in refined_pose_delta.items()
        }
        return self.decode_scene(sample, moved_delta)


__all__ = [
    "SAM3DBackend",
    "SAM3DBackendError",
    "SAM3DUnavailableError",
    "SAM3DConfigurationError",
    "SAM3DDecodeError",
    "load_sam3d_pipeline",
    "get_sam3d_backbone",
    "initialize_refiner_latent_mapping",
    "decode_legacy_scene_sample",
]
