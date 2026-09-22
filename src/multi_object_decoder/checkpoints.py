"""Versioned, self-describing checkpoints for MOD models.

The public checkpoint schema is intentionally small and backend-neutral.  A
version-1 checkpoint always contains these top-level fields::

    {
        "format": "multi_object_decoder.checkpoint",
        "version": 1,
        "backend": "...",
        "model_config": {...},
        "state_dict": {...},
        "optimizer": {...} | None,
        "training_state": {...},
    }

``load_checkpoint`` also accepts the previous
``"messykitchens_mod.checkpoint"`` and historical ``"mod_clean.checkpoint"``
format tags, plus the ``{"model_state_dict": ...}`` wrapper.  All are
normalized to the current public schema.  A legacy wrapper has version ``0``
and backend ``"legacy"`` unless those values can be recovered from the input
mapping.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT = "multi_object_decoder.checkpoint"
LEGACY_CHECKPOINT_FORMATS = frozenset(
    {
        "messykitchens_mod.checkpoint",
        "mod_clean.checkpoint",
    }
)
CHECKPOINT_VERSION = 1
LEGACY_CHECKPOINT_VERSION = 0


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint does not satisfy a supported schema."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointValidationError(
            f"Checkpoint field {field!r} must be a mapping, got "
            f"{type(value).__name__}."
        )
    non_string_key = next((key for key in value if not isinstance(key, str)), None)
    if non_string_key is not None:
        raise CheckpointValidationError(
            f"Checkpoint field {field!r} must use string keys; found "
            f"{non_string_key!r}."
        )
    return value


def _plain_config_value(value: Any, field: str) -> Any:
    """Copy configuration data into pickle-safe, backend-neutral containers."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise CheckpointValidationError(
                    f"Checkpoint field {field!r} must use string keys; found "
                    f"{key!r}."
                )
            result[key] = _plain_config_value(child, f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _plain_config_value(child, f"{field}[{index}]")
            for index, child in enumerate(value)
        ]
    raise CheckpointValidationError(
        f"Checkpoint field {field!r} contains unsupported value type "
        f"{type(value).__name__}; use strings, numbers, booleans, paths, "
        "lists, or string-keyed mappings."
    )


def _optimizer_state(optimizer: Any) -> Mapping[str, Any] | None:
    if optimizer is None:
        return None
    if isinstance(optimizer, Mapping):
        return _require_mapping(optimizer, "optimizer")
    state_dict_method = getattr(optimizer, "state_dict", None)
    if not callable(state_dict_method):
        raise CheckpointValidationError(
            "Checkpoint field 'optimizer' must be an optimizer with a "
            "state_dict() method, a state mapping, or None."
        )
    return _require_mapping(state_dict_method(), "optimizer")


def build_checkpoint(
    *,
    backend: str,
    model_config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    optimizer: Any = None,
    training_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a version-1 checkpoint payload.

    Args:
        backend: Stable identifier for the model backend/decoder contract.
        model_config: Constructor configuration needed to rebuild the model.
        state_dict: Model state, normally ``model.state_dict()``.
        optimizer: An optimizer, its state mapping, or ``None``.
        training_state: Step/epoch and other resumable training metadata.

    Returns:
        A validated mapping ready for :func:`save_checkpoint` or ``torch.save``.
    """

    if not isinstance(backend, str) or not backend.strip():
        raise CheckpointValidationError(
            "Checkpoint field 'backend' must be a non-empty string."
        )
    config_mapping = _require_mapping(model_config, "model_config")
    model_state = _require_mapping(state_dict, "state_dict")
    optimizer_state = _optimizer_state(optimizer)
    if training_state is None:
        training_state = {}
    training_mapping = _require_mapping(training_state, "training_state")

    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "backend": backend.strip(),
        "model_config": _plain_config_value(config_mapping, "model_config"),
        "state_dict": model_state,
        "optimizer": optimizer_state,
        "training_state": dict(training_mapping),
    }


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync after an atomic replacement."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    backend: str,
    model_config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    optimizer: Any = None,
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a self-describing MOD checkpoint.

    The temporary file is created beside the destination, flushed to disk, and
    replaced with :func:`os.replace`, so readers observe either the previous
    complete checkpoint or the new complete checkpoint.
    """

    destination = Path(path).expanduser()
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Checkpoint destination is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint(
        backend=backend,
        model_config=model_config,
        state_dict=state_dict,
        optimizer=optimizer,
        training_state=training_state,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        # ``fdopen`` owns and closes the descriptor after it succeeds.  The
        # explicit close also covers the rare case where ``fdopen`` itself
        # fails; EBADF simply means the context manager already closed it.
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def _validate_new_checkpoint(raw: Mapping[str, Any]) -> dict[str, Any]:
    required_fields = {
        "format",
        "version",
        "backend",
        "model_config",
        "state_dict",
        "optimizer",
        "training_state",
    }
    missing = sorted(required_fields.difference(raw))
    if missing:
        raise CheckpointValidationError(
            "Checkpoint is missing required field(s): " + ", ".join(missing)
        )
    supported_formats = {CHECKPOINT_FORMAT, *LEGACY_CHECKPOINT_FORMATS}
    if raw["format"] not in supported_formats:
        raise CheckpointValidationError(
            f"Unsupported checkpoint format {raw['format']!r}; expected one of "
            f"{sorted(supported_formats)!r}."
        )
    version = raw["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise CheckpointValidationError(
            f"Checkpoint field 'version' must be an integer, got {version!r}."
        )
    if version != CHECKPOINT_VERSION:
        relation = "newer than" if version > CHECKPOINT_VERSION else "older than"
        raise CheckpointValidationError(
            f"Checkpoint version {version} is {relation} the supported version "
            f"{CHECKPOINT_VERSION}."
        )
    backend = raw["backend"]
    if not isinstance(backend, str) or not backend.strip():
        raise CheckpointValidationError(
            "Checkpoint field 'backend' must be a non-empty string."
        )
    model_config = _require_mapping(raw["model_config"], "model_config")
    state_dict = _require_mapping(raw["state_dict"], "state_dict")
    optimizer = raw["optimizer"]
    if optimizer is not None:
        optimizer = _require_mapping(optimizer, "optimizer")
    training_state = _require_mapping(raw["training_state"], "training_state")
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "backend": backend.strip(),
        "model_config": dict(model_config),
        "state_dict": state_dict,
        "optimizer": optimizer,
        "training_state": dict(training_state),
    }


def _normalize_legacy_checkpoint(raw: Mapping[str, Any]) -> dict[str, Any]:
    state_dict = _require_mapping(raw["model_state_dict"], "model_state_dict")
    backend = raw.get("backend", "legacy")
    if not isinstance(backend, str) or not backend.strip():
        backend = "legacy"

    raw_config = raw.get("model_config", {})
    model_config = _require_mapping(raw_config, "model_config")
    optimizer = raw.get("optimizer", raw.get("optimizer_state_dict"))
    if optimizer is not None:
        optimizer = _require_mapping(optimizer, "optimizer")

    raw_training_state = raw.get("training_state")
    if raw_training_state is None:
        raw_training_state = {
            key: raw[key]
            for key in ("epoch", "global_step", "step")
            if key in raw
        }
    training_state = _require_mapping(raw_training_state, "training_state")
    return {
        "format": CHECKPOINT_FORMAT,
        "version": LEGACY_CHECKPOINT_VERSION,
        "backend": backend.strip(),
        "model_config": dict(model_config),
        "state_dict": state_dict,
        "optimizer": optimizer,
        "training_state": dict(training_state),
    }


def _safe_torch_load(path: Path, map_location: Any) -> Any:
    """Load a checkpoint with PyTorch's restricted weights-only unpickler.

    Loading an untrusted pickle can execute arbitrary code.  Older PyTorch
    versions without ``weights_only`` are rejected instead of silently
    falling back to the unrestricted loader.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as error:
        raise CheckpointValidationError(
            "The installed PyTorch does not support the restricted "
            "weights_only checkpoint loader. Upgrade to PyTorch >= 2.5; "
            f"refusing to load {path} with unsafe pickle deserialization."
        ) from error
    except Exception as error:
        raise CheckpointValidationError(
            f"Could not safely load checkpoint {path}: {error}"
        ) from error


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: Any = "cpu",
    expected_backend: str | None = None,
) -> dict[str, Any]:
    """Load and normalize a current or legacy MOD checkpoint.

    Args:
        path: Checkpoint file to read.
        map_location: Forwarded explicitly to ``torch.load``; defaults to CPU.
        expected_backend: Optional backend identifier to enforce.  This is
            useful before applying a state dict with dynamic backend modules.

    Returns:
        A normalized mapping with the seven public schema keys.  Legacy inputs
        use version ``0`` and otherwise follow the same shape.
    """

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"Checkpoint path is not a file: {source}")

    raw = _safe_torch_load(source, map_location)
    raw_mapping = _require_mapping(raw, "checkpoint")
    if "format" in raw_mapping or "version" in raw_mapping:
        checkpoint = _validate_new_checkpoint(raw_mapping)
    elif "model_state_dict" in raw_mapping:
        checkpoint = _normalize_legacy_checkpoint(raw_mapping)
    else:
        raise CheckpointValidationError(
            "Unrecognized checkpoint: expected the versioned MOD schema or "
            "a legacy mapping containing 'model_state_dict'."
        )

    if expected_backend is not None:
        if not isinstance(expected_backend, str) or not expected_backend.strip():
            raise CheckpointValidationError(
                "expected_backend must be a non-empty string when provided."
            )
        if checkpoint["backend"] != expected_backend.strip():
            raise CheckpointValidationError(
                f"Checkpoint backend {checkpoint['backend']!r} does not match "
                f"expected backend {expected_backend.strip()!r}."
            )
    return checkpoint


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "LEGACY_CHECKPOINT_FORMATS",
    "LEGACY_CHECKPOINT_VERSION",
    "CheckpointValidationError",
    "build_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
]
