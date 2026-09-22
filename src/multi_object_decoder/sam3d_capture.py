"""Observe the pinned SAM3D sampler without modifying its sampling computation.

The observer saves actual last-step tensors. It neither changes a solver state
nor re-runs the backbone, and restores all instance methods on exit.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


SAM3D_REVISION = "afdf6a31522d038c44c68a0bb57aa68827380797"


def tensor_preserving_asdict(value: Any) -> Any:
    """Dataclass conversion that preserves tensor identities and their gradients."""
    if isinstance(value, torch.Tensor):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: tensor_preserving_asdict(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, dict):
        return {key: tensor_preserving_asdict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_preserving_asdict(item) for item in value)
    return copy.deepcopy(value)


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    if isinstance(value, Mapping):
        return {key: _numpy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_numpy(item) for item in value)
    return value


@contextmanager
def _replace_method(owner: Any, name: str, replacement: Any) -> Iterator[None]:
    previous = vars(owner).get(name)
    had_instance_value = name in vars(owner)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


class SAM3DCacheCapture:
    """Scoped observer for the upstream pointmap pipeline's sparse-structure stage."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.payload: dict[str, Any] = {}
        self.image_condition: Any = None
        self.completed = False

    @contextmanager
    def observe(self) -> Iterator["SAM3DCacheCapture"]:
        pipeline = self.pipeline
        generator = pipeline.models["ss_generator"]
        backbone = generator.reverse_fn.backbone
        solver = getattr(generator, "_solver", None)
        if solver is None or getattr(generator, "_solver_method", None) != "euler":
            raise RuntimeError("MOD cache capture requires SAM3D's Euler sparse-structure solver")
        sampler = getattr(pipeline, "sample_sparse_structure", None)
        condition = getattr(pipeline, "get_condition_input", None)
        split = getattr(backbone, "split_latent_share_transformer", None)
        if not all(callable(item) for item in (sampler, condition, split)):
            raise RuntimeError(
                "SAM3D cache capture API is incompatible; use upstream revision " + SAM3D_REVISION
            )
        original_step = solver.step
        self.payload = {}
        self.image_condition = None
        self.completed = False
        sampling = False
        stepping = False

        def capture_split(*args: Any, **kwargs: Any) -> Any:
            output = split(*args, **kwargs)
            if stepping:
                # Preserve the last backbone invocation of the original Euler
                # step, including upstream classifier-free-guidance ordering.
                self.payload["pose_latents_raw"] = _snapshot(output)
            return output

        def capture_step(dynamics: Any, x_t: Any, t: Any, dt: Any,
                         *args: Any, **kwargs: Any) -> Any:
            nonlocal stepping
            self.payload.pop("pose_latents_raw", None)
            self.payload["x_t_last_step"] = _snapshot(x_t)
            self.payload["solver_state"] = _snapshot({"t": t, "dt": dt})
            stepping = True
            try:
                return original_step(dynamics, x_t, t, dt, *args, **kwargs)
            finally:
                stepping = False

        def capture_condition(*args: Any, **kwargs: Any) -> Any:
            output = condition(*args, **kwargs)
            condition_args, _ = output
            if not condition_args or not isinstance(condition_args[0], torch.Tensor):
                raise RuntimeError("SAM3D must expose a tensor as its first image condition")
            self.image_condition = _snapshot(condition_args[0])
            return output

        def capture_sample(inputs: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
            nonlocal sampling
            if sampling or self.completed:
                raise RuntimeError("One MOD object cache must contain exactly one SS sampling call")
            sampling = True
            for key in ("pointmap_scale", "pointmap_shift"):
                if not isinstance(inputs.get(key), torch.Tensor):
                    raise RuntimeError("SAM3D preprocessing omitted " + key)
                self.payload[key] = _snapshot(inputs[key])
            try:
                with ExitStack() as stack:
                    stack.enter_context(_replace_method(solver, "step", capture_step))
                    stack.enter_context(_replace_method(
                        backbone, "split_latent_share_transformer", capture_split
                    ))
                    stack.enter_context(_replace_method(
                        pipeline, "get_condition_input", capture_condition
                    ))
                    output = sampler(inputs, *args, **kwargs)
                required = {
                    "pose_latents_raw", "x_t_last_step", "solver_state",
                    "pointmap_scale", "pointmap_shift",
                }
                missing = sorted(required - self.payload.keys())
                if missing or self.image_condition is None:
                    raise RuntimeError("SAM3D did not produce a complete cache: " + str(missing))
                self.completed = True
                return output
            finally:
                sampling = False

        with _replace_method(pipeline, "sample_sparse_structure", capture_sample):
            yield self

    def save(self, directory: Path, object_id: str) -> None:
        """Write the existing per-object cache contract from observed tensors."""
        if not self.completed:
            raise RuntimeError("Cannot save a SAM3D cache before successful SS sampling")
        if not object_id.isdigit():
            raise ValueError("SAM3D object ID must contain only digits")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = _numpy(self.payload)
        arrays = {}
        for key, value in payload.items():
            if isinstance(value, Mapping):
                boxed = np.empty((), dtype=object)
                boxed[()] = dict(value)
                arrays[key] = boxed
            else:
                arrays[key] = np.asarray(value)
        # This legacy per-object format remains trusted only within prepare.
        # Bundled/downloadable scene caches use the pickle-free scene format.
        np.savez_compressed(directory / ("obj_" + object_id + ".npz"), **arrays)
        np.save(directory / ("img_cond_" + object_id + ".npy"),
                _numpy(self.image_condition), allow_pickle=False)
