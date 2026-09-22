"""Shared MOD training and inference engine.

Both CLIs use this module, so checkpoint loading, tensor preparation and pose
decoding cannot silently diverge between training and inference.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol

import torch
import torch.nn as nn

from .models import (
    SAM3DPoseRefinementModel,
    create_direct_pose_refiner,
)
from .objectives import ObjectiveSettings, TrainingObjective


SUPPORTED_BACKENDS = {"direct", "sam3d"}


class DecoderBackend(Protocol):
    def initialize_model(self, model: SAM3DPoseRefinementModel) -> None: ...

    def decode(
        self,
        sample: Any,
        refined_pose_delta: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor: ...


@dataclass
class WorkflowSystem:
    backend_name: str
    model: nn.Module
    model_config: dict[str, Any]
    decoder: DecoderBackend | None = None
    use_shape_tokens: bool = True

    def predict(self, sample: Any, device: torch.device) -> torch.Tensor:
        pose_tokens = sample.pose_tokens.to(device=device, dtype=torch.float32)
        shape_tokens = sample.shape_tokens
        if shape_tokens is not None and self.use_shape_tokens:
            shape_tokens = shape_tokens.to(device=device, dtype=torch.float32)
        else:
            shape_tokens = None

        if self.backend_name == "direct":
            if sample.base_poses is None:
                raise ValueError(
                    f"{sample.scene_id}: direct backend requires baseline poses "
                    "(base_poses_path/sam3d_pose_path or pose_inference.json)"
                )
            base_poses = sample.base_poses.to(device=device, dtype=torch.float32)
            predicted, _ = self.model(pose_tokens, shape_tokens, base_poses)
            return predicted

        if self.backend_name == "sam3d":
            if self.decoder is None:
                raise RuntimeError("sam3d backend was created without a decoder")
            _, refined_delta = self.model(pose_tokens, shape_tokens)
            if not isinstance(refined_delta, Mapping) or not refined_delta:
                raise RuntimeError(
                    "SAM3D refiner returned no latent residuals. The model must be "
                    "initialized from the exact SAM3D backbone before loading weights."
                )
            return self.decoder.decode(sample, refined_delta, device=device)

        raise RuntimeError(f"Unsupported workflow backend: {self.backend_name}")


def build_system(
    backend: str,
    model_config: Mapping[str, Any],
    *,
    decoder: DecoderBackend | None = None,
) -> WorkflowSystem:
    backend = backend.lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(SUPPORTED_BACKENDS)}, got {backend!r}")
    config = dict(model_config)
    use_shape_tokens = bool(config.pop("use_shape_tokens", True))
    config.pop("shape_token_len", None)
    if backend == "direct":
        model = create_direct_pose_refiner(config)
        resolved_config = dict(model.model_config)
        resolved_config["use_shape_tokens"] = use_shape_tokens
        return WorkflowSystem(
            backend_name=backend,
            model=model,
            model_config=resolved_config,
            use_shape_tokens=use_shape_tokens,
        )

    allowed = {
        "feature_dim",
        "num_heads",
        "mlp_hidden_dim",
        "dropout",
        "num_layers",
        "use_global_attn",
        "use_layer_norm",
        "input_dim",
        "reduced_dim",
    }
    unknown = sorted(set(config) - allowed - {"pose_head_hidden_dim"})
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {unknown}")
    config.pop("pose_head_hidden_dim", None)
    model = SAM3DPoseRefinementModel(**{key: value for key, value in config.items() if key in allowed})
    if decoder is None:
        raise ValueError("backend='sam3d' requires a configured SAM3D decoder")
    decoder.initialize_model(model)
    resolved_config = {key: value for key, value in config.items() if key in allowed}
    resolved_config["use_shape_tokens"] = use_shape_tokens
    return WorkflowSystem(
        backend_name=backend,
        model=model,
        model_config=resolved_config,
        decoder=decoder,
        use_shape_tokens=use_shape_tokens,
    )


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError(f"{precision} precision requires a CUDA device")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _average_metrics(metrics: Iterable[Mapping[str, float]]) -> dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {}
    keys = set.intersection(*(set(row) for row in rows))
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


@torch.no_grad()
def evaluate(
    system: WorkflowSystem,
    dataset: Any,
    *,
    device: torch.device,
    loss_weights: Mapping[str, float],
    max_scenes: int | None = None,
    objective: TrainingObjective | None = None,
) -> dict[str, float]:
    system.model.eval()
    if objective is None:
        objective = TrainingObjective(
            ObjectiveSettings.from_mapping(None, workflow_backend=system.backend_name),
            loss_weights,
        )
    rows = []
    count = len(dataset) if max_scenes is None else min(len(dataset), max_scenes)
    for index in range(count):
        sample = dataset[index]
        if sample.target_poses is None:
            raise ValueError(f"{sample.scene_id}: validation sample has no target poses")
        predicted = system.predict(sample, device)
        _, components = objective(
            predicted.float(),
            sample.target_poses.to(device=device, dtype=torch.float32),
            sample,
            # The recovered camera-ready validation loop used raw pose loss
            # but did not add its training-time CD term.
            include_chamfer=False,
        )
        rows.append({key: float(value.cpu()) for key, value in components.items()})
    result = _average_metrics(rows)
    result["scenes"] = float(count)
    return result


def train(
    system: WorkflowSystem,
    train_dataset: Any,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    loss_weights: Mapping[str, float],
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    max_steps: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
    seed: int = 42,
    validation_dataset: Any | None = None,
    validation_max_scenes: int | None = None,
    scheduler: Any | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    objective: TrainingObjective | None = None,
) -> dict[str, Any]:
    """Train scene-by-scene, preserving inter-object attention boundaries."""
    if len(train_dataset) == 0:
        raise ValueError("Training manifest contains no scenes")
    if epochs <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16 or bf16")
    if max_steps is not None and global_step >= max_steps:
        return {
            "epoch": start_epoch,
            "global_step": global_step,
            "metrics": {"status": "max_steps_already_reached"},
        }

    if objective is None:
        objective = TrainingObjective(
            ObjectiveSettings.from_mapping(None, workflow_backend=system.backend_name),
            loss_weights,
        )

    system.model.to(device)
    rng = random.Random(seed)
    optimizer.zero_grad(set_to_none=True)
    last_metrics: dict[str, Any] = {}
    micro_step = 0
    stop = False
    for epoch in range(start_epoch, epochs):
        order = list(range(len(train_dataset)))
        rng.shuffle(order)
        epoch_rows = []
        system.model.train()
        for position, index in enumerate(order):
            sample = train_dataset[index]
            if sample.target_poses is None:
                raise ValueError(f"{sample.scene_id}: training sample has no target poses")
            with _autocast(device, precision):
                predicted = system.predict(sample, device)
                loss, components = objective(
                    predicted.float(),
                    sample.target_poses.to(device=device, dtype=torch.float32),
                    sample,
                )
                scaled_loss = loss / gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{sample.scene_id}: non-finite training loss")
            scaled_loss.backward()
            micro_step += 1
            update = (
                micro_step % gradient_accumulation_steps == 0
                or position == len(order) - 1
            )
            grad_norm = None
            if update:
                if max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        system.model.parameters(), float(max_grad_norm)
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(f"{sample.scene_id}: non-finite gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            row = {key: float(value.cpu()) for key, value in components.items()}
            row.update(
                {
                    "scene_id": sample.scene_id,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "updated": update,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if grad_norm is not None:
                row["grad_norm"] = float(grad_norm.detach().cpu())
            epoch_rows.append({key: value for key, value in row.items() if isinstance(value, float)})
            last_metrics = row
            if on_step is not None:
                on_step(row)
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break

        epoch_metrics = _average_metrics(epoch_rows)
        state: dict[str, Any] = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": epoch_metrics,
        }
        if validation_dataset is not None and len(validation_dataset):
            state["validation"] = evaluate(
                system,
                validation_dataset,
                device=device,
                loss_weights=loss_weights,
                max_scenes=validation_max_scenes,
                objective=objective,
            )
            system.model.train()
        if on_epoch is not None:
            on_epoch(state)
        last_metrics = state
        if stop:
            break
    return {
        "epoch": int(last_metrics.get("epoch", epochs)),
        "global_step": global_step,
        "metrics": last_metrics,
    }
