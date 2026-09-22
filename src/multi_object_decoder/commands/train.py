#!/usr/bin/env python3
"""Train MOD from validated cached SAM3D scene artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from multi_object_decoder.checkpoints import load_checkpoint, save_checkpoint
from multi_object_decoder.cli import (
    append_jsonl,
    load_config,
    make_cosine_scheduler,
    seed_everything,
    select_device,
)
from multi_object_decoder.data import SceneDataset
from multi_object_decoder.objectives import ObjectiveSettings, TrainingObjective
from multi_object_decoder.workflow import build_system, train


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the MOD pose refiner. The direct backend is self-contained after "
            "SAM3D token caching; the sam3d backend reproduces latent decoding and "
            "requires the patched external checkout."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-json",
        type=Path,
        help=(
            "Training manifest, relative to the current working directory. Overrides "
            "train.data_json in the config and dot-list arguments; config paths remain "
            "relative to the config file."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help=(
            "Trust legacy tokens/obj_*.npz caches containing pickle-backed object "
            "arrays. Portable scene NPZ caches do not need this flag."
        ),
    )
    parser.add_argument("--sam3d-root", type=Path, default=os.environ.get("SAM3D_ROOT"))
    parser.add_argument(
        "--sam3d-config",
        type=Path,
        default=os.environ.get("SAM3D_PIPELINE_CONFIG"),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, for example train.epochs=2",
    )
    return parser.parse_args()


def _config_path(value: Any, config_path: Path, field: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{field} must be configured")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def _create_decoder(args: argparse.Namespace, backend: str):
    if backend != "sam3d":
        return None
    if args.sam3d_root is None or args.sam3d_config is None:
        raise ValueError(
            "backend=sam3d requires --sam3d-root and --sam3d-config "
            "(or SAM3D_ROOT and SAM3D_PIPELINE_CONFIG)"
        )
    from multi_object_decoder.sam3d_backend import SAM3DBackend

    return SAM3DBackend.from_paths(args.sam3d_root, args.sam3d_config)


def _optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("name", "adamw")).lower()
    kwargs = {
        "lr": float(config.get("lr", 5e-4)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "eps": float(config.get("eps", 1e-8)),
    }
    if "betas" in config:
        kwargs["betas"] = tuple(float(value) for value in config["betas"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError("optimizer.name must be adam or adamw")


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, args.overrides)
    if args.data_json is not None:
        config.setdefault("train", {})["data_json"] = str(args.data_json.expanduser().resolve())
    train_config = dict(config.get("train", {}))
    val_config = dict(config.get("val", {}))
    workflow_config = dict(config.get("workflow", {}))

    checkpoint = load_checkpoint(args.resume) if args.resume else None
    configured_backend = str(workflow_config.get("backend", "direct")).lower()
    if checkpoint and checkpoint["version"] > 0:
        backend = checkpoint["backend"]
        model_config = dict(checkpoint["model_config"])
        if configured_backend != backend:
            raise ValueError(
                f"Configured backend {configured_backend!r} does not match resume "
                f"checkpoint backend {backend!r}"
            )
    else:
        backend = configured_backend
        model_config = dict(config.get("model", {}))
    if checkpoint and checkpoint["version"] == 0 and not model_config:
        raise ValueError("A legacy checkpoint requires a complete model section in --config")

    seed = int(args.seed if args.seed is not None else train_config.get("seed", 42))
    seed_everything(seed)
    device = select_device(args.device)
    allow_tf32 = train_config.get("allow_tf32", False)
    if not isinstance(allow_tf32, bool):
        raise ValueError("train.allow_tf32 must be true or false")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    decoder = _create_decoder(args, backend)
    system = build_system(backend, model_config, decoder=decoder)
    system.model.to(device)
    if checkpoint is not None:
        missing, unexpected = system.model.load_state_dict(checkpoint["state_dict"], strict=True)
        if missing or unexpected:  # strict=True normally raises before this point.
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    train_manifest = _config_path(train_config.get("data_json"), config_path, "train.data_json")
    val_value = val_config.get("data_json")
    train_dataset = SceneDataset(
        train_manifest,
        require_target=True,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    validation_dataset = (
        SceneDataset(
            _config_path(val_value, config_path, "val.data_json"),
            require_target=True,
            allow_legacy_pickle=args.allow_legacy_pickle,
        )
        if val_value
        else None
    )

    objective_value = train_config.get("objective")
    if objective_value is not None and not isinstance(objective_value, dict):
        raise ValueError("train.objective must be a mapping")
    loss_weights = dict(train_config.get("loss_weights", {}))
    objective_settings = ObjectiveSettings.from_mapping(
        objective_value,
        workflow_backend=backend,
    )
    objective = TrainingObjective(objective_settings, loss_weights)

    optimizer = _optimizer(system.model, dict(config.get("optimizer", {})))
    training_state = dict(checkpoint["training_state"]) if checkpoint else {}
    if checkpoint and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])

    epochs = int(train_config.get("epochs", 1))
    accumulation = int(train_config.get("gradient_accumulation_steps", 1))
    estimated_steps = max(1, epochs * ((len(train_dataset) + accumulation - 1) // accumulation))
    max_steps = (
        args.max_train_steps
        if args.max_train_steps is not None
        else train_config.get("max_train_steps")
    )
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("--max-train-steps must be positive")
        estimated_steps = max_steps
    scheduler_config = dict(config.get("lr_scheduler", {}))
    scheduler_total_steps = int(scheduler_config.get("total_steps", estimated_steps))
    if scheduler_total_steps <= 0:
        raise ValueError("lr_scheduler.total_steps must be positive")
    scheduler = make_cosine_scheduler(
        optimizer,
        total_steps=scheduler_total_steps,
        warmup_steps=int(scheduler_config.get("num_warmup_steps", 0)),
    )
    if training_state.get("scheduler"):
        scheduler.load_state_dict(training_state["scheduler"])

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    save_frequency = int(train_config.get("save_freq", 0))
    last_saved_step = -1
    best_validation = float(training_state.get("best_validation_loss", "inf"))

    def save(name: str, state: dict[str, Any]) -> Path:
        resumable_state = dict(state)
        resumable_state["scheduler"] = scheduler.state_dict()
        resumable_state["best_validation_loss"] = best_validation
        return save_checkpoint(
            checkpoint_dir / name,
            backend=system.backend_name,
            model_config=system.model_config,
            state_dict=system.model.state_dict(),
            optimizer=optimizer,
            training_state=resumable_state,
        )

    def on_step(row: dict[str, Any]) -> None:
        nonlocal last_saved_step
        append_jsonl(metrics_path, {"split": "train_step", **row})
        step = int(row["global_step"])
        if (
            row.get("updated")
            and save_frequency > 0
            and step > 0
            and step % save_frequency == 0
            and step != last_saved_step
        ):
            save(f"step_{step:08d}.pt", {"epoch": int(row["epoch"]) - 1, "global_step": step})
            last_saved_step = step

    def on_epoch(state: dict[str, Any]) -> None:
        nonlocal best_validation
        append_jsonl(metrics_path, {"split": "epoch", **state})
        validation = state.get("validation")
        if validation and float(validation["loss"]) < best_validation:
            best_validation = float(validation["loss"])
            save("best.pt", state)
        save("last.pt", state)

    precision = args.precision or str(train_config.get("precision", "fp32"))
    result = train(
        system,
        train_dataset,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        loss_weights=loss_weights,
        precision=precision,
        gradient_accumulation_steps=accumulation,
        max_grad_norm=(
            None
            if train_config.get("max_grad_norm") is None
            else float(train_config.get("max_grad_norm"))
        ),
        max_steps=max_steps,
        start_epoch=int(training_state.get("epoch", 0)),
        global_step=int(training_state.get("global_step", 0)),
        seed=seed,
        validation_dataset=validation_dataset,
        validation_max_scenes=val_config.get("max_val_scenes"),
        scheduler=scheduler,
        on_step=on_step,
        on_epoch=on_epoch,
        objective=objective,
    )
    final_path = save("last.pt", result)
    summary = {
        "backend": backend,
        "device": str(device),
        "checkpoint": str(final_path),
        **result,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
