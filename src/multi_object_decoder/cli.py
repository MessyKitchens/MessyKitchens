"""Small shared helpers for the public command-line entry points."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf


def load_config(path: Path | str, overrides: Sequence[str] = ()) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    config = OmegaConf.load(source)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"{source}: top-level configuration must be a mapping")
    return resolved


def select_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path | str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(0, min(int(warmup_steps), total_steps))

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def scene_directory_name(scene_id: str) -> str:
    safe = scene_id.replace("\\", "__").replace("/", "__").strip("._")
    if not safe:
        raise ValueError(f"Invalid empty scene_id after sanitizing {scene_id!r}")
    return safe


def validated_scene_directory_names(scene_ids: Sequence[str]) -> tuple[str, ...]:
    """Sanitize scene IDs and reject two records mapping to one directory."""

    names: list[str] = []
    source_by_name: dict[str, str] = {}
    for scene_id in scene_ids:
        name = scene_directory_name(scene_id)
        if name in source_by_name:
            raise ValueError(
                "Scene identifiers map to the same output directory "
                f"{name!r}: {source_by_name[name]!r} and {scene_id!r}"
            )
        source_by_name[name] = scene_id
        names.append(name)
    return tuple(names)
