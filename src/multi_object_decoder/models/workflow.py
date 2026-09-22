"""Checkpoint-stable MOD models used by the portable workflow."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ..poses import apply_pose_delta
from .pose_refiner import SAM3DPoseRefinementModel, _init_weights_improved


class DirectPoseRefinementModel(nn.Module):
    """MOD attention refiner with a portable, identity-initialized pose head.

    The camera-ready latent decoder remains available through the ``sam3d``
    backend.  This head lets cached-token training and inference run without
    importing the external SAM3D repository after token extraction.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 1024,
        num_heads: int = 16,
        mlp_hidden_dim: int = 4096,
        dropout: float = 0.1,
        num_layers: int = 4,
        use_global_attn: bool = True,
        use_layer_norm: bool = True,
        input_dim: int | None = None,
        reduced_dim: int | None = None,
        pose_head_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim if input_dim is not None else feature_dim)
        self.model_config = {
            "feature_dim": int(feature_dim),
            "num_heads": int(num_heads),
            "mlp_hidden_dim": int(mlp_hidden_dim),
            "dropout": float(dropout),
            "num_layers": int(num_layers),
            "use_global_attn": bool(use_global_attn),
            "use_layer_norm": bool(use_layer_norm),
            "input_dim": input_dim,
            "reduced_dim": int(reduced_dim) if reduced_dim is not None else None,
            "pose_head_hidden_dim": (
                int(pose_head_hidden_dim) if pose_head_hidden_dim is not None else input_dim
            ),
        }
        self.refiner = SAM3DPoseRefinementModel(
            feature_dim=feature_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            use_global_attn=use_global_attn,
            use_layer_norm=use_layer_norm,
            input_dim=input_dim,
            reduced_dim=reduced_dim,
        )
        hidden_dim = self.model_config["pose_head_hidden_dim"]
        self.pose_head = nn.Sequential(
            nn.LayerNorm(4 * input_dim),
            nn.Linear(4 * input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10),
        )
        self.pose_head.apply(_init_weights_improved)
        # A fresh model is an exact identity refiner, which stabilizes training.
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)

    def forward(
        self,
        pose_tokens: torch.Tensor,
        shape_tokens: torch.Tensor | None,
        base_poses: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        refined_tokens, _ = self.refiner(pose_tokens, shape_tokens)
        pose_delta = self.pose_head(refined_tokens.flatten(start_dim=1))
        predicted_poses = apply_pose_delta(base_poses, pose_delta)
        return predicted_poses, {
            "pose_delta": pose_delta,
            "refined_pose_tokens": refined_tokens,
        }


def create_direct_pose_refiner(config: Mapping[str, Any]) -> DirectPoseRefinementModel:
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
        "pose_head_hidden_dim",
    }
    unknown = sorted(set(config) - allowed - {"use_shape_tokens", "shape_token_len"})
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {unknown}")
    return DirectPoseRefinementModel(**{key: value for key, value in config.items() if key in allowed})
