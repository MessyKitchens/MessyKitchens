"""Multi-Object Decoder (MOD) pose refiner built on SAM3D pose and shape tokens.

Each refiner block runs three attention stages followed by an MLP:

1. Intra-object self-attention over the four pose tokens of one object, which
   relates the object's own pose parameters.
2. Inter-object self-attention over the pose tokens of every object in the
   scene (``(B, 4, F)`` reshaped to ``(1, B*4, F)``), which imposes relative
   spatial constraints between objects.
3. Cross-attention from the refined pose tokens (queries) to the object's full
   shape tokens (keys and values), which retrieves geometric evidence.
4. An MLP. Every residual branch is zero-initialised, so a fresh model starts
   as an identity refiner.

The latent-mapping heads copied from the SAM3D backbone turn the refined pose
tokens into residuals for SAM3D's pose latents. Inputs are ``pose_tokens`` with
shape ``(B, 4, F)`` and optional ``shape_tokens`` with shape ``(B, S, F)``; the
output is ``(refined_pose_tokens, pose_latent_residuals)``.
"""

import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple, Dict


def _truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02):
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)


def _init_weights_improved(module: nn.Module) -> None:
    """Xavier + standard LayerNorm."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class V4RefinerBlock(nn.Module):
    """One MOD refiner block: intra-object SA -> [inter-object SA] -> cross-attention -> MLP.

    Pose tokens act as queries that retrieve geometric evidence from the shape
    tokens. Inter-object self-attention is optional: set ``use_global_attn=False``
    for single-object inputs or batches of unrelated objects.
    """
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.1,
        use_global_attn: bool = True,
    ):
        super().__init__()
        self.use_global_attn = use_global_attn

        # 1. Intra-object self-attention over the four pose tokens.
        self.norm_intra = nn.LayerNorm(feature_dim)
        self.intra_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_intra = nn.Dropout(dropout)

        # 2. Inter-object self-attention over all objects (B, 4 -> 1, B*4).
        self.norm_inter = nn.LayerNorm(feature_dim)
        self.inter_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_inter = nn.Dropout(dropout)

        # 3. Cross-attention: pose tokens query the shape tokens.
        self.norm_cross = nn.LayerNorm(feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_cross = nn.Dropout(dropout)

        # 4. MLP
        self.norm_mlp = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.dropout_mlp = nn.Dropout(dropout)

        self.apply(_init_weights_improved)
        # Zero-initialised residual branches: a fresh block is the identity.
        nn.init.zeros_(self.intra_attn.out_proj.weight)
        nn.init.zeros_(self.intra_attn.out_proj.bias)
        nn.init.zeros_(self.inter_attn.out_proj.weight)
        nn.init.zeros_(self.inter_attn.out_proj.bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(
        self,
        pose_tokens: torch.Tensor,
        shape_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # pose_tokens: (B, 4, C)
        B, num_pose, C = pose_tokens.shape

        # 1. Intra-object self-attention (pose tokens only).
        residual = pose_tokens
        x_norm = self.norm_intra(pose_tokens)
        x_sa, _ = self.intra_attn(x_norm, x_norm, x_norm)
        pose_tokens = residual + self.dropout_intra(x_sa)

        # 2. Inter-object self-attention; optional when the batch is unrelated.
        if self.use_global_attn:
            residual = pose_tokens
            x_norm = self.norm_inter(pose_tokens)
            x_flat = x_norm.reshape(1, B * num_pose, C)
            x_global, _ = self.inter_attn(x_flat, x_flat, x_flat)
            pose_tokens = residual + self.dropout_inter(x_global.reshape(B, num_pose, C))

        # 3. Cross-attention from pose queries to shape keys/values.
        if shape_tokens is not None:
            residual = pose_tokens
            x_norm = self.norm_cross(pose_tokens)
            # Q: (B, 4, C); K, V: (B, S, C); batched per object.
            x_cross, _ = self.cross_attn(x_norm, shape_tokens, shape_tokens)
            pose_tokens = residual + self.dropout_cross(x_cross)

        # 4. MLP
        pose_tokens = pose_tokens + self.dropout_mlp(self.mlp(self.norm_mlp(pose_tokens)))

        return pose_tokens


class SAM3DPoseRefinementModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1024,
        num_heads: int = 16,
        mlp_hidden_dim: int = 4096,
        dropout: float = 0.1,
        num_layers: int = 4,
        use_global_attn: bool = True,  # inter-object self-attention
        use_layer_norm: bool = True,
        shape_token_len: int = 4096,  # unused; kept for configuration compatibility
        input_dim: int = 1024,  # backbone token dim (pose/shape from SAM3D)
        reduced_dim: Optional[int] = None,  # capacity control: internal dim 1024->reduced_dim (e.g. 256)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reduced_dim = reduced_dim
        # Internal feature dim: reduced when capacity control is on to limit overfitting
        self.feature_dim = reduced_dim if reduced_dim is not None else feature_dim

        # Capacity control: project input_dim -> reduced_dim at the input and back at the output.
        if reduced_dim is not None:
            self.input_proj_pose = nn.Linear(input_dim, reduced_dim)
            self.input_proj_shape = nn.Linear(input_dim, reduced_dim)
            self.output_proj = nn.Linear(reduced_dim, input_dim)
            _init_weights_improved(self.input_proj_pose)
            _init_weights_improved(self.input_proj_shape)
            _init_weights_improved(self.output_proj)
        else:
            self.input_proj_pose = None
            self.input_proj_shape = None
            self.output_proj = None

        # Positional embedding that distinguishes the four pose tokens
        # (rotation, translation, scale, translation_scale); attention is
        # permutation-invariant without it.
        self.pose_pos_embed = nn.Parameter(torch.randn(1, 4, self.feature_dim) * 0.02)

        # Refiner blocks operate at self.feature_dim so capacity control applies.
        self.use_global_attn = use_global_attn
        self.blocks = nn.ModuleList([
            V4RefinerBlock(
                feature_dim=self.feature_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                use_global_attn=use_global_attn,
            )
            for _ in range(num_layers)
        ])

        if use_layer_norm:
            self.final_norm = nn.LayerNorm(self.feature_dim)
        else:
            self.final_norm = nn.Identity()
        _init_weights_improved(self.final_norm)

        # Latent-mapping heads are attached by load_latent_mapping_from_backbone.
        self.latent_mapping = nn.ModuleDict()
        self.input_latent_mappings = []
        self._adapter = nn.ModuleDict()
        self._pose_latent_keys = []
        self.register_buffer("delta_alpha", torch.tensor(1.0))

    def load_latent_mapping_from_backbone(self, backbone: nn.Module) -> None:
        """Copy the SAM3D latent heads and create near-zero-initialised adapters."""
        if not hasattr(backbone, "latent_mapping") or not backbone.latent_mapping:
            return
        self.latent_mapping = nn.ModuleDict(
            {k: copy.deepcopy(v) for k, v in backbone.latent_mapping.items()}
        )
        self.input_latent_mappings = list(self.latent_mapping.keys())
        self._pose_latent_keys = [k for k in self.input_latent_mappings if k != "shape"]

        # Adapter input dim: 4 * input_dim (refiner output is projected to input_dim when reduced_dim)
        in_flat = 4 * self.input_dim
        adapter = nn.ModuleDict()
        for k in self._pose_latent_keys:
            m = self.latent_mapping[k]
            model_channels = m.out_layer.in_features
            adapter[k] = nn.Linear(in_flat, model_channels)
        self._adapter = adapter
        _init_weights_improved(self._adapter)

        for k in self._pose_latent_keys:
            m = self.latent_mapping[k]
            if hasattr(m, "out_layer"):
                nn.init.normal_(m.out_layer.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.out_layer.bias)

    def forward_pose_residual(self, refined_pose_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._pose_latent_keys or len(self._adapter) == 0:
            return {}
        B, C, F = refined_pose_tokens.shape
        x_flat = refined_pose_tokens.reshape(B, C * F)
        out = {}
        for k in self._pose_latent_keys:
            h = self._adapter[k](x_flat)
            latent = self.latent_mapping[k].to_output(h)
            out[k] = latent.unsqueeze(1)
        return out

    def forward(
        self,
        pose_tokens: torch.Tensor,
        shape_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine pose tokens and return SAM3D latent residuals.

        Args:
            pose_tokens: ``(B, 4, F)`` with ``F = input_dim``.
            shape_tokens: ``(B, S, F)`` or ``None``; all shape tokens are used
                directly, without aggregation.
        """
        if self.input_proj_pose is not None:
            pose_tokens = self.input_proj_pose(pose_tokens)
            if shape_tokens is not None:
                shape_tokens = self.input_proj_shape(shape_tokens)

        B, num_pose_tokens, F = pose_tokens.shape
        x = pose_tokens + self.pose_pos_embed

        # Intra-object SA -> inter-object SA -> cross-attention -> MLP, per block.
        for block in self.blocks:
            x = block(x, shape_tokens)

        refined_pose_tokens = self.final_norm(x)
        if self.output_proj is not None:
            refined_pose_tokens = self.output_proj(refined_pose_tokens)
        delta = self.forward_pose_residual(refined_pose_tokens) if self._pose_latent_keys else {}
        return refined_pose_tokens, delta


def create_sam3d_pose_refinement_model(
    feature_dim: int = 1024,
    num_heads: int = 16,
    mlp_hidden_dim: int = 4096,
    dropout: float = 0.1,
    num_layers: int = 4,
    use_global_attn: bool = True,
    use_layer_norm: bool = True,
    shape_token_len: int = 4096,  # unused; kept for configuration compatibility
    input_dim: int = 1024,
    reduced_dim: Optional[int] = None,  # capacity control: internal dim 1024->reduced_dim (e.g. 256)
) -> SAM3DPoseRefinementModel:
    return SAM3DPoseRefinementModel(
        feature_dim=feature_dim,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        use_global_attn=use_global_attn,
        use_layer_norm=use_layer_norm,
        shape_token_len=shape_token_len,
        input_dim=input_dim,
        reduced_dim=reduced_dim,
    )


# Backward compatibility
SAM3DPoseRefiner = SAM3DPoseRefinementModel
create_sam3d_pose_refiner = create_sam3d_pose_refinement_model


class SimpleMLPRefiner(nn.Module):
    """MLP ablation baseline without attention.

    Concatenates the pose tokens (``4*F``) with max-pooled shape tokens (``F``)
    and regresses the token residual with an MLP.
    """
    def __init__(
        self,
        feature_dim: int = 1024,
        mlp_hidden_dim: int = 4096,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        # Input: Pose (4*F) + Shape (1*F after MaxPool) = 5*F
        input_dim = 5 * feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 4 * feature_dim),
        )

        self.latent_mapping = nn.ModuleDict()
        self.input_latent_mappings = []
        self._pose_latent_keys = []
        self._adapter = nn.ModuleDict()
        self.register_buffer("delta_alpha", torch.tensor(1.0))

        self.apply(_init_weights_improved)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def load_latent_mapping_from_backbone(self, backbone: nn.Module) -> None:
        if not hasattr(backbone, "latent_mapping") or not backbone.latent_mapping:
            return
        self.latent_mapping = nn.ModuleDict(
            {k: copy.deepcopy(v) for k, v in backbone.latent_mapping.items()}
        )
        self.input_latent_mappings = list(self.latent_mapping.keys())
        self._pose_latent_keys = [k for k in self.input_latent_mappings if k != "shape"]

        in_flat = 4 * self.feature_dim
        adapter = nn.ModuleDict()
        for k in self._pose_latent_keys:
            m = self.latent_mapping[k]
            model_channels = m.out_layer.in_features
            adapter[k] = nn.Linear(in_flat, model_channels)
            nn.init.zeros_(adapter[k].weight)
            nn.init.zeros_(adapter[k].bias)
        self._adapter = adapter

        for k in self._pose_latent_keys:
            m = self.latent_mapping[k]
            if hasattr(m, "out_layer"):
                nn.init.normal_(m.out_layer.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.out_layer.bias)

    def forward_pose_residual(self, x_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._pose_latent_keys or len(self._adapter) == 0:
            return {}
        out = {}
        for k in self._pose_latent_keys:
            h = self._adapter[k](x_flat)
            latent = self.latent_mapping[k].to_output(h)
            out[k] = latent.unsqueeze(1)
        return out

    def forward(
        self,
        pose_tokens: torch.Tensor,
        shape_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # pose_tokens: (B, 4, F), shape_tokens: (B, N, F)
        B, _, F = pose_tokens.shape

        if shape_tokens is not None:
            shape_feat = torch.max(shape_tokens, dim=1)[0]
        else:
            shape_feat = torch.zeros(B, F, device=pose_tokens.device, dtype=pose_tokens.dtype)

        pose_flat = pose_tokens.reshape(B, -1)
        x = torch.cat([pose_flat, shape_feat], dim=1)

        delta_flat = self.mlp(x)
        refined_flat = pose_flat + delta_flat
        refined_tokens = refined_flat.reshape(B, 4, F)

        delta_out = self.forward_pose_residual(delta_flat)
        return refined_tokens, delta_out


def create_simple_mlp_refiner(
    feature_dim: int = 1024,
    mlp_hidden_dim: int = 4096,
) -> SimpleMLPRefiner:
    return SimpleMLPRefiner(
        feature_dim=feature_dim,
        mlp_hidden_dim=mlp_hidden_dim,
    )

