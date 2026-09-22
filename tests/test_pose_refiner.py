"""CPU-only smoke tests for the released pose refiner."""

import torch

from multi_object_decoder.models.pose_refiner import SAM3DPoseRefinementModel


def test_pose_refiner_synthetic_forward_cpu() -> None:
    torch.manual_seed(0)
    batch_size = 2
    feature_dim = 16
    model = SAM3DPoseRefinementModel(
        feature_dim=feature_dim,
        input_dim=feature_dim,
        num_heads=4,
        mlp_hidden_dim=32,
        dropout=0.0,
        num_layers=2,
        use_global_attn=True,
    ).cpu().eval()
    pose_tokens = torch.randn(batch_size, 4, feature_dim)
    shape_tokens = torch.randn(batch_size, 12, feature_dim)

    with torch.inference_mode():
        refined_tokens, pose_delta = model(pose_tokens, shape_tokens)

    assert refined_tokens.device.type == "cpu"
    assert refined_tokens.shape == pose_tokens.shape
    assert torch.isfinite(refined_tokens).all()
    assert isinstance(pose_delta, dict)
    assert pose_delta == {}
