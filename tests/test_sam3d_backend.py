"""CPU-only coverage for the optional, differentiable SAM3D backend."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch import nn

from multi_object_decoder.data import DecodeState, SceneSample
from multi_object_decoder.poses import pose_loss
from multi_object_decoder.sam3d_backend import SAM3DBackend
from multi_object_decoder.workflow import build_system


RELEASE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIM = 4
OUTPUT_DIMS = {
    "6drotation_normalized": 6,
    "translation": 3,
    "scale": 3,
    "translation_scale": 1,
    "shape": 8,
}


class _FakeLatentMapping(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.out_layer = nn.Linear(FEATURE_DIM, output_dim)
        nn.init.constant_(self.out_layer.weight, 0.02)
        nn.init.constant_(self.out_layer.bias, 0.05)

    def to_output(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_layer(value)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_mapping = nn.ModuleDict(
            {
                key: _FakeLatentMapping(output_dim)
                for key, output_dim in OUTPUT_DIMS.items()
            }
        )
        self.input_latent_mappings = list(OUTPUT_DIMS)


class _FakePipeline:
    def __init__(self) -> None:
        backbone = _FakeBackbone()
        generator = SimpleNamespace(reverse_fn=SimpleNamespace(backbone=backbone))
        self.models = {"ss_generator": generator}

    def pose_decoder(
        self,
        state: dict[str, torch.Tensor],
        *,
        scene_scale: torch.Tensor,
        scene_shift: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        translation = state["translation"].reshape(1, 3)
        translation = translation + state["translation_scale"].reshape(1, 1)
        translation = translation * scene_scale.reshape(1, -1)[:, :1]
        translation = translation + scene_shift.reshape(1, 3)
        rotation = state["6drotation_normalized"].reshape(1, 6)[:, :4]
        scale = torch.exp(state["scale"].reshape(1, 3).clamp(min=-4.0, max=4.0))
        return {
            "translation": translation,
            "rotation": rotation,
            "scale": scale,
        }


def _scene_sample() -> SceneSample:
    object_count = 2
    generator = torch.Generator().manual_seed(17)
    raw_latents = {
        key: torch.randn(
            object_count,
            2 if key == "shape" else 1,
            FEATURE_DIM,
            generator=generator,
        )
        for key in OUTPUT_DIMS
    }
    x_t_last_step = {
        key: torch.randn(
            object_count,
            2 if key == "shape" else 1,
            output_dim,
            generator=generator,
        )
        * 0.1
        for key, output_dim in OUTPUT_DIMS.items()
    }
    decode_state = DecodeState(
        object_ids=(0, 3),
        pose_latents_raw=raw_latents,
        pointmap_scale=torch.ones(object_count, 1),
        pointmap_shift=torch.zeros(object_count, 3),
        solver_state=({"dt": 0.1}, {"dt": 0.1}),
        x_t_last_step=x_t_last_step,
        image_condition_per_object=torch.randn(
            object_count, 2, FEATURE_DIM, generator=generator
        ),
        source_path=Path("synthetic-cache.npz"),
    )
    target_poses = torch.tensor(
        [
            [0.5, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [-0.4, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return SceneSample(
        scene_id="synthetic/sam3d",
        object_names=("obj_0", "obj_3"),
        pose_tokens=torch.randn(
            object_count, 4, FEATURE_DIM, generator=generator
        ),
        shape_tokens=torch.randn(
            object_count, 2, FEATURE_DIM, generator=generator
        ),
        base_poses=target_poses.clone(),
        target_poses=target_poses,
        decode_state=decode_state,
        record={},
    )


def test_sam3d_workflow_predict_pose_loss_backward_cpu() -> None:
    torch.manual_seed(0)
    backend = SAM3DBackend(
        sam3d_root=Path("synthetic-sam3d"),
        config_path=Path("synthetic-pipeline.yaml"),
        pipeline=_FakePipeline(),
    )
    system = build_system(
        "sam3d",
        {
            "feature_dim": FEATURE_DIM,
            "input_dim": FEATURE_DIM,
            "num_heads": 1,
            "mlp_hidden_dim": 8,
            "dropout": 0.0,
            "num_layers": 1,
            "use_global_attn": True,
            "use_layer_norm": True,
            "use_shape_tokens": True,
        },
        decoder=backend,
    )
    sample = _scene_sample()

    predicted = system.predict(sample, torch.device("cpu"))
    assert predicted.shape == (2, 10)
    assert predicted.device.type == "cpu"
    assert torch.isfinite(predicted).all()

    assert sample.target_poses is not None
    loss, components = pose_loss(predicted, sample.target_poses)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in system.model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_sam3d_backend_import_is_lazy() -> None:
    source_root = RELEASE_ROOT / "src"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else os.pathsep.join((str(source_root), existing_pythonpath))
    )
    code = r'''
import importlib.abc
import sys

blocked = ("sam3d_objects", "hydra", "omegaconf")

class RejectOptionalDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(f"optional dependency imported eagerly: {fullname}")
        return None

sys.meta_path.insert(0, RejectOptionalDependency())
import multi_object_decoder.sam3d_backend
assert not any(name in sys.modules for name in blocked)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"lazy import subprocess exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
