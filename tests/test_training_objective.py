"""CPU tests for the opt-in recovered paper training objective."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from multi_object_decoder.objectives import (
    ObjectiveSettings,
    TrainingObjective,
    compute_pose_chamfer_loss,
    load_canonical_points,
    torch_chamfer_distance,
    transform_points_by_pose,
)
from multi_object_decoder.poses import pose_loss


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _identity_poses(count: int) -> torch.Tensor:
    poses = torch.zeros(count, 10, dtype=torch.float32)
    poses[:, 3] = 1.0
    poses[:, 7:10] = 1.0
    return poses


def _paper_mapping(backend: str = "pytorch3d") -> dict:
    return {
        "profile": "paper",
        "scale_loss": "raw_mse",
        "chamfer": {
            "enabled": True,
            "backend": backend,
            "canonical_source": "points",
            "canonical_point_count": 4,
            "num_samples": 4,
            "sampling_seed": 7,
        },
    }


def test_raw_scale_mse_is_explicit_and_portable_default_is_unchanged() -> None:
    target = _identity_poses(1)
    predicted = target.clone()
    predicted[:, 7:10] = 2.0

    default_loss, default_components = pose_loss(
        predicted,
        target,
        translation_weight=0.0,
        rotation_weight=0.0,
    )
    raw_loss, raw_components = pose_loss(
        predicted,
        target,
        translation_weight=0.0,
        rotation_weight=0.0,
        scale_loss_mode="raw_mse",
    )

    assert default_components["scale_loss"] == pytest.approx(np.log(2.0) ** 2)
    assert default_loss == pytest.approx(np.log(2.0) ** 2)
    assert raw_components["scale_loss"] == pytest.approx(1.0)
    assert raw_loss == pytest.approx(1.0)


def test_paper_profile_rejects_portable_backends_and_loss_fallbacks() -> None:
    settings = ObjectiveSettings.from_mapping(_paper_mapping(), workflow_backend="sam3d")
    assert settings.profile == "paper"
    assert settings.scale_loss == "raw_mse"
    assert settings.chamfer.backend == "pytorch3d"

    with pytest.raises(ValueError, match="workflow.backend=sam3d"):
        ObjectiveSettings.from_mapping(_paper_mapping(), workflow_backend="direct")
    with pytest.raises(ValueError, match="chamfer.backend=pytorch3d"):
        ObjectiveSettings.from_mapping(_paper_mapping("torch"), workflow_backend="sam3d")

    portable = ObjectiveSettings.from_mapping(None, workflow_backend="direct")
    assert portable.scale_loss == "log_mse"
    assert not portable.chamfer.enabled


def test_sanitized_paper_config_matches_recovered_checkpoint_run() -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(
        OmegaConf.load(RELEASE_ROOT / "configs" / "pose_refiner_paper.yaml"),
        resolve=True,
    )
    assert isinstance(config, dict)
    assert config["workflow"]["backend"] == "sam3d"
    assert config["train"]["epochs"] == 200
    assert config["train"]["gradient_accumulation_steps"] == 2
    assert config["train"]["allow_tf32"] is True
    assert config["train"]["loss_weights"] == {
        "translation": 100.0,
        "rotation": 0.1,
        "scale": 100.0,
        "cd": 100.0,
    }
    assert config["train"]["objective"]["scale_loss"] == "raw_mse"
    assert config["train"]["objective"]["chamfer"]["backend"] == "pytorch3d"
    assert config["train"]["objective"]["chamfer"]["num_samples"] == 1024
    assert config["optimizer"]["lr"] == pytest.approx(1.0e-5)
    assert config["lr_scheduler"] == {
        "name": "cosine_warmup",
        "num_warmup_steps": 500,
        "total_steps": 909000,
    }


def test_canonical_points_follow_numeric_object_ids_and_fail_on_missing(tmp_path) -> None:
    points_zero = np.arange(18, dtype=np.float32).reshape(6, 3)
    points_three = points_zero + 100.0
    np.save(tmp_path / "obj_003_local.npy", points_three)
    np.save(tmp_path / "obj_0_local.npy", points_zero)

    loaded = load_canonical_points(
        tmp_path,
        (0, 3),
        source="points",
        point_count=4,
        sampling_seed=11,
    )
    # Existing pre-sampled NPY clouds are preserved rather than silently
    # resampled; canonical_point_count is only the PLY fallback sample count.
    assert loaded.shape == (2, 6, 3)
    assert float(loaded[0].max()) < 100.0
    assert float(loaded[1].min()) >= 100.0

    (tmp_path / "obj_003_local.npy").unlink()
    with pytest.raises(FileNotFoundError, match=r"missing=\[3\]"):
        load_canonical_points(tmp_path, (0, 3), source="points", point_count=4)


def test_mesh_sampling_is_deterministic_and_does_not_write_point_cache(tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        ),
        faces=np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
        process=False,
    )
    mesh.export(tmp_path / "obj_0_local.ply")
    files_before = {path.name for path in tmp_path.iterdir()}

    first = load_canonical_points(
        tmp_path, (0,), source="mesh", point_count=32, sampling_seed=5
    )
    second = load_canonical_points(
        tmp_path, (0,), source="mesh", point_count=32, sampling_seed=5
    )

    assert torch.equal(first, second)
    assert {path.name for path in tmp_path.iterdir()} == files_before
    assert not (tmp_path / "obj_0_local.npy").exists()


def test_torch_chamfer_uses_symmetric_squared_convention_and_has_gradients() -> None:
    canonical = torch.zeros(1, 1, 3)
    target = _identity_poses(1)
    predicted = target.clone()
    predicted[:, 0] = 1.0
    predicted.requires_grad_()

    loss = compute_pose_chamfer_loss(
        predicted,
        target,
        canonical,
        num_samples=1,
        backend="torch",
    )
    assert float(loss.detach()) == pytest.approx(2.0)
    loss.backward()
    assert predicted.grad is not None
    assert predicted.grad[0, 0] == pytest.approx(4.0)


def test_torch_chamfer_matches_pytorch3d_when_optional_backend_is_available() -> None:
    pytorch3d_loss = pytest.importorskip("pytorch3d.loss")
    generator = torch.Generator().manual_seed(13)
    first = torch.randn(2, 7, 3, generator=generator)
    second = torch.randn(2, 5, 3, generator=generator)
    expected, _ = pytorch3d_loss.chamfer_distance(first, second, batch_reduction="mean")
    assert torch.allclose(torch_chamfer_distance(first, second), expected)


def test_chamfer_rejects_invalid_canonical_tensors() -> None:
    poses = _identity_poses(1)
    with pytest.raises(ValueError, match=r"\[N, P, 3\]"):
        compute_pose_chamfer_loss(
            poses, poses, torch.zeros(1, 4, 2), num_samples=4, backend="torch"
        )
    invalid = torch.zeros(1, 4, 3)
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        compute_pose_chamfer_loss(
            poses, poses, invalid, num_samples=4, backend="torch"
        )


def test_paper_objective_can_be_dependency_injected_for_cpu_logic(tmp_path) -> None:
    np.save(
        tmp_path / "obj_0_local.npy",
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    )
    settings = ObjectiveSettings.from_mapping(_paper_mapping(), workflow_backend="sam3d")
    objective = TrainingObjective(
        settings,
        {"translation": 100.0, "rotation": 0.1, "scale": 100.0, "cd": 100.0},
        transform_fn=transform_points_by_pose,
        chamfer_fn=torch_chamfer_distance,
    )
    sample = SimpleNamespace(
        scene_id="cpu-paper-objective",
        object_ids=(0,),
        record={"sam3d_canonical_dir": str(tmp_path)},
    )
    target = _identity_poses(1)
    predicted = target.clone()
    predicted[:, 0] = 0.25
    predicted.requires_grad_()

    loss, components = objective(predicted, target, sample)
    assert set(components) == {
        "loss",
        "translation_loss",
        "rotation_loss",
        "scale_loss",
        "cd_loss",
    }
    assert float(components["cd_loss"]) > 0
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_canonical_cache_is_bounded_lru(tmp_path) -> None:
    settings = ObjectiveSettings.from_mapping(_paper_mapping(), workflow_backend="sam3d")
    settings = replace(settings, chamfer=replace(settings.chamfer, cache_size=2))
    objective = TrainingObjective(
        settings,
        {"translation": 1.0, "rotation": 1.0, "scale": 1.0, "cd": 1.0},
        transform_fn=transform_points_by_pose,
        chamfer_fn=torch_chamfer_distance,
    )
    poses = _identity_poses(1)
    directories = []
    for index in range(3):
        directory = tmp_path / f"scene_{index}"
        directory.mkdir()
        np.save(directory / "obj_0_local.npy", np.zeros((4, 3), dtype=np.float32))
        directories.append(directory.resolve())
        sample = SimpleNamespace(
            scene_id=f"scene-{index}",
            object_ids=(0,),
            record={"sam3d_canonical_dir": str(directory)},
        )
        objective(poses, poses, sample)

    assert len(objective._canonical_cache) == 2
    cached_directories = {key[0] for key in objective._canonical_cache}
    assert str(directories[0]) not in cached_directories
    assert cached_directories == {str(directories[1]), str(directories[2])}
