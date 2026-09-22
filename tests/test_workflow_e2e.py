"""CPU-only train/checkpoint/infer coverage for the portable MOD workflow."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from multi_object_decoder.checkpoints import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
)


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _assert_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    assert result.returncode == 0, (
        f"{label} exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_portable_cpu_train_checkpoint_infer_workflow(tmp_path: Path) -> None:
    object_count, feature_dim, shape_tokens = 2, 8, 4
    # The cache is deliberately non-contiguous and out of numeric order.
    object_ids = np.asarray([3, 0], dtype=np.int64)
    rng = np.random.default_rng(17)

    archive: dict[str, np.ndarray] = {"object_ids": object_ids}
    for key in (
        "6drotation_normalized",
        "translation",
        "scale",
        "translation_scale",
    ):
        archive[f"pose_latents_raw__{key}"] = rng.normal(
            size=(object_count, 1, feature_dim)
        ).astype(np.float32)
    archive["pose_latents_raw__shape"] = rng.normal(
        size=(object_count, shape_tokens, feature_dim)
    ).astype(np.float32)

    x_t_dimensions = {
        "6drotation_normalized": (1, 6),
        "translation": (1, 3),
        "scale": (1, 3),
        "translation_scale": (1, 1),
        "shape": (shape_tokens, 8),
    }
    for key, (token_count, output_dim) in x_t_dimensions.items():
        archive[f"x_t_last_step__{key}"] = rng.normal(
            size=(object_count, token_count, output_dim)
        ).astype(np.float32)

    archive["pointmap_scale"] = np.asarray(
        [[1.3, 1.3, 1.3], [0.8, 0.9, 1.0]], dtype=np.float32
    )
    archive["pointmap_shift"] = np.asarray(
        [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    archive["solver_state__t"] = np.asarray([0.75, 0.25], dtype=np.float32)
    archive["solver_state__dt"] = np.asarray([0.05, 0.05], dtype=np.float32)
    archive["image_condition_per_object"] = rng.normal(
        size=(object_count, 1, feature_dim)
    ).astype(np.float32)

    # Rows still follow the on-disk [3, 0] order. The large separation makes a
    # row/ID mismatch observable after the loader sorts IDs to [0, 3].
    base_poses = np.asarray(
        [
            [30.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.3, 1.3, 1.3],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8, 0.9, 1.0],
        ],
        dtype=np.float32,
    )
    target_poses = base_poses.copy()
    target_poses[:, :3] += np.asarray([0.2, -0.1, 0.05], dtype=np.float32)
    archive["base_poses"] = base_poses
    archive["target_poses"] = target_poses

    scene_path = tmp_path / "portable_scene.npz"
    np.savez(scene_path, **archive)
    with np.load(scene_path, allow_pickle=False) as portable_scene:
        assert set(portable_scene.files) == set(archive)
        assert all(portable_scene[key].dtype != object for key in portable_scene.files)

    scene_id = "portable/non_contiguous_ids"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "scene_id": scene_id,
                    "scene_npz_path": scene_path.name,
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "workflow": {"backend": "direct"},
        "model": {
            "feature_dim": feature_dim,
            "input_dim": feature_dim,
            "num_heads": 2,
            "mlp_hidden_dim": 16,
            "dropout": 0.0,
            "num_layers": 1,
            "use_global_attn": True,
            "use_layer_norm": True,
            "pose_head_hidden_dim": 16,
            "use_shape_tokens": True,
        },
        "train": {
            "data_json": manifest_path.name,
            "epochs": 2,
            "gradient_accumulation_steps": 1,
            "precision": "fp32",
            "seed": 23,
            "save_freq": 0,
            "max_grad_norm": 1.0,
            "loss_weights": {
                "translation": 1.0,
                "rotation": 1.0,
                "scale": 1.0,
            },
        },
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0,
            "eps": 1e-8,
            "betas": [0.9, 0.999],
        },
        "lr_scheduler": {"num_warmup_steps": 0},
    }
    config_path = tmp_path / "smoke_config.yaml"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    train_output = tmp_path / "training"
    train_result = _run(
        [
            sys.executable,
            str(RELEASE_ROOT / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(train_output),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--seed",
            "23",
            "--max-train-steps",
            "2",
        ],
        environment,
    )
    _assert_success(train_result, "training")

    checkpoint_path = train_output / "checkpoints" / "last.pt"
    assert checkpoint_path.is_file()
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_backend="direct",
    )
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["backend"] == "direct"
    assert checkpoint["model_config"]["feature_dim"] == feature_dim
    assert checkpoint["model_config"]["input_dim"] == feature_dim
    assert checkpoint["model_config"]["num_heads"] == 2
    assert checkpoint["model_config"]["use_shape_tokens"] is True
    assert checkpoint["state_dict"]
    assert checkpoint["optimizer"] is not None
    assert checkpoint["training_state"]["global_step"] == 2
    assert checkpoint["training_state"]["epoch"] == 2
    assert "scheduler" in checkpoint["training_state"]

    inference_output = tmp_path / "inference"
    inference_result = _run(
        [
            sys.executable,
            str(RELEASE_ROOT / "scripts" / "infer.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--data-json",
            str(manifest_path),
            "--output-dir",
            str(inference_output),
            "--device",
            "cpu",
            "--overwrite",
        ],
        environment,
    )
    _assert_success(inference_result, "inference")

    pose_path = inference_output / "portable__non_contiguous_ids" / "mod_pose.json"
    pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
    assert list(pose_payload["objects"]) == ["obj_0", "obj_3"]
    assert pose_payload["metadata"] == {
        "backend": "direct",
        "checkpoint": "last.pt",
        "scene_id": scene_id,
    }
    object_zero_x = float(pose_payload["objects"]["obj_0"]["translation"][0])
    object_three_x = float(pose_payload["objects"]["obj_3"]["translation"][0])
    assert math.isfinite(object_zero_x) and math.isfinite(object_three_x)
    assert object_three_x - object_zero_x > 20.0

    inference_manifest = json.loads(
        (inference_output / "inference_manifest.json").read_text(encoding="utf-8")
    )
    assert inference_manifest["backend"] == "direct"
    assert inference_manifest["successful"] == 1
    assert inference_manifest["failed"] == 0
    assert inference_manifest["skipped"] == 0
    assert len(inference_manifest["scenes"]) == 1
    scene_result = inference_manifest["scenes"][0]
    assert scene_result["scene_id"] == scene_id
    assert scene_result["status"] == "success"
    assert scene_result["num_objects"] == object_count
    assert scene_result["outputs"] == [str(pose_path)]
