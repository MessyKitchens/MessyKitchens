"""Independent bundled-data, training, and inference CPU demo coverage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _run_demo_stage(
    stage: str,
    demo_root: Path,
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(RELEASE_ROOT / "scripts/demo.sh"),
            stage,
            "--cpu-smoke",
            "--demo-dir",
            str(demo_root),
            *arguments,
        ],
        cwd=demo_root.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_independent_demo_stages_run_inference_before_training_on_cpu(tmp_path: Path) -> None:
    demo_root = tmp_path / "demo"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MOD_PYTHON": sys.executable,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    stages = (
        # A fresh output directory: inference must use the bundled checkpoint.
        ("infer", ()),
        ("train", ("--max-train-steps", "1")),
        ("prepare", ()),
    )
    for stage, arguments in stages:
        result = _run_demo_stage(
            stage,
            demo_root,
            environment,
            *arguments,
        )
        assert result.returncode == 0, (
            f"{stage} stdout:\n{result.stdout}\n"
            f"{stage} stderr:\n{result.stderr}"
        )
    assert (demo_root / "data" / "prepared_data.json").is_file()
    assert (demo_root / "training" / "checkpoints" / "last.pt").is_file()

    inference_root = demo_root / "inference"
    manifest = json.loads(
        (inference_root / "inference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["successful"] == 2
    assert manifest["failed"] == 0
    assert manifest["skipped"] == 0

    expected_objects = {
        "demo__scene_000": (0, 2),
        "demo__scene_001": (0, 1, 4),
    }
    for scene_directory, object_ids in expected_objects.items():
        scene_root = inference_root / scene_directory
        assert (scene_root / "mod_pose.json").is_file()
        assert (scene_root / "posed_meshes" / "scene.glb").is_file()
        for object_id in object_ids:
            assert (
                scene_root / "posed_meshes" / f"obj_{object_id}.ply"
            ).is_file()


def test_demo_help_needs_no_model_environment(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MOD_PYTHON"] = str(tmp_path / "missing-python")
    for stage in ("prepare", "train", "infer"):
        result = subprocess.run([str(RELEASE_ROOT / "scripts/demo.sh"), stage, "--help"],
                                cwd=tmp_path, env=environment, capture_output=True,
                                text=True, check=False, timeout=15)
        assert result.returncode == 0, result.stderr
        assert "--cpu-smoke" in result.stdout


def test_inference_default_does_not_select_a_previous_training_run(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MOD_PYTHON"] = sys.executable
    previous = tmp_path / "training" / "checkpoints"
    previous.mkdir(parents=True)
    (previous / "last.pt").write_bytes(b"invalid prior checkpoint")
    result = _run_demo_stage("infer", tmp_path, environment, "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert str(RELEASE_ROOT / "assets/demo" / "portable" / "model.pt") in payload["inputs"]
    assert str(previous / "last.pt") not in payload["inputs"]


def test_cpu_preparation_accepts_an_existing_empty_directory(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MOD_PYTHON"] = sys.executable
    output = tmp_path / "empty"
    output.mkdir()
    result = _run_demo_stage("prepare", tmp_path, environment,
                             "--output-dir", str(output))
    assert result.returncode == 0, result.stderr
    assert (output / "prepared_data.json").is_file()


def test_cpu_preparation_rejects_an_ignored_custom_manifest(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MOD_PYTHON"] = sys.executable
    result = _run_demo_stage("prepare", tmp_path, environment,
                             "--data-json", str(tmp_path / "missing.json"), "--dry-run")
    assert result.returncode == 2
    assert "--data-json is not supported" in result.stderr
