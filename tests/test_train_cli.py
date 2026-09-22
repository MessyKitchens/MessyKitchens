"""Training manifests keep explicit CLI and config-relative path semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("manifest_source", ["cli", "config", "dotlist"])
def test_training_manifest_paths_from_another_working_directory(
    tmp_path: Path, manifest_source: str,
) -> None:
    working_dir = tmp_path / "run"
    config_dir = tmp_path / "settings"
    working_dir.mkdir()
    config_dir.mkdir()
    data_dir = (working_dir if manifest_source == "cli" else config_dir) / "data"
    shutil.copytree(ROOT / "assets/demo/portable/prepared", data_dir)
    relative_manifest = "data/prepared_data.json"
    manifest = data_dir / "prepared_data.json"
    config = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/demo_direct.yaml"))
    config["train"]["data_json"] = (
        relative_manifest if manifest_source == "config" else "missing.json"
    )
    config_path = config_dir / "train.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = working_dir / "trained"
    command = [
        sys.executable, str(ROOT / "scripts/train.py"),
        "--config", str(config_path), "--output-dir", str(output_dir),
        "--device", "cpu", "--max-train-steps", "1",
    ]
    if manifest_source == "cli":
        # The named argument must also take precedence over a dot-list value.
        command += ["--data-json", relative_manifest, "train.data_json=also_missing.json"]
    elif manifest_source == "dotlist":
        command += [f"train.data_json={relative_manifest}"]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    result = subprocess.run(
        command, cwd=working_dir, env=environment, capture_output=True,
        text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["global_step"] == 1
    assert (output_dir / "checkpoints/last.pt").is_file()
    resolved = json.loads((output_dir / "resolved_config.json").read_text())
    assert resolved["train"]["data_json"] == (
        str(manifest) if manifest_source == "cli" else relative_manifest
    )
