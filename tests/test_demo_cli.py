"""Model-free checks for the demo training-GT preparation interface."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/demo.py"), *arguments],
        capture_output=True, text=True, check=False, timeout=15,
    )


def _raw_input(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("rgb.png", "mask.png", "gt.glb"):
        (raw / name).write_bytes(b"unused by dry-run")
    source = raw / "input.json"
    source.write_text(json.dumps([{
        "scene_id": "example", "image_path": "rgb.png",
        "instance_mask_path": "mask.png", "mesh_path": "gt.glb",
        "gt_object_map": {"0": "object_A", "3": "object_B"},
        "target_poses_path": "historical.json", "gt_poses_path": "historical.json",
    }]), encoding="utf-8")
    mapping = raw / "mapping.json"
    mapping.write_text(json.dumps({"0": "object_A", "3": "object_B"}), encoding="utf-8")
    return source, raw / "gt.glb", mapping


def _prepare_arguments(source: Path, output: Path) -> list[str]:
    return ["prepare", "--data-json", str(source), "--output-dir", str(output),
            "--sam3d-root", str(output.parent / "sam3d"),
            "--sam3d-config", str(output.parent / "pipeline.yaml")]


def test_gt_dry_run_rebases_raw_paths_and_does_not_reuse_historical_targets(tmp_path: Path) -> None:
    source, mesh, mapping = _raw_input(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "prepared"
    result = _run(*_prepare_arguments(source, output), "--generate-training-gt",
                  "--gt-mesh", str(mesh), "--gt-object-map", str(mapping),
                  "--gaps-binary", str(tmp_path / "mshalign"), "--gt-timeout", "30",
                  "--allow-legacy-pickle", "--dry-run")
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    derived = receipt["generated_input"]["records"][0]
    assert derived["image_path"] == str(source.parent / "rgb.png")
    assert derived["instance_mask_path"] == str(source.parent / "mask.png")
    assert derived["mesh_path"] == str(mesh)
    assert derived["gt_object_map"] == {"0": "object_A", "3": "object_B"}
    assert "target_poses_path" not in derived
    assert "gt_poses_path" not in derived
    command = receipt["command"]
    assert command[command.index("--data-json") + 1] == str(output / "_inputs/training_gt_input.json")
    assert command[command.index("--gt-timeout") + 1] == "30.0"
    assert command[command.index("--gaps-binary") + 1] == str(tmp_path / "mshalign")
    assert "--allow-legacy-pickle" in command
    assert "--generate-training-gt" in command
    assert source.read_bytes() == original
    assert not output.exists()


def test_gt_can_use_per_record_manifest_fields_without_global_overrides(tmp_path: Path) -> None:
    source, _, _ = _raw_input(tmp_path)
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows.append({**rows[0], "scene_id": "second"})
    source.write_text(json.dumps(rows), encoding="utf-8")
    result = _run(*_prepare_arguments(source, tmp_path / "prepared"),
                  "--generate-training-gt", "--dry-run")
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert len(receipt["generated_input"]["records"]) == 2
    assert receipt["command"][receipt["command"].index("--gt-timeout") + 1] == "120.0"


def test_gt_writes_derived_manifest_before_preparation_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source, _, _ = _raw_input(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "prepared"
    spec = importlib.util.spec_from_file_location("demo_gt_cli", ROOT / "scripts/demo.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        manifest = Path(command[command.index("--data-json") + 1])
        assert manifest == output / "_inputs/training_gt_input.json"
        assert manifest.is_file()
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        assert "target_poses_path" not in rows[0]
        assert rows[0]["mesh_path"] == str(source.parent / "gt.glb")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["demo.py", *_prepare_arguments(source, output),
                                     "--generate-training-gt"])
    assert module.main() == 0
    capsys.readouterr()
    assert (output / "demo_run.json").is_file()
    assert source.read_bytes() == original


@pytest.mark.parametrize("arguments, message", [
    (["prepare", "--cpu-smoke", "--generate-training-gt"], "without --cpu-smoke"),
    (["train", "--generate-training-gt"], "SAM3D prepare stage"),
    (["prepare", "--gt-mesh", "missing.glb"], "require --generate-training-gt"),
    (["prepare", "--generate-training-gt", "--gt-timeout", "nan"], "positive finite"),
    (["prepare", "--generate-training-gt", "--gt-timeout", "0"], "positive finite"),
])
def test_gt_rejects_misleading_options(arguments: list[str], message: str) -> None:
    result = _run(*arguments, "--dry-run")
    assert result.returncode == 2
    assert message in result.stderr


def test_gt_requires_user_supplied_ground_truth_for_bundled_demo(tmp_path: Path) -> None:
    result = _run("prepare", "--output-dir", str(tmp_path / "prepared"),
                  "--generate-training-gt", "--dry-run")
    assert result.returncode == 2
    assert "gt_object_map" in result.stderr


def test_gt_rejects_duplicate_gt_correspondence(tmp_path: Path) -> None:
    source, _, _ = _raw_input(tmp_path)
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows[0]["gt_object_map"] = {"0": "same_node", "1": "same_node"}
    source.write_text(json.dumps(rows), encoding="utf-8")
    result = _run(*_prepare_arguments(source, tmp_path / "prepared"),
                  "--generate-training-gt", "--dry-run")
    assert result.returncode == 2
    assert "one-to-one" in result.stderr


def test_training_forwards_explicit_cache_trust_option(tmp_path: Path) -> None:
    result = _run("train", "--output-dir", str(tmp_path / "training"),
                  "--sam3d-root", str(tmp_path / "sam3d"),
                  "--sam3d-config", str(tmp_path / "pipeline.yaml"),
                  "--allow-legacy-pickle", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--allow-legacy-pickle" in json.loads(result.stdout)["command"]


@pytest.mark.parametrize("name", ["prepared_data.json", "prepare_manifest.json", "demo_run.json"])
def test_gt_rejects_source_manifest_that_would_be_replaced_by_output(tmp_path: Path, name: str) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    source = output / name
    source.write_text("[{\"scene_id\": \"keep\"}]\n", encoding="utf-8")
    original = source.read_bytes()
    result = _run(*_prepare_arguments(source, output), "--generate-training-gt", "--dry-run")
    assert result.returncode == 2
    assert "source manifest must not be a generated output" in result.stderr
    assert source.read_bytes() == original
    assert not (output / "_inputs").exists()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("field", ["target_poses_path", "gt_poses_path"])
def test_gt_preserves_original_targets_inside_output_even_when_references_are_removed(
    tmp_path: Path, overwrite: bool, field: str,
) -> None:
    source, _, _ = _raw_input(tmp_path)
    output = tmp_path / "prepared"
    target = output / "example" / "target_poses.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"keep original supervision")
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows[0][field] = str(target)
    source.write_text(json.dumps(rows), encoding="utf-8")
    arguments = [*_prepare_arguments(source, output), "--generate-training-gt", "--dry-run"]
    if overwrite:
        arguments.append("--overwrite")
    result = _run(*arguments)
    assert result.returncode == 2
    assert f"source {field} is inside the GT output" in result.stderr
    assert target.read_bytes() == b"keep original supervision"
    assert not (output / "_inputs").exists()


def test_gt_overwrite_preserves_original_manifest_hidden_by_derived_copy(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    source = output / "example" / "input.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"keep original input")
    result = _run(*_prepare_arguments(source, output), "--generate-training-gt",
                  "--overwrite", "--dry-run")
    assert result.returncode == 2
    assert "source manifest is inside --output-dir" in result.stderr
    assert source.read_bytes() == b"keep original input"
