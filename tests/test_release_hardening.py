"""Regression coverage for release-facing failure and security behavior."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from multi_object_decoder import checkpoints
from scripts.dev.check_release import _POSIX_ABSOLUTE_RE, _check_manifest


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_release_path_scan_distinguishes_env_paths_from_private_paths() -> None:
    assert _POSIX_ABSOLUTE_RE.search("${DEMO_ROOT}/data/example.json") is None
    private_path = "/" + "home/author/private.json"
    assert _POSIX_ABSOLUTE_RE.search(private_path) is not None


def test_release_manifest_counts_must_match_scanned_tree(tmp_path: Path) -> None:
    (tmp_path / "docs/release").mkdir(parents=True)
    (tmp_path / "docs/release/manifest.json").write_text(
        json.dumps(
            {
                "inventory": {
                    "source_files": 8,
                    "python_files": 3,
                    "test_cases": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    findings: list[str] = []

    _check_manifest(
        tmp_path,
        findings,
        actual_file_count=9,
        actual_python_file_count=4,
    )

    assert findings == [
        "manifest inventory.source_files=8 does not match actual count 9",
        "manifest inventory.python_files=3 does not match actual count 4",
    ]


def _run_evaluator(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RELEASE_ROOT / "scripts" / "evaluate.py"), *arguments],
        cwd=RELEASE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _run_inference(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RELEASE_ROOT / "scripts" / "infer.py"), *arguments],
        cwd=RELEASE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _write_validator(path: Path) -> None:
    path.write_text(
        "def validate_predicted_poses(**kwargs):\n"
        "    return {'status': 'success', 'metrics': {}}\n",
        encoding="utf-8",
    )


def test_checkpoint_loader_refuses_unrestricted_pickle_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "untrusted.pt"
    checkpoint_path.write_bytes(b"placeholder")
    calls = []

    def unsupported_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("weights_only is unsupported")

    monkeypatch.setattr(checkpoints.torch, "load", unsupported_loader)
    with pytest.raises(checkpoints.CheckpointValidationError, match="refusing.*unsafe"):
        checkpoints.load_checkpoint(checkpoint_path)

    assert len(calls) == 1
    assert calls[0][1]["weights_only"] is True


def test_inference_rejects_scene_directory_collision_before_writing(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "colliding.json"
    manifest.write_text(
        json.dumps([{"scene_id": "a/b"}, {"scene_id": "a__b"}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "inference"

    result = _run_inference(
        "--checkpoint",
        str(tmp_path / "missing.pt"),
        "--data-json",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
    )

    assert result.returncode != 0
    assert "map to the same output directory" in result.stderr
    assert not output_dir.exists()


def test_evaluator_returns_two_when_no_entries_are_selected(tmp_path: Path) -> None:
    config = tmp_path / "empty.json"
    config.write_text("[]\n", encoding="utf-8")

    result = _run_evaluator("--config", str(config), "--workers", "1")

    assert result.returncode == 2
    assert "No valid evaluation entries" in result.stderr


def test_evaluator_returns_one_when_any_entry_fails(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    config = tmp_path / "one.json"
    config.write_text(
        json.dumps([{"id": "missing-gt", "sam3d_results_path": str(output_dir)}]),
        encoding="utf-8",
    )
    validator = tmp_path / "validator.py"
    _write_validator(validator)

    result = _run_evaluator(
        "--config",
        str(config),
        "--validator-script",
        str(validator),
        "--workers",
        "1",
    )

    assert result.returncode == 1
    assert "Evaluation failed for 1/1 entries" in result.stderr


def test_evaluator_returns_zero_only_when_all_entries_succeed(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "metrics.txt").write_text("{}\n", encoding="utf-8")
    config = tmp_path / "one.json"
    config.write_text(
        json.dumps([{"id": "complete", "sam3d_results_path": str(output_dir)}]),
        encoding="utf-8",
    )
    validator = tmp_path / "validator.py"
    _write_validator(validator)

    result = _run_evaluator(
        "--config",
        str(config),
        "--validator-script",
        str(validator),
        "--workers",
        "1",
        "--skip-existing",
    )

    assert result.returncode == 0, result.stderr
    assert "Finished: 1/1 entries evaluated" in result.stderr


def _minimal_release(root: Path) -> Path:
    from scripts.dev.check_release import REQUIRED_RELEASE_PATHS

    for relative, kind in REQUIRED_RELEASE_PATHS.items():
        target = root / relative
        if kind == "directory":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("Release fixture\n", encoding="utf-8")
    (root / "tests" / "test_smoke.py").write_text("def test_smoke(): pass\n")
    (root / "scripts" / "entry.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "scripts" / "entry.sh").chmod(0o755)
    (root / "docs/release/manifest.json").write_text(
        json.dumps({"schema_version": 3, "release": {"status": "demo_only"}})
    )
    (root / "assets/demo" / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "assets": []})
    )
    return root


def _write_demo_asset(root: Path, name: str = "cache/sample.npz") -> dict:
    path = root / "assets/demo" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytes([0, 255, 17, 23])
    path.write_bytes(data)
    entry = {
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "license": "CC0-1.0",
        "source": "Generated fixture from test_release_hardening",
    }
    (root / "assets/demo" / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "assets": [entry]})
    )
    return entry


def _run_release(root: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RELEASE_ROOT / "scripts" / "dev" / "check_release.py"), str(root), *flags],
        capture_output=True, text=True, check=False, timeout=30,
    )


def _commit_fixture(root: Path) -> None:
    for command in (
        ["init", "--quiet"],
        ["add", "--all"],
        ["-c", "user.name=Release Tests", "-c", "user.email=tests@example.invalid",
         "commit", "--quiet", "-m", "Release test fixture"],
    ):
        subprocess.run(["git", "-C", str(root), *command], check=True, capture_output=True)


def test_static_release_accepts_verified_demo_binary_without_git(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    _write_demo_asset(root)
    result = _run_release(root)
    assert result.returncode == 0, result.stderr
    assert not (root / ".git").exists()


@pytest.mark.parametrize("mutation", ["unlisted", "tampered", "size", "traversal", "wildcard"])
def test_release_rejects_invalid_demo_asset(tmp_path: Path, mutation: str) -> None:
    root = _minimal_release(tmp_path)
    entry = _write_demo_asset(root)
    manifest = root / "assets/demo" / "MANIFEST.json"
    if mutation == "unlisted":
        manifest.write_text(json.dumps({"schema_version": 1, "assets": []}))
    elif mutation == "tampered":
        (root / "assets/demo" / entry["path"]).write_bytes(bytes([0, 255, 18, 23]))
    else:
        if mutation == "size":
            entry["size_bytes"] += 1
        elif mutation == "traversal":
            entry["path"] = "../README.md"
        else:
            entry["path"] = "cache/*"
        manifest.write_text(json.dumps({"schema_version": 1, "assets": [entry]}))
    result = _run_release(root)
    assert result.returncode == 1
    expected = {
        "unlisted": "binary/checkpoint artifact",
        "tampered": "sha256 mismatch",
        "size": "size mismatch",
        "traversal": "unsafe demo asset path",
        "wildcard": "unsafe demo asset path",
    }
    assert expected[mutation] in result.stderr


def test_demo_manifest_cannot_allow_symlink_directory(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path / "release")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sample.npz").write_bytes(bytes([0, 255, 17, 23]))
    (root / "assets/demo" / "cache").symlink_to(outside, target_is_directory=True)
    entry = {
        "path": "cache/sample.npz",
        "sha256": hashlib.sha256((outside / "sample.npz").read_bytes()).hexdigest(),
        "size_bytes": 4,
        "license": "CC0-1.0",
        "source": "Generated test fixture",
    }
    (root / "assets/demo" / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "assets": [entry]})
    )
    result = _run_release(root)
    assert result.returncode == 1
    assert "symbolic link" in result.stderr


def test_demo_text_still_scanned_when_manifested(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    entry = _write_demo_asset(root, "details.txt")
    data = ("/" + "home/author/private.json\n").encode()
    (root / "assets/demo" / entry["path"]).write_bytes(data)
    entry.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    (root / "assets/demo" / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "assets": [entry]})
    )
    result = _run_release(root)
    assert result.returncode == 1
    assert "private/cluster absolute path: assets/demo/details.txt" in result.stderr


def test_demo_asset_limits_are_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.dev import check_release as release_check

    root = _minimal_release(tmp_path)
    _write_demo_asset(root)
    monkeypatch.setattr(release_check, "MAX_DEMO_ASSET_BYTES", 3)
    findings, _, _ = release_check.check_release(root)
    assert any("exceeds 95 MiB" in finding for finding in findings)
    monkeypatch.setattr(release_check, "MAX_DEMO_ASSET_BYTES", 8)
    monkeypatch.setattr(release_check, "MAX_DEMO_TOTAL_BYTES", 3)
    findings, _, _ = release_check.check_release(root)
    assert "demo assets exceed 256 MiB total" in findings


@pytest.mark.parametrize("mutation", ["untracked", "modified", "staged"])
def test_require_git_rejects_uncommitted_payload(tmp_path: Path, mutation: str) -> None:
    root = _minimal_release(tmp_path)
    _commit_fixture(root)
    clean = _run_release(root, "--require-git")
    assert clean.returncode == 0, clean.stderr
    assert "Git HEAD:" in clean.stdout
    path = root / ("extra.txt" if mutation == "untracked" else "README.md")
    path.write_text("New release content\n")
    if mutation == "staged":
        subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
    result = _run_release(root, "--require-git")
    assert result.returncode == 1
    assert "untracked release payload" in result.stderr if mutation == "untracked" else (
        "clean Git index and working tree" in result.stderr
    )


def test_require_git_rejects_distribution_without_git(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    result = _run_release(root, "--require-git")
    assert result.returncode == 1
    assert "Git working tree root" in result.stderr


def test_paper_exact_rejects_current_unpublished_release_artifacts(tmp_path: Path) -> None:
    root = _minimal_release(tmp_path)
    payload = json.loads((RELEASE_ROOT / "docs/release/manifest.json").read_text())
    payload.pop("inventory", None)
    payload["release"]["status"] = "demo_only"
    checkpoint = payload["paper_artifacts"]["mod_checkpoint"]
    checkpoint.update(public_url=None, revision=None, license=None)
    payload["paper_artifacts"]["full_evaluator"]["status"] = "unpublished"
    (root / "docs/release/manifest.json").write_text(json.dumps(payload))
    default = _run_release(root)
    assert default.returncode == 0, default.stderr
    exact = _run_release(root, "--paper-exact")
    assert exact.returncode == 1
    for missing in ("mod_checkpoint.public_url", "mod_checkpoint.revision",
                    "mod_checkpoint.license", "full_evaluator is not published"):
        assert missing in exact.stderr
