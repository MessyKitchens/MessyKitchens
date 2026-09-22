#!/usr/bin/env python3
"""Reject private paths, credentials, generated artifacts, and invalid Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Sequence, Tuple


TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".gitignore",
    ".ini",
    ".in",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REQUIRED_RELEASE_PATHS = {
    "README.md": "file",
    "LICENSE": "file",
    "NOTICE": "file",
    "CITATION.cff": "file",
    "THIRD_PARTY_NOTICES.md": "file",
    "pyproject.toml": "file",
    "docs/release/manifest.json": "file",
    "assets/demo/MANIFEST.json": "file",
    "MANIFEST.in": "file",
    "tests": "directory",
    "scripts": "directory",
}
CACHE_DIR_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "wandb",
}
GENERATED_DIR_NAMES = {"build", "dist", "checkpoints", "outputs", "runs"}
BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bz2",
    ".ckpt",
    ".dll",
    ".dylib",
    ".egg",
    ".gz",
    ".joblib",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".so",
    ".tar",
    ".xz",
    ".zip",
}

MAX_DEMO_ASSET_BYTES = 95 * 1024 * 1024
MAX_DEMO_TOTAL_BYTES = 256 * 1024 * 1024

_PRIVATE_POSIX_PREFIXES = tuple(
    "/" + value
    for value in (
        "cluster/",
        "data/",
        "gpfs/",
        "home/",
        "lustre/",
        "mnt/",
        "nfs/",
        "nfs-stor/",
        "scratch/",
        "workspace/",
        "work/",
    )
)
_POSIX_ABSOLUTE_RE = re.compile(
    # A slash after ``}`` belongs to a braced environment-variable expansion
    # such as ``${DEMO_ROOT}/data``; it is not a hard-coded absolute path.
    r"(?<![A-Za-z0-9_.}-])(?:"
    + "|".join(re.escape(value) for value in _PRIVATE_POSIX_PREFIXES)
    + ")"
)
_WINDOWS_ABSOLUTE_RE = re.compile(
    r"\b[A-Za-z]:[\\/](?:Users|data|mnt|scratch|workspace)[\\/]",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|auth[_-]?token|access[_-]?token|password|passwd|"
    r"secret(?:[_-]?key)?|wandb[_-]?api[_-]?key)\b\s*[:=]\s*"
    r"(?P<quote>['\"])(?P<value>[^'\"]+)(?P=quote)",
    re.IGNORECASE,
)
_KNOWN_SECRET_RES = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile("-----BEGIN " + "PRIVATE KEY-----"),
)
_PLACEHOLDER_VALUES = {
    "changeme",
    "dummy",
    "example",
    "none",
    "redacted",
    "replace_me",
    "test",
}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        normalized in _PLACEHOLDER_VALUES
        or normalized.startswith(("${", "<", "your_", "your-"))
        or normalized.endswith(">")
    )


def _iter_release_files(root: Path, findings: List[str]) -> Iterable[Path]:
    for current, directory_names, file_names in os.walk(root):
        current_path = Path(current)
        retained_directories = []
        for directory_name in sorted(directory_names):
            directory_path = current_path / directory_name
            if directory_name == ".git":
                # Git internals are not part of the release payload and may
                # contain arbitrary binary objects or historical credentials.
                continue
            if directory_path.is_symlink():
                findings.append(f"symbolic link: {_relative(directory_path, root)}")
                continue
            if directory_name in CACHE_DIR_NAMES or directory_name in GENERATED_DIR_NAMES:
                findings.append(f"generated/cache directory: {_relative(directory_path, root)}")
            elif directory_name.endswith(".egg-info"):
                findings.append(f"generated package metadata: {_relative(directory_path, root)}")
            else:
                retained_directories.append(directory_name)
        directory_names[:] = retained_directories
        for file_name in sorted(file_names):
            if file_name == ".git":
                # Also handle a worktree/submodule-style .git pointer file.
                continue
            yield current_path / file_name


def _check_required_release_paths(root: Path, findings: List[str]) -> None:
    for relative_path, expected_kind in REQUIRED_RELEASE_PATHS.items():
        path = root / relative_path
        if expected_kind == "file" and not path.is_file():
            findings.append(f"missing required release file: {relative_path}")
        elif expected_kind == "directory" and not path.is_dir():
            findings.append(f"missing required release directory: {relative_path}/")

    tests_dir = root / "tests"
    if tests_dir.is_dir() and not any(tests_dir.glob("test_*.py")):
        findings.append("required tests/ directory contains no test_*.py files")


def _check_shell_entrypoints(root: Path, findings: List[str]) -> None:
    root_scripts = sorted((root / "scripts").glob("*.sh"))
    if not root_scripts:
        findings.append("no scripts/ shell entrypoints found")
        return
    for path in root_scripts:
        try:
            mode = path.stat().st_mode
        except OSError as error:
            findings.append(f"unreadable scripts/ shell entrypoint: {path.name}: {error}")
            continue
        if mode & 0o111 == 0:
            findings.append(f"scripts/ shell entrypoint is not executable: {path.name}")


def _check_manifest_path(
    raw_path: object,
    *,
    root: Path,
    location: str,
    findings: List[str],
) -> None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        findings.append(f"invalid manifest file path at {location}")
        return
    normalized = PurePosixPath(raw_path)
    if normalized.is_absolute() or ".." in normalized.parts:
        findings.append(f"unsafe manifest file path at {location}: {raw_path}")
        return
    if not (root / normalized).exists():
        findings.append(f"manifest references missing path at {location}: {raw_path}")


def _check_manifest(
    root: Path,
    findings: List[str],
    *,
    actual_file_count: int | None = None,
    actual_python_file_count: int | None = None,
) -> None:
    manifest_path = root / "docs/release/manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        findings.append(f"invalid docs/release/manifest.json: {error}")
        return
    if not isinstance(payload, dict):
        findings.append("docs/release/manifest.json must contain a JSON object")
        return

    inventory = payload.get("inventory")
    if inventory is None:
        # Early release candidates may carry metadata before their generated
        # file inventory is frozen. Once present, the inventory is validated
        # below without requiring a particular manifest schema revision.
        return
    if not isinstance(inventory, (dict, list)):
        findings.append("docs/release/manifest.json must contain an inventory object or list")
        return

    declared_lists: List[Tuple[str, object]] = []
    count_fields_seen = False
    if isinstance(inventory, list):
        declared_lists.append(("inventory", inventory))
    else:
        for key in ("files", "paths", "source_paths"):
            if key in inventory:
                declared_lists.append((f"inventory.{key}", inventory[key]))
        actual_counts = {
            "source_files": actual_file_count,
            "python_files": actual_python_file_count,
        }
        for key in ("source_files", "python_files", "test_cases"):
            if key not in inventory:
                continue
            count_fields_seen = True
            value = inventory[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                findings.append(f"manifest inventory.{key} must be a non-negative integer")
                continue
            actual = actual_counts.get(key)
            if actual is not None and value != actual:
                findings.append(
                    f"manifest inventory.{key}={value} does not match actual count {actual}"
                )

    groups = payload.get("artifact_groups")
    if groups is not None:
        if not isinstance(groups, list):
            findings.append("manifest artifact_groups must be a list")
        else:
            for group_index, group in enumerate(groups):
                if not isinstance(group, dict):
                    findings.append(f"manifest artifact_groups[{group_index}] must be an object")
                    continue
                if "files" in group:
                    declared_lists.append(
                        (f"artifact_groups[{group_index}].files", group["files"])
                    )

    if not declared_lists and not count_fields_seen:
        findings.append("manifest inventory does not declare files or count fields")
        return

    for location, values in declared_lists:
        if not isinstance(values, list):
            findings.append(f"manifest {location} must be a list")
            continue
        seen = set()
        for index, raw_path in enumerate(values):
            if isinstance(raw_path, str) and raw_path in seen:
                findings.append(f"duplicate manifest path at {location}[{index}]: {raw_path}")
            elif isinstance(raw_path, str):
                seen.add(raw_path)
            _check_manifest_path(
                raw_path,
                root=root,
                location=f"{location}[{index}]",
                findings=findings,
            )



def _check_demo_assets(root: Path, findings: List[str]) -> set[str]:
    """Permit only individually declared, verified files under assets/demo/."""
    manifest = root / "assets/demo" / "MANIFEST.json"
    allowed: set[str] = set()
    if not manifest.is_file() or manifest.is_symlink():
        return allowed
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        findings.append(f"invalid assets/demo/MANIFEST.json: {error}")
        return allowed
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        findings.append("assets/demo/MANIFEST.json requires schema_version 1")
        return allowed
    assets = payload.get("assets")
    if not isinstance(assets, list):
        findings.append("assets/demo/MANIFEST.json assets must be a list")
        return allowed

    seen: set[str] = set()
    total_bytes = 0
    demo_root = root / "assets/demo"
    for index, asset in enumerate(assets):
        location = f"assets/demo/MANIFEST.json assets[{index}]"
        if not isinstance(asset, dict):
            findings.append(f"{location} must be an object")
            continue
        before = len(findings)
        raw_path = asset.get("path")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or any(char in raw_path for char in "\\:*?[]")
            or raw_path.startswith("/")
            or any(part in ("", ".", "..") for part in raw_path.split("/"))
        ):
            findings.append(f"unsafe demo asset path at {location}: {raw_path!r}")
            continue
        if raw_path in seen:
            findings.append(f"duplicate demo asset path: {raw_path}")
            continue
        seen.add(raw_path)
        path = demo_root / PurePosixPath(raw_path)
        components = [demo_root, *path.parents]
        if path.is_symlink() or any(
            part.is_symlink() for part in components if part != root and root in part.parents
        ):
            findings.append(f"symbolic link in demo asset path: {raw_path}")
            continue
        if not path.is_file():
            findings.append(f"missing demo asset file: {raw_path}")
            continue

        digest = asset.get("sha256")
        size = asset.get("size_bytes")
        license_id = asset.get("license")
        source = asset.get("source")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            findings.append(f"invalid demo asset sha256: {raw_path}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            findings.append(f"invalid demo asset size_bytes: {raw_path}")
        if (
            not isinstance(license_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", license_id)
            or _is_placeholder(license_id)
        ):
            findings.append(f"demo asset requires an SPDX license identifier: {raw_path}")
        if not isinstance(source, str) or not source.strip() or _is_placeholder(source):
            findings.append(f"demo asset requires source provenance: {raw_path}")
        try:
            actual_size = path.stat().st_size
            total_bytes += actual_size
            if actual_size > MAX_DEMO_ASSET_BYTES:
                findings.append(f"demo asset exceeds 95 MiB: {raw_path}")
                continue
            if actual_size != size:
                findings.append(f"demo asset size mismatch: {raw_path}")
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if not isinstance(digest, str) or hasher.hexdigest() != digest.lower():
                findings.append(f"demo asset sha256 mismatch: {raw_path}")
        except OSError as error:
            findings.append(f"unreadable demo asset {raw_path}: {error}")
        if len(findings) == before:
            allowed.add(_relative(path, root))

    if total_bytes > MAX_DEMO_TOTAL_BYTES:
        findings.append("demo assets exceed 256 MiB total")
        allowed.clear()
    return allowed


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _check_git_state(root: Path, files: Sequence[Path], findings: List[str]) -> None:
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
        findings.append("--require-git requires the release root to be a Git working tree root")
        return
    head = _git(root, "rev-parse", "--verify", "HEAD")
    if head is None or head.returncode:
        findings.append("--require-git requires a committed HEAD")
    tracked = _git(root, "ls-files", "--cached", "-z", "--", ".")
    if tracked is None or tracked.returncode:
        findings.append("unable to list tracked release payload files")
        return
    tracked_paths = set(tracked.stdout.split("\0"))
    for path in files:
        relative_path = _relative(path, root)
        if relative_path not in tracked_paths:
            findings.append(f"untracked release payload: {relative_path}")
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".")
    if status is None or status.returncode:
        findings.append("unable to inspect Git index/worktree state")
    elif status.stdout:
        findings.append("--require-git requires a clean Git index and working tree")


def _check_paper_exact(root: Path, findings: List[str]) -> None:
    """Schema 3's explicitly incomplete paper artifacts must fail closed."""
    try:
        payload = json.loads((root / "docs/release/manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        findings.append("--paper-exact requires a readable docs/release/manifest.json")
        return
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        findings.append("--paper-exact requires docs/release/manifest.json schema_version 3")
        return
    release = payload.get("release", {})
    if not isinstance(release, dict) or release.get("status") not in {
        "paper_exact", "paper_exact_release", "published_paper_exact",
    }:
        findings.append("paper-exact: release.status must explicitly declare a paper_exact release")

    artifacts = payload.get("paper_artifacts", {})
    if not isinstance(artifacts, dict):
        findings.append("paper-exact: missing paper_artifacts object")
        return
    published_states = {
        "published", "released", "published_verified", "published_clean_clone_runnable",
    }
    for name in (
        "mod_checkpoint", "split_manifests", "sam3d_camera_ready_integration", "full_evaluator",
    ):
        artifact = artifacts.get(name, {})
        if not isinstance(artifact, dict):
            artifact = {}
        if artifact.get("status") not in published_states:
            findings.append(f"paper-exact: {name} is not published and verified")
        required_fields = list(artifact.get("required", [])) if isinstance(
            artifact.get("required", []), list
        ) else []
        if name in ("mod_checkpoint", "sam3d_camera_ready_integration"):
            required_fields.extend(("public_url", "revision", "license"))
        if name == "split_manifests":
            required_fields.extend(("portable_relative_paths", "stable_sample_ids", "published_urls"))
        if name == "full_evaluator":
            required_fields.extend(("public_url", "revision"))
            if artifact.get("missing_sources"):
                findings.append("paper-exact: full_evaluator still has missing_sources")
        for field in dict.fromkeys(str(field) for field in required_fields):
            value = artifact.get(field)
            if not value or (isinstance(value, str) and _is_placeholder(value)):
                findings.append(f"paper-exact: {name}.{field} is required")
            elif field == "public_url" and (
                not isinstance(value, str) or not re.match(r"https?://[^/\s]+", value)
            ):
                findings.append(f"paper-exact: {name}.public_url must be a public HTTP(S) URL")

    dataset = payload.get("dataset_snapshot", {})
    if not isinstance(dataset, dict) or dataset.get("real_data_processing_ready") is not True:
        findings.append("paper-exact: real dataset processing is not ready")
    split = artifacts.get("split_manifests", {})
    evaluation = split.get("messykitchens_evaluation", {}) if isinstance(split, dict) else {}
    if isinstance(evaluation, dict):
        for field in ("duplicate_review_required", "paper_weighting_author_confirmation_required"):
            if evaluation.get(field) is not False:
                findings.append(f"paper-exact: split_manifests.messykitchens_evaluation.{field} unresolved")
    physical = artifacts.get("physical_contact_metrics", {})
    if not isinstance(physical, dict) or physical.get("included_in_release") is not True:
        findings.append("paper-exact: paper physical_contact_metrics are not included")


def _scan_file(
    path: Path, root: Path, findings: List[str], allowed_assets: set[str] | None = None
) -> None:
    relative_path = _relative(path, root)
    if path.is_symlink():
        findings.append(f"symbolic link: {relative_path}")
        return
    allowed_binary = relative_path in (allowed_assets or set())
    if path.suffix.lower() in BINARY_SUFFIXES and not allowed_binary:
        findings.append(f"binary/checkpoint artifact: {relative_path}")
        return

    is_declared_text = path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore"
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        if allowed_binary and not is_declared_text:
            return
        findings.append(f"non-UTF-8/binary file: {relative_path}")
        return
    except OSError as error:
        findings.append(f"unreadable file: {relative_path}: {error}")
        return

    if not is_declared_text and b"\x00" in raw_bytes:
        if allowed_binary:
            return
        findings.append(f"binary file: {relative_path}")
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        if _POSIX_ABSOLUTE_RE.search(line) or _WINDOWS_ABSOLUTE_RE.search(line):
            findings.append(f"private/cluster absolute path: {relative_path}:{line_number}")
        assignment_match = _SECRET_ASSIGNMENT_RE.search(line)
        if assignment_match and not _is_placeholder(assignment_match.group("value")):
            findings.append(f"hard-coded secret value: {relative_path}:{line_number}")
        if any(pattern.search(line) for pattern in _KNOWN_SECRET_RES):
            findings.append(f"credential-like value: {relative_path}:{line_number}")


def _compile_python(files: Sequence[Path], root: Path, findings: List[str]) -> int:
    python_files = [path for path in files if path.suffix == ".py" and not path.is_symlink()]
    with tempfile.TemporaryDirectory(prefix="multi_object_decoder_compile_") as temporary_directory:
        compile_root = Path(temporary_directory)
        for index, path in enumerate(python_files):
            bytecode_path = compile_root / f"{index}.pyc"
            try:
                py_compile.compile(str(path), cfile=str(bytecode_path), doraise=True)
            except py_compile.PyCompileError as error:
                findings.append(f"Python compile failure: {_relative(path, root)}: {error.msg}")
            except OSError as error:
                findings.append(f"Python compile failure: {_relative(path, root)}: {error}")
    return len(python_files)


def check_release(
    root: Path, *, require_git: bool = False, paper_exact: bool = False
) -> Tuple[List[str], int, int]:
    findings: List[str] = []
    _check_required_release_paths(root, findings)
    _check_shell_entrypoints(root, findings)
    files = list(_iter_release_files(root, findings))
    allowed_assets = _check_demo_assets(root, findings)
    for path in files:
        _scan_file(path, root, findings, allowed_assets)
    compiled_count = _compile_python(files, root, findings)
    _check_manifest(
        root,
        findings,
        actual_file_count=len(files),
        actual_python_file_count=compiled_count,
    )
    if require_git:
        _check_git_state(root, files, findings)
    if paper_exact:
        _check_paper_exact(root, findings)
    return findings, len(files), compiled_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Release tree to inspect (default: repository root).",
    )
    parser.add_argument(
        "--require-git", action="store_true",
        help="Require every payload file to be tracked and the Git index/worktree clean.",
    )
    parser.add_argument(
        "--paper-exact", action="store_true",
        help="Also require published paper artifacts and complete paper reproduction metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: release root is not a directory: {root}", file=sys.stderr)
        return 2

    findings, file_count, compiled_count = check_release(
        root, require_git=args.require_git, paper_exact=args.paper_exact
    )
    if args.require_git:
        head = _git(root, "rev-parse", "--verify", "HEAD")
        if head is not None and head.returncode == 0:
            print(f"Git HEAD: {head.stdout.strip()}", flush=True)
    if findings:
        print(f"Release check failed with {len(findings)} finding(s):", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(
        f"Release check passed: scanned {file_count} files and compiled "
        f"{compiled_count} Python files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
