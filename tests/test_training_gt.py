"""Registration-target contracts, independent of SAM3D weights and GPU access."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from multi_object_decoder import training_gt
from multi_object_decoder.mesh_io import transform_mesh


trimesh = pytest.importorskip("trimesh")


def _stdout(matrix: np.ndarray, *, converged: int = 1) -> str:
    with np.errstate(invalid="ignore"):
        scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    rows = [
        f"  Matrix[{row}][0-3] = " + " ".join(f"{value:.9g}" for value in matrix[row])
        for row in range(4)
    ]
    return "\n".join([
        "Computed alignment transformation ...", *rows,
        f"  Scale = {scale:.9g}", f"  Converged = {converged}", "  RMSD = 0.0001",
    ])


def _similarity(angle: float, scale: float, translation: list[float]) -> np.ndarray:
    transform = trimesh.transformations.rotation_matrix(angle, [0.2, 0.3, 0.8])
    transform[:3, :3] *= scale
    transform[:3, 3] = translation
    return transform


@pytest.fixture
def inputs(tmp_path):
    canonical = tmp_path / "local_meshes"
    canonical.mkdir()
    mesh = trimesh.creation.box(extents=[0.7, 1.1, 1.4])
    mesh.export(canonical / "obj_0_local.ply")
    base_pose = {
        "translation": [0.2, -0.3, 1.1],
        "rotation": [math.cos(0.35), 0, 0, math.sin(0.35)],
        "scale": [0.8, 1.2, 1.7],
    }
    base_path = tmp_path / "base_poses.json"
    base_path.write_text(json.dumps({"objects": {"obj_000": base_pose}}))
    scene = trimesh.Scene()
    node_transform = _similarity(0.6, 1.3, [3, 4, 5])
    scene.add_geometry(mesh, geom_name="different_geometry_name", node_name="annotated_object",
                       transform=node_transform)
    scene.add_geometry(trimesh.creation.box(), geom_name="unmatched_background",
                       node_name="background", transform=trimesh.transformations.translation_matrix(
                           [100, 100, 100]))
    scene_path = tmp_path / "gt.glb"
    scene.export(scene_path)
    # Only its existence and executability matter when subprocess is mocked.
    binary = tmp_path / "fake mshalign"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return {
        "canonical_dir": canonical,
        "base_poses_path": base_path,
        "gt_scene_path": scene_path,
        "object_node_map": {"0": "annotated_object"},
        "output_path": tmp_path / "targets" / "target_poses.json",
        "gaps_binary": binary,
    }


def _mock_registrations(monkeypatch, transforms):
    observed = []

    def run(command, **kwargs):
        assert isinstance(command, list)
        assert command[-1] == "-v"
        assert kwargs == {"capture_output": True, "text": True, "check": True, "timeout": 120.0}
        source = trimesh.load(command[1], process=False)
        target = trimesh.load(command[2], process=False)
        observed.append((source.copy(), target.copy()))
        transform = transforms[len(observed) - 1]
        source.apply_transform(transform)
        source.export(command[3])
        return SimpleNamespace(stdout=_stdout(transform), stderr="", returncode=0)

    monkeypatch.setattr(training_gt.subprocess, "run", run)
    return observed


def test_generate_targets_uses_graph_world_transform_and_prediction_frame(inputs, monkeypatch):
    scene_transform = _similarity(-0.4, 2.3, [0.1, -1.4, 0.9])
    object_transform = _similarity(0.8, 0.7, [0.3, -0.5, 0.2])
    observed = _mock_registrations(monkeypatch, [scene_transform, object_transform])
    payload = training_gt.generate_training_gt(**inputs)

    assert payload == json.loads(inputs["output_path"].read_text())
    assert payload["metadata"]["output_pose_space"] == "sam3d_pred"
    assert payload["metadata"]["num_matched"] == 1
    assert payload["metadata"]["source_path_base"] == "target_json_directory"
    assert (inputs["output_path"].parent / payload["metadata"]["gt_scene_path"]).resolve() == (
        inputs["gt_scene_path"].resolve()
    )
    assert len(payload["metadata"]["gaps_binary_sha256"]) == 64
    assert payload["metadata"]["object_alignments"]["obj_000"]["gt_scene_node"] == "annotated_object"
    assert len(observed) == 2
    # The unrelated background mesh is not used in scene registration.
    loaded_gt = trimesh.load(inputs["gt_scene_path"], force="scene", process=False)
    node_transform, geometry_name = loaded_gt.graph["annotated_object"]
    matched_gt = loaded_gt.geometry[geometry_name].copy()
    matched_gt.apply_transform(node_transform)
    np.testing.assert_allclose(observed[0][1].vertices, matched_gt.vertices, rtol=1e-6, atol=1e-6)
    matched_gt.apply_transform(np.linalg.inv(scene_transform))
    np.testing.assert_allclose(observed[1][1].vertices, matched_gt.vertices, rtol=1e-6, atol=1e-6)
    # Non-identity rotation and anisotropic base scale expose row/column and
    # multiply-order bugs: recovered target mesh must equal object @ base.
    local = trimesh.load(inputs["canonical_dir"] / "obj_0_local.ply", process=False)
    target_pose = payload["objects"]["obj_000"]
    recovered = transform_mesh(local, **target_pose)
    expected = observed[1][0].copy()
    expected.apply_transform(object_transform)
    np.testing.assert_allclose(recovered.vertices, expected.vertices, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(target_pose["scale"], np.array([0.8, 1.2, 1.7]) * 0.7)


def test_successful_identity_registration_is_valid(inputs, monkeypatch):
    _mock_registrations(monkeypatch, [np.eye(4), np.eye(4)])
    payload = training_gt.generate_training_gt(**inputs)
    original = json.loads(inputs["base_poses_path"].read_text())["objects"]["obj_000"]
    for field in ("translation", "rotation", "scale"):
        np.testing.assert_allclose(payload["objects"]["obj_000"][field], original[field])


def test_noncontiguous_ids_and_hierarchical_scene_correspondence(inputs, monkeypatch):
    base = json.loads(inputs["base_poses_path"].read_text())
    base["objects"]["obj_012"] = {
        "translation": [2, 0, 0], "rotation": [1, 0, 0, 0], "scale": [1, 1, 1]
    }
    inputs["base_poses_path"].write_text(json.dumps(base))
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    mesh.export(inputs["canonical_dir"] / "obj_12_local.ply")
    scene = trimesh.load(inputs["gt_scene_path"], force="scene", process=False)
    parent = trimesh.transformations.translation_matrix([8, 9, 10])
    child = trimesh.transformations.rotation_matrix(0.3, [0, 0, 1])
    scene.graph.update(frame_to="parent", matrix=parent)
    scene.add_geometry(mesh, geom_name="second_geometry", node_name="child",
                       parent_node_name="parent", transform=child)
    scene.export(inputs["gt_scene_path"])
    # Map insertion and node-name sort order deliberately differ from ID order.
    inputs["object_node_map"] = {"12": "child", "obj_0": "annotated_object"}
    observed = _mock_registrations(monkeypatch, [np.eye(4), np.eye(4), np.eye(4)])
    result = training_gt.generate_training_gt(**inputs)
    assert set(result["objects"]) == {"obj_000", "obj_012"}
    assert result["metadata"]["num_matched"] == 2
    expected = mesh.copy()
    expected.apply_transform(parent @ child)
    np.testing.assert_allclose(observed[2][1].vertices, expected.vertices, rtol=1e-6, atol=1e-6)
    assert result["metadata"]["object_alignments"]["obj_012"]["gt_scene_node"] == "child"


@pytest.mark.parametrize("mapping, message", [
    ({}, "non-empty object_node_map"),
    ({"1": "annotated_object"}, "cover exactly all base objects"),
    ({"obj_000": "annotated_object", "0": "another"}, "duplicate normalized"),
    ({"0": "annotated_object", "1": "annotated_object"}, "one-to-one"),
    ({"0": "different_geometry_name"}, "no geometry-bearing node"),
    ({"0": ""}, "non-empty string"),
    ({"bad_id": "annotated_object"}, "Invalid object ID"),
])
def test_explicit_correspondence_errors_never_produce_targets(inputs, monkeypatch, mapping, message):
    def should_not_run(*args, **kwargs):
        pytest.fail("Invalid correspondence must be rejected before GAPs runs")

    monkeypatch.setattr(training_gt.subprocess, "run", should_not_run)
    inputs["object_node_map"] = mapping
    with pytest.raises(ValueError, match=message):
        training_gt.generate_training_gt(**inputs)
    assert not inputs["output_path"].exists()


@pytest.mark.parametrize("field, value, message", [
    ("rotation", [0, 0, 0, 0], "quaternion must be nonzero"),
    ("rotation", [1, 0, 0], "four|4 finite"),
    ("translation", [float("nan"), 0, 0], "finite"),
    ("scale", [1, 0, 1], "scale must be positive"),
    ("scale", [-1, 1, 1], "scale must be positive"),
])
def test_invalid_base_pose_is_rejected(inputs, field, value, message):
    payload = json.loads(inputs["base_poses_path"].read_text())
    payload["objects"]["obj_000"][field] = value
    inputs["base_poses_path"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        training_gt.generate_training_gt(**inputs)
    assert not inputs["output_path"].exists()


def test_duplicate_base_ids_and_canonical_meshes_rejected(inputs):
    payload = json.loads(inputs["base_poses_path"].read_text())
    payload["objects"]["0"] = payload["objects"]["obj_000"]
    inputs["base_poses_path"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate normalized object ID"):
        training_gt.generate_training_gt(**inputs)
    del payload["objects"]["0"]
    inputs["base_poses_path"].write_text(json.dumps(payload))
    trimesh.creation.box().export(inputs["canonical_dir"] / "obj_000_local.ply")
    with pytest.raises(ValueError, match="Duplicate canonical mesh"):
        training_gt.generate_training_gt(**inputs)


@pytest.mark.parametrize("failure", ["missing", "failed", "timeout", "invalid", "no_output", "mismatch"])
def test_registration_failures_preserve_existing_targets(inputs, monkeypatch, failure):
    output = inputs["output_path"]
    output.parent.mkdir()
    output.write_text("previous verified targets")
    if failure == "missing":
        inputs["gaps_binary"] = inputs["gaps_binary"].with_name("does_not_exist")

    def run(command, **kwargs):
        if failure == "failed":
            raise subprocess.CalledProcessError(7, command, stderr="registration failed")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "invalid":
            return SimpleNamespace(stdout="Nothing usable", stderr="", returncode=0)
        if failure == "mismatch":
            mesh = trimesh.load(command[1], process=False)
            mesh.apply_translation([100, 100, 100])
            mesh.export(command[3])
        return SimpleNamespace(stdout=_stdout(np.eye(4)), stderr="", returncode=0)

    monkeypatch.setattr(training_gt.subprocess, "run", run)
    with pytest.raises((ValueError, RuntimeError)):
        training_gt.generate_training_gt(**inputs)
    assert output.read_text() == "previous verified targets"
    assert list(output.parent.iterdir()) == [output]


def test_late_object_failure_does_not_write_partial_gt(inputs, monkeypatch):
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, command, stderr="object failed")
        trimesh.load(command[1], process=False).export(command[3])
        return SimpleNamespace(stdout=_stdout(np.eye(4)), stderr="", returncode=0)

    monkeypatch.setattr(training_gt.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="object_0 registration failed"):
        training_gt.generate_training_gt(**inputs)
    assert calls == 2
    assert not inputs["output_path"].exists()


def test_parser_uses_final_block_not_trial_transform():
    trial = _stdout(_similarity(0.1, 0.5, [4, 5, 6])).replace(
        "Computed alignment transformation ...", "Computed alignment transformation for flip ..."
    )
    final = _similarity(0.4, 2, [1, 2, 3])
    matrix, diagnostics = training_gt._parse_gaps_output(trial + "\n" + _stdout(final))
    np.testing.assert_allclose(matrix, final, atol=1e-8)
    assert diagnostics["converged"]


@pytest.mark.parametrize("transform, message", [
    (np.diag([0, 1, 1, 1]), "singular"),
    (np.diag([-1, 1, 1, 1]), "reflections"),
    (np.diag([1, 2, 1, 1]), "uniform positive scale"),
    (np.diag([1, 1, 1, 2]), "homogeneous"),
    (np.array([[1, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), "uniform"),
    (np.diag([float("nan"), 1, 1, 1]), "finite 4x4"),
])
def test_parser_rejects_invalid_transforms(transform, message):
    with pytest.raises(ValueError, match=message):
        training_gt._parse_gaps_output(_stdout(transform))


@pytest.mark.parametrize("stdout, message", [
    ("", "missing the final"),
    (_stdout(np.eye(4)).replace("Matrix[3][0-3] = 0 0 0 1", ""), "complete 4x4"),
    (_stdout(np.eye(4)).replace("Scale = 1", "Scale = -1"), "invalid scale"),
    (_stdout(np.eye(4)).replace("Scale = 1", "Scale = 2"), "disagrees"),
    (_stdout(np.eye(4), converged=0), "did not converge"),
    (_stdout(np.eye(4)).replace("RMSD = 0.0001", "RMSD = nan"), "invalid RMSD"),
    (_stdout(np.eye(4)).replace("RMSD = 0.0001", ""), "missing Scale"),
])
def test_parser_requires_success_diagnostics(stdout, message):
    with pytest.raises(ValueError, match=message):
        training_gt._parse_gaps_output(stdout)


def test_empty_geometry_rejected():
    with pytest.raises(ValueError, match="invalid or empty"):
        training_gt._checked_mesh(trimesh.Trimesh(), "empty")
    degenerate = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                 faces=[[0, 1, 2]], process=False)
    with pytest.raises(ValueError, match="positive finite surface area"):
        training_gt._checked_mesh(degenerate, "line")


def test_output_cannot_overwrite_inputs(inputs):
    inputs["output_path"] = inputs["base_poses_path"]
    original = inputs["base_poses_path"].read_text()
    with pytest.raises(ValueError, match="must not overwrite"):
        training_gt.generate_training_gt(**inputs)
    assert inputs["base_poses_path"].read_text() == original


@pytest.mark.skipif(
    not os.environ.get("GAPS_MSHALIGN"),
    reason="Set GAPS_MSHALIGN to an executable mshalign for native registration coverage",
)
def test_native_gaps_generates_valid_prediction_frame_targets(tmp_path):
    """Exercise the actual CLI/stdout/mesh contract without asserting ICP accuracy."""
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    # Fully deterministic, asymmetric geometry avoids relying on external data
    # while still exercising PCA/ICP on a genuine three-dimensional surface.
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.vertices *= [0.4, 0.7, 1.2]
    mesh.vertices[mesh.vertices[:, 2] > 0, 0] += 0.2
    mesh.export(canonical_dir / "obj_0_local.ply")
    pose = {
        "translation": [0.2, -0.3, 1.1],
        "rotation": [1, 0, 0, 0],
        "scale": [1, 1, 1],
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"objects": {"obj_000": pose}}))
    posed_mesh = transform_mesh(mesh, **pose)
    world_transform = _similarity(0.3, 1.4, [2, 3, 4])
    scene = trimesh.Scene()
    scene.add_geometry(posed_mesh, geom_name="shape", node_name="gt_object",
                       transform=world_transform)
    scene_path = tmp_path / "scene.glb"
    scene.export(scene_path)
    output = tmp_path / "target.json"

    result = training_gt.generate_training_gt(
        canonical_dir=canonical_dir,
        base_poses_path=base_path,
        gt_scene_path=scene_path,
        object_node_map={"0": "gt_object"},
        output_path=output,
        gaps_binary=Path(os.environ["GAPS_MSHALIGN"]),
    )

    assert json.loads(output.read_text()) == result
    assert result["metadata"]["output_pose_space"] == "sam3d_pred"
    assert result["metadata"]["num_matched"] == 1
    for diagnostics in (
        result["metadata"]["scene_alignment"],
        result["metadata"]["object_alignments"]["obj_000"],
    ):
        assert diagnostics["converged"]
        assert math.isfinite(diagnostics["rmsd"]) and diagnostics["rmsd"] >= 0
        assert math.isfinite(diagnostics["scale"]) and diagnostics["scale"] > 0
    target = result["objects"]["obj_000"]
    assert np.isfinite(target["translation"] + target["rotation"] + target["scale"]).all()
    assert np.all(np.asarray(target["scale"]) > 0)
    assert np.linalg.norm(target["rotation"]) == pytest.approx(1.0)
    forward = np.asarray(result["metadata"]["scene_transform_sam3d_to_gt"])
    inverse = np.asarray(result["metadata"]["scene_transform_gt_to_sam3d"])
    np.testing.assert_allclose(forward @ inverse, np.eye(4), atol=1e-10)
