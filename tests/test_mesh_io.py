"""Contract tests for the small release-owned mesh conversion layer."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from multi_object_decoder.mesh_io import (
    export_mesh_scene,
    mesh_result_to_trimesh,
    transform_mesh,
    transform_points_by_pose,
)


def test_pose_transform_uses_wxyz_scale_rotation_translation_and_gradients() -> None:
    points = torch.tensor([[[1.0, 0.0, 0.0]]], requires_grad=True)
    half_angle = math.pi / 4.0
    pose = torch.tensor(
        [
            [
                10.0,
                20.0,
                30.0,
                math.cos(half_angle),
                0.0,
                0.0,
                math.sin(half_angle),
                2.0,
                3.0,
                4.0,
            ]
        ],
        requires_grad=True,
    )

    transformed = transform_points_by_pose(points, pose)

    # This matches SAM3D's PyTorch3D row-vector convention: +90 degrees in the
    # conventional matrix appears as clockwise multiplication for row points.
    torch.testing.assert_close(transformed, torch.tensor([[[10.0, 18.0, 30.0]]]))
    transformed.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()


def test_sam3d_mesh_conversion_transform_and_glb_round_trip(tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")
    result = SimpleNamespace(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        vertex_attrs=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    )
    local = mesh_result_to_trimesh(result)
    posed = transform_mesh(
        local,
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        translation=torch.tensor([[2.0, 3.0, 4.0]]),
        scale=torch.tensor([[2.0, 1.0, 1.0]]),
    )

    assert np.asarray(local.visual.vertex_colors).shape == (3, 4)
    assert np.asarray(posed.vertices) == pytest.approx(
        np.asarray([[2.0, 3.0, 4.0], [4.0, 3.0, 4.0], [2.0, 4.0, 4.0]])
    )
    destination = export_mesh_scene(
        [local, posed], ["object_local", "object_posed"], tmp_path / "scene.glb"
    )
    loaded = trimesh.load(destination, force="scene", process=False)
    assert isinstance(loaded, trimesh.Scene)
    assert set(loaded.geometry) == {"object_local", "object_posed"}
    assert all(len(mesh.faces) == 1 for mesh in loaded.geometry.values())


def test_mesh_helpers_reject_invalid_geometry_and_ambiguous_exports(tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")
    invalid = SimpleNamespace(
        vertices=torch.zeros(3, 3),
        faces=torch.tensor([[0, 1, 7]]),
        vertex_attrs=None,
    )
    with pytest.raises(ValueError, match="outside the vertex array"):
        mesh_result_to_trimesh(invalid)

    mesh = trimesh.Trimesh(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    with pytest.raises(ValueError, match="same non-zero length"):
        export_mesh_scene([mesh], [], tmp_path / "mismatch.glb")
    with pytest.raises(ValueError, match="non-empty and unique"):
        export_mesh_scene([mesh, mesh], ["same", "same"], tmp_path / "duplicate.glb")
