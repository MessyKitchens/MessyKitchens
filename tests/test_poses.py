"""Pose JSON schema checks shared by inference output and target files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from multi_object_decoder.poses import load_pose_json, poses_to_json, save_pose_json


def _poses() -> torch.Tensor:
    poses = torch.zeros(2, 10)
    poses[:, 3] = 1.0
    poses[:, 7:10] = 1.0
    poses[1, :3] = torch.tensor([0.5, -0.25, 2.0])
    return poses


def test_pose_json_round_trip_keeps_numeric_object_order(tmp_path: Path) -> None:
    path = tmp_path / "mod_pose.json"
    save_pose_json(path, ("obj_10", "obj_2"), _poses(), metadata={"backend": "direct"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload["objects"]) == ["obj_10", "obj_2"]
    assert payload["metadata"] == {"backend": "direct"}
    names, loaded = load_pose_json(path)
    assert names == ["obj_2", "obj_10"]
    torch.testing.assert_close(loaded, _poses()[[1, 0]])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_pose_json_rejects_non_finite_poses(tmp_path: Path, value: float) -> None:
    poses = _poses()
    poses[0, 0] = value
    with pytest.raises(ValueError, match="NaN or infinity"):
        poses_to_json(("obj_0", "obj_1"), poses)
    path = tmp_path / "mod_pose.json"
    with pytest.raises(ValueError, match="NaN or infinity"):
        save_pose_json(path, ("obj_0", "obj_1"), poses)
    assert not path.exists()
