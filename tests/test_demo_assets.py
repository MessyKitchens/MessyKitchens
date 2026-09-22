"""Complete-scene object shard integrity and bundled original-data contracts."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image
import pytest
import torch

from multi_object_decoder.data import ArtifactSchemaError, SceneDataset, load_scene

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[dict, list[dict]]:
    factory = runpy.run_path(str(ROOT / "scripts/dev/make_demo_data.py"))["_scene_archive"]
    arrays = factory(np.random.default_rng(17), np.asarray([5, 1], dtype=np.int64), 0)
    np.savez(tmp_path / "full.npz", **arrays)
    shards = []
    for index in range(2):
        shard = {key: value[index : index + 1] for key, value in arrays.items()}
        np.savez(tmp_path / f"object_{index}.npz", **shard)
        shards.append(shard)
    return {"scene_id": "fixture", "scene_npz_paths": ["object_0.npz", "object_1.npz"]}, shards


def test_object_shards_reconstruct_one_scene_in_numeric_id_order(tmp_path: Path) -> None:
    record, _ = _fixture(tmp_path)
    scene = load_scene(record, tmp_path)
    reference = load_scene({"scene_id": "fixture", "scene_npz_path": "full.npz"}, tmp_path)
    assert scene.object_ids == (1, 5)
    assert scene.num_objects == 2
    for field in ("pose_tokens", "shape_tokens", "base_poses", "target_poses"):
        assert torch.equal(getattr(scene, field), getattr(reference, field))
    assert torch.equal(scene.decode_state.image_condition, reference.decode_state.image_condition)
    assert all(Path(path).is_absolute() for path in scene.record["scene_npz_paths"])


def test_duplicate_objects_across_shards_are_rejected(tmp_path: Path) -> None:
    record, shards = _fixture(tmp_path)
    shards[1]["object_ids"] = shards[0]["object_ids"].copy()
    np.savez(tmp_path / "object_1.npz", **shards[1])
    with pytest.raises(ArtifactSchemaError, match="duplicate object IDs"):
        load_scene(record, tmp_path)


@pytest.mark.parametrize("change", ["missing_field", "wrong_shape", "wrong_dtype"])
def test_shards_reject_inconsistent_field_schema(tmp_path: Path, change: str) -> None:
    record, shards = _fixture(tmp_path)
    key = "image_condition_per_object"
    if change == "missing_field":
        del shards[1][key]
    elif change == "wrong_shape":
        shards[1][key] = np.repeat(shards[1][key], 2, axis=1)
    else:
        shards[1][key] = shards[1][key].astype(np.float64)
    np.savez(tmp_path / "object_1.npz", **shards[1])
    with pytest.raises(ArtifactSchemaError, match="identical fields, dtypes"):
        load_scene(record, tmp_path)


def test_shards_cannot_be_mixed_with_another_cache_source(tmp_path: Path) -> None:
    record, _ = _fixture(tmp_path)
    record["scene_npz_path"] = "full.npz"
    with pytest.raises(ArtifactSchemaError, match="conflicts"):
        load_scene(record, tmp_path)


def test_bundled_gso_scene_has_all_four_original_objects_and_targets() -> None:
    root = ROOT / "assets/demo/real"
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    assert provenance["sample_id"] == "train_0000_az_055_el_073_fr_0000"
    assert provenance["real_camera_capture"] is False
    raw_record = json.loads((root / "raw/input.json").read_text())[0]
    assert "target_poses_path" not in raw_record
    assert "gt_poses_path" not in raw_record
    assert not (root / "raw/target_poses.json").exists()
    sample = SceneDataset(root / "prepared/prepared_data.json", require_target=True)[0]
    assert sample.object_ids == (0, 1, 2, 3)
    assert sample.pose_tokens.shape == (4, 4, 1024)
    assert sample.shape_tokens.shape == (4, 4096, 1024)
    assert sample.decode_state.image_condition.shape == (4, 7528, 1024)
    assert sample.target_poses is not None
    with Image.open(root / "raw/image.png") as image:
        assert image.size == (512, 512)
    with Image.open(root / "raw/instance_mask.png") as image:
        assert image.mode == "L"
        assert set(np.unique(np.asarray(image)).tolist()) == {0, 1, 2, 3, 4}
    for item in provenance["objects"]:
        assert item["mask_label"] == item["object_id"] + 1
        assert (root / item["canonical_file"]).is_file()
        shard = root / item["cache_shard"]
        assert shard.stat().st_size <= 50 * 1024**2
        with np.load(shard, allow_pickle=False) as archive:
            assert archive["pose_latents_raw__shape"].dtype == np.float32
            assert archive["image_condition_per_object"].dtype == np.float16
            assert all(not archive[key].dtype.hasobject for key in archive.files)
