"""CLI contract test for direct SAM3D pipeline preparation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from multi_object_decoder.commands.prepare import (
    _prepared_record,
    _planned_output_directories,
    _record_identity,
    _remove_scene_output,
    _validated_output_root,
)
from multi_object_decoder.commands import prepare as prepare_cli
from multi_object_decoder.data import ArtifactSchemaError, SceneDataset


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_record_identity_distinguishes_views_of_the_same_scene() -> None:
    assert _record_identity(
        {"scene_id": "scene_10", "image_id": "IMG_8873", "id": 0}
    ) == "scene_10/IMG_8873"
    assert _record_identity(
        {"scene_id": "train_0410", "id": "medium/train_0410/view_001"}
    ) == "medium/train_0410/view_001"


def test_prepare_rejects_unsafe_overwrite_roots(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()

    assert _validated_output_root(None, manifest_dir, overwrite=False) is None
    with pytest.raises(ValueError, match="requires an explicit --output-dir"):
        _validated_output_root(None, manifest_dir, overwrite=True)
    with pytest.raises(ValueError, match="filesystem root"):
        _validated_output_root(Path("/"), manifest_dir, overwrite=True)
    with pytest.raises(ValueError, match="user home"):
        _validated_output_root(Path.home(), manifest_dir, overwrite=True)
    with pytest.raises(ValueError, match="manifest directory"):
        _validated_output_root(manifest_dir, manifest_dir, overwrite=True)


def test_prepare_refuses_symlink_or_non_child_deletion(tmp_path: Path) -> None:
    output_root = tmp_path / "prepared"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    symlink = output_root / "scene"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _remove_scene_output(symlink, output_root)
    assert marker.is_file()

    with pytest.raises(ValueError, match="strict child"):
        _remove_scene_output(outside, output_root)
    assert marker.is_file()

    nested_manifest = output_root / "manifest_scene"
    nested_manifest.mkdir()
    manifest_marker = nested_manifest / "input.json"
    manifest_marker.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="protected path"):
        _remove_scene_output(
            nested_manifest,
            output_root,
            manifest_dir=nested_manifest,
        )
    assert manifest_marker.is_file()

    real_scene = output_root / "real_scene"
    real_scene.mkdir()
    (real_scene / "cache.bin").write_bytes(b"cache")
    _remove_scene_output(real_scene, output_root)
    assert not real_scene.exists()


def test_prepare_rejects_sanitized_scene_directory_collisions(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "prepared"
    with pytest.raises(ValueError, match="same output directory"):
        _planned_output_directories(
            [{"scene_id": "a/b"}, {"scene_id": "a__b"}],
            tmp_path,
            output_root,
        )


def test_prepared_record_discards_old_shards_and_pose_aliases(tmp_path: Path) -> None:
    prepared = _prepared_record(
        {"scene_id": "scene", "scene_npz_paths": ["old.npz"],
         "sam3d_pose_path": "old_poses.json"},
        tmp_path, tmp_path, tmp_path / "scene", tmp_path / "scene/pose_inference.json",
        "scene", False,
    )
    assert "scene_npz_paths" not in prepared
    assert "sam3d_pose_path" not in prepared
    assert prepared["base_poses_path"] == "scene/pose_inference.json"


def test_prepare_preserves_in_place_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "prepared_data.json"
    contents = json.dumps([{"scene_id": "scene", "tokens_path": "scene/tokens"}])
    manifest.write_text(contents)
    monkeypatch.setattr(sys, "argv", ["prepare", "--data-json", str(manifest)])
    with pytest.raises(ValueError, match="input manifest must not be"):
        prepare_cli.main()
    assert manifest.read_text() == contents


def test_prepare_overwrite_protects_other_scene_gt_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs"
    first_output = output_dir / "first"
    first_output.mkdir(parents=True)
    gt_input = first_output / "source.glb"
    gt_input.write_bytes(b"original GT must not be deleted")
    manifest = input_dir / "input.json"
    manifest.write_text(json.dumps([
        {"scene_id": "first"},
        {"scene_id": "second", "mesh_path": str(gt_input)},
    ]))
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--data-json", str(manifest), "--output-dir", str(output_dir), "--overwrite",
    ])
    with pytest.raises(ValueError, match="contains source mesh_path"):
        prepare_cli.main()
    assert gt_input.read_bytes() == b"original GT must not be deleted"


@pytest.mark.parametrize("generate_training_gt", [False, True])
def test_prepare_writes_loadable_manifest_without_notebook_inference(
    tmp_path: Path, generate_training_gt: bool,
) -> None:
    external = tmp_path / "fake_sam3d"
    package = external / "sam3d_objects"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "fake_pipeline.py").write_text(
        r'''
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _boxed(mapping):
    value = np.empty((), dtype=object)
    value[()] = mapping
    return value


class _Mesh:
    def __init__(self):
        self.vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            dtype=torch.float32,
        )
        self.faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int64)
        self.vertex_attrs = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )


class FakeSolver:
    def step(self, dynamics, x_t, t, dt, *args, **kwargs):
        return dynamics(x_t, t, *args, **kwargs)


class FakePipeline:
    def __init__(self, **kwargs):
        backbone = SimpleNamespace(
            latent_mapping={"translation": object()},
            split_latent_share_transformer=lambda raw: raw,
        )
        generator = SimpleNamespace(
            reverse_fn=SimpleNamespace(backbone=backbone),
            _solver=FakeSolver(),
            _solver_method="euler",
        )
        self.models = {"ss_generator": generator}
        self.pose_decoder = lambda *args, **kwargs: {}

    def get_condition_input(self, condition):
        return (condition,), {}

    def sample_sparse_structure(self, inputs):
        self.get_condition_input(inputs["condition"])
        generator = self.models["ss_generator"]
        def dynamics(x_t, t):
            return generator.reverse_fn.backbone.split_latent_share_transformer(inputs["raw"])
        return generator._solver.step(dynamics, inputs["x_t"], 0.5, 0.1)

    def run(self, image, mask, seed, **kwargs):
        assert image.dtype == np.uint8
        assert image.shape == (6, 8, 4)
        assert mask is None
        assert set(np.unique(image[..., 3]).tolist()) == {0, 255}
        assert kwargs == {
            "stage1_only": False,
            "with_mesh_postprocess": False,
            "with_texture_baking": False,
            "with_layout_postprocess": True,
            "use_vertex_color": True,
            "stage1_inference_steps": None,
            "pointmap": None,
        }

        object_id = int(os.environ["SAM3D_OBJ_ID"])
        token_dir = Path(os.environ["SAM3D_TOKEN_DIR"])
        token_dir.mkdir(parents=True, exist_ok=True)
        feature_dim, shape_count = 8, 4
        raw = {
            key: np.full((1, 1, feature_dim), object_id + 0.1, np.float32)
            for key in (
                "6drotation_normalized",
                "translation",
                "scale",
                "translation_scale",
            )
        }
        raw["shape"] = np.full(
            (1, shape_count, feature_dim), object_id + 0.2, np.float32
        )
        x_t = {
            "6drotation_normalized": np.zeros((1, 1, 6), np.float32),
            "translation": np.zeros((1, 1, 3), np.float32),
            "scale": np.zeros((1, 1, 3), np.float32),
            "translation_scale": np.zeros((1, 1, 1), np.float32),
            "shape": np.zeros((1, shape_count, 8), np.float32),
        }
        self.sample_sparse_structure({
            "raw": {key: torch.from_numpy(value) for key, value in raw.items()},
            "x_t": {key: torch.from_numpy(value) for key, value in x_t.items()},
            "pointmap_scale": torch.ones((1, 3)),
            "pointmap_shift": torch.zeros((1, 3)),
            "condition": torch.zeros((1, 2, feature_dim)),
        })
        (token_dir.parent / f"contract_{object_id:03d}.json").write_text(
            json.dumps({"seed": seed, "alpha": [0, 255]})
        )
        return {
            "translation": torch.tensor([[float(object_id), 0.0, 0.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "scale": torch.ones((1, 3)),
            "mesh": [_Mesh()],
        }
'''.lstrip(),
        encoding="utf-8",
    )
    # Any attempt to restore the old notebook path makes the subprocess fail.
    notebook = external / "notebook"
    notebook.mkdir()
    (notebook / "inference.py").write_text(
        "raise RuntimeError('notebook inference must not be imported')\n",
        encoding="utf-8",
    )
    pipeline_config = external / "pipeline.yaml"
    pipeline_config.write_text(
        "_target_: sam3d_objects.fake_pipeline.FakePipeline\n",
        encoding="utf-8",
    )

    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((6, 8, 3), 127, dtype=np.uint8)).save(image_path)
    mask_path = tmp_path / "mask.npy"
    instance_mask = np.zeros((6, 8), dtype=np.uint16)
    instance_mask[1:3, 1:3] = 1
    instance_mask[3:5, 5:7] = 4
    np.save(mask_path, instance_mask)
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "objects": {
                    f"obj_{object_id}": {
                        "translation": [float(object_id) + 0.1, 0.0, 0.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0],
                    }
                    for object_id in (0, 3)
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "input.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "scene_id": "fake/scene",
                    "num_parts": 2,
                    "image_path": image_path.name,
                    "instance_mask_path": mask_path.name,
                    "gt_poses_path": gt_path.name,
                }
            ]
        ),
        encoding="utf-8",
    )
    if generate_training_gt:
        import trimesh

        gt_scene = trimesh.Scene()
        for object_id in (0, 3):
            mesh = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
            mesh.apply_translation([float(object_id), 0.0, 0.0])
            gt_scene.add_geometry(mesh, node_name=f"ground_truth_{object_id}")
        gt_mesh_path = tmp_path / "gt_scene.glb"
        gt_scene.export(gt_mesh_path)
        raw_records = json.loads(manifest_path.read_text())
        raw_records[0].update({
            "mesh_path": gt_mesh_path.name,
            "gt_object_map": {"0": "ground_truth_0", "3": "ground_truth_3"},
        })
        manifest_path.write_text(json.dumps(raw_records), encoding="utf-8")
        gaps_binary = tmp_path / "fake_mshalign"
        gaps_binary.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "import trimesh\n"
            "mesh = trimesh.load(sys.argv[1], force='mesh', process=False)\n"
            "mesh.apply_translation([0.25, 0.0, 0.0])\n"
            "mesh.export(sys.argv[3])\n"
            "print('Computed alignment transformation ...')\n"
            "print('  Matrix[0][0-3] = 1 0 0 0.25')\n"
            "print('  Matrix[1][0-3] = 0 1 0 0')\n"
            "print('  Matrix[2][0-3] = 0 0 1 0')\n"
            "print('  Matrix[3][0-3] = 0 0 0 1')\n"
            "print('  Scale = 1')\n"
            "print('  Converged = 1')\n"
            "print('  RMSD = 0.01')\n",
            encoding="utf-8",
        )
        gaps_binary.chmod(0o755)
    output_dir = tmp_path / "prepared"
    environment = os.environ.copy()
    environment.pop("CONDA_PREFIX", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["SAM3D_PYTHON"] = sys.executable
    environment["SAM3D_ROOT"] = str(external)
    environment["SAM3D_PIPELINE_CONFIG"] = str(pipeline_config)
    prepare_command = [
        "bash",
        str(RELEASE_ROOT / "scripts/prepare.sh"),
        "--data-json",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
    ]
    if generate_training_gt:
        prepare_command += ["--generate-training-gt", "--gaps-binary", str(gaps_binary)]
    result = subprocess.run(
        prepare_command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    prepared_manifest = output_dir / "prepared_data.json"
    prepared_row = json.loads(prepared_manifest.read_text())[0]
    assert prepared_row["sam3d_results_path"] == "fake__scene"
    assert prepared_row["sam3d_pose_json"].endswith("/pose_inference.json")
    assert prepared_row["sam3d_mesh_path"].endswith("/complete_multi_object_mesh.glb")
    assert not Path(prepared_row["tokens_path"]).is_absolute()
    assert not Path(prepared_row["image_path"]).is_absolute()
    with pytest.raises(ArtifactSchemaError, match="--allow-legacy-pickle"):
        SceneDataset(prepared_manifest, require_target=True)[0]
    dataset = SceneDataset(
        prepared_manifest,
        require_target=True,
        allow_legacy_pickle=True,
    )
    sample = dataset[0]
    assert sample.scene_id == "fake/scene"
    assert sample.object_ids == (0, 3)
    assert sample.pose_tokens.shape == (2, 4, 8)
    assert sample.shape_tokens.shape == (2, 4, 8)
    assert sample.decode_state.pointmap_scale.shape == (2, 3)
    assert sample.decode_state.image_condition_per_object.shape == (2, 2, 8)
    assert sample.base_poses.shape == (2, 10)
    assert sample.target_poses is not None
    assert sample.target_poses.shape == (2, 10)
    if generate_training_gt:
        generated_target = output_dir / prepared_row["target_poses_path"]
        assert prepared_row["target_poses_path"] == prepared_row["gt_poses_path"]
        assert generated_target.name == "target_poses.json"
        payload = json.loads(generated_target.read_text())
        assert payload["metadata"]["output_pose_space"] == "sam3d_pred"
        np.testing.assert_allclose(sample.target_poses[:, 0].numpy(), [0.25, 3.25])
        # Fresh-cache targets must replace, not mutate, the historical source target.
        assert json.loads(gt_path.read_text())["objects"]["obj_0"]["translation"][0] == 0.1

    scene_dir = output_dir / "fake__scene"
    assert sorted(path.name for path in (scene_dir / "local_meshes").glob("*.ply")) == [
        "obj_0_local.ply",
        "obj_3_local.ply",
    ]
    assert (scene_dir / "complete_multi_object_mesh.glb").is_file()
    assert (scene_dir / "object_label_map.json").is_file()
    assert (scene_dir / "contract_000.json").is_file()
    assert (scene_dir / "contract_003.json").is_file()

    report = json.loads((output_dir / "prepare_manifest.json").read_text())
    assert report["prepared_manifest"] == "prepared_data.json"
    assert report["scenes"][0]["output_dir"] == "fake__scene"
    assert report["successful"] == 1
    assert report["failed"] == 0
    # Trusted-cache reuse/GT registration must not load the SAM3D model again.
    pipeline_config.write_text("_target_: nonexistent.Pipeline\n", encoding="utf-8")

    untrusted_reuse = subprocess.run(
        prepare_command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert untrusted_reuse.returncode == 1
    assert "--allow-legacy-pickle" in untrusted_reuse.stdout

    trusted_reuse = subprocess.run(
        [*prepare_command, "--allow-legacy-pickle"],
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert trusted_reuse.returncode == 0, (
        f"stdout:\n{trusted_reuse.stdout}\nstderr:\n{trusted_reuse.stderr}"
    )
    trusted_report = json.loads((output_dir / "prepare_manifest.json").read_text())
    assert trusted_report["skipped"] == (0 if generate_training_gt else 1)
    if generate_training_gt:
        assert trusted_report["scenes"][0]["training_gt_generated"] is True
        assert trusted_report["scenes"][0]["cache_status"] == "reused"
        saved_target = generated_target.read_bytes()
        gaps_binary.write_text(f"#!{sys.executable}\nraise SystemExit(7)\n", encoding="utf-8")
        failed_registration = subprocess.run(
            [*prepare_command, "--allow-legacy-pickle"], cwd=RELEASE_ROOT, env=environment,
            capture_output=True, text=True, check=False, timeout=60,
        )
        assert failed_registration.returncode == 1
        failed_report = json.loads((output_dir / "prepare_manifest.json").read_text())
        assert failed_report["failed"] == 1
        assert json.loads(prepared_manifest.read_text()) == []
        assert generated_target.read_bytes() == saved_target
