"""Self-contained SAM3D cache generation for MOD data processing.

This module deliberately avoids SAM3D's notebook ``Inference`` wrapper.  That
wrapper imports UI and visualization packages at module import time and assumes
``CONDA_PREFIX`` is set.  The small adapter below preserves its actual pipeline
call contract without those notebook-only side effects.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np


class SAM3DPrepareInference:
    """Minimal equivalent of ``notebook.inference.Inference.__call__``."""

    def __init__(self, pipeline: Any):
        self._pipeline = pipeline

    def __call__(
        self,
        image: Any,
        mask: Any,
        seed: int | None = None,
        pointmap: Any = None,
    ) -> Mapping[str, Any]:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask, dtype=bool)
        if image_array.ndim != 3 or image_array.shape[-1] < 3:
            raise ValueError(
                f"SAM3D image must have at least three channels, got {image_array.shape}"
            )
        if mask_array.shape != image_array.shape[:2]:
            raise ValueError(
                "SAM3D object mask and image dimensions differ: "
                f"{mask_array.shape} versus {image_array.shape[:2]}"
            )
        rgb = np.asarray(image_array[..., :3], dtype=np.uint8)
        alpha = mask_array.astype(np.uint8)[..., None] * np.uint8(255)
        rgba = np.concatenate((rgb, alpha), axis=-1)
        from multi_object_decoder.sam3d_capture import SAM3DCacheCapture

        capture = SAM3DCacheCapture(self._pipeline)
        with capture.observe():
            result = self._pipeline.run(
                rgba,
                None,
                seed,
                stage1_only=False,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=True,
                use_vertex_color=True,
                stage1_inference_steps=None,
                pointmap=pointmap,
            )
        token_dir = os.environ.get("SAM3D_TOKEN_DIR")
        object_id = os.environ.get("SAM3D_OBJ_ID")
        if token_dir is None or object_id is None:
            raise RuntimeError("SAM3D preparation requires an explicit per-object cache destination")
        capture.save(Path(token_dir), object_id)
        if not isinstance(result, Mapping):
            raise TypeError(
                f"SAM3D pipeline returned {type(result).__name__}, expected a mapping"
            )
        return result


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Unsupported RGB image shape at {path}: {array.shape}")
    return np.asarray(array[..., :3], dtype=np.uint8)


def _load_instance_mask(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            mask = np.asarray(image)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Instance mask must be two-dimensional at {path}: {mask.shape}")
    if not (
        np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.bool_)
    ):
        if not bool(np.isfinite(mask).all()) or not bool(np.equal(mask, np.floor(mask)).all()):
            raise ValueError(f"Instance mask must contain finite integer labels: {path}")
    mask = mask.astype(np.int64, copy=False)
    if mask.shape != image_shape:
        # Pillow's I mode preserves integer instance IDs while nearest-neighbor
        # resizing avoids inventing labels at object boundaries.
        resized = Image.fromarray(mask.astype(np.int32), mode="I").resize(
            (image_shape[1], image_shape[0]),
            resample=Image.Resampling.NEAREST,
        )
        mask = np.asarray(resized, dtype=np.int64)
    return mask


def _as_vector(value: Any, size: int, field: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"SAM3D {field} must contain {size} values, got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"SAM3D {field} contains NaN or infinity")
    return [float(item) for item in array]


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _save_meshes(
    output: Mapping[str, Any],
    object_id: int,
    local_meshes_dir: Path,
) -> tuple[Any, Any]:
    meshes = output.get("mesh")
    if meshes is None or len(meshes) == 0:
        raise RuntimeError(f"SAM3D produced no mesh for object {object_id}")

    # Import mesh dependencies only for real preparation.  Importing this
    # module and asking for --help stays independent of trimesh/PyTorch3D.
    from multi_object_decoder.mesh_io import (
        mesh_result_to_trimesh,
        transform_mesh,
    )

    local_mesh = mesh_result_to_trimesh(meshes[0])
    local_path = local_meshes_dir / f"obj_{object_id}_local.ply"
    local_mesh.export(local_path)
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Could not export local mesh for object {object_id}: {local_path}")
    scene_mesh = transform_mesh(
        local_mesh,
        output["rotation"],
        output["translation"],
        output["scale"],
    )
    return local_mesh, scene_mesh


def generate_multi_object_cache(
    inference: SAM3DPrepareInference,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    seed: int = 42,
    background_value: int = 0,
    save_image_condition_only: bool = False,
    save_tokens_only: bool = False,
) -> dict[str, Any]:
    """Run SAM3D once per visible instance and write the MOD cache contract."""

    if save_image_condition_only or save_tokens_only:
        raise ValueError("Partial-cache modes are not supported by mod-prepare")
    started = time.perf_counter()
    image = _load_rgb(Path(image_path))
    mask = _load_instance_mask(Path(mask_path), image.shape[:2])
    labels = sorted(int(value) for value in np.unique(mask) if int(value) != background_value)
    if not labels:
        raise ValueError(f"Instance mask has no object labels other than {background_value}")

    object_ids = [label - 1 for label in labels]
    if any(object_id < 0 for object_id in object_ids):
        raise ValueError(
            "Object labels must be positive because label 1 maps to obj_000; "
            f"found labels {labels} with background={background_value}"
        )
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(f"Instance labels do not map to unique object IDs: {labels}")

    output_dir = Path(output_dir)
    token_dir = output_dir / "tokens"
    local_meshes_dir = output_dir / "local_meshes"
    token_dir.mkdir(parents=True, exist_ok=True)
    local_meshes_dir.mkdir(parents=True, exist_ok=True)

    poses: dict[str, dict[str, Any]] = {}
    scene_meshes: list[Any] = []
    scene_names: list[str] = []
    label_rows: list[dict[str, Any]] = []

    for visible_index, (mask_value, object_id) in enumerate(zip(labels, object_ids)):
        object_name = f"obj_{object_id:03d}"
        object_mask = mask == mask_value
        with _temporary_environment(
            {
                "SAM3D_TOKEN_DIR": str(token_dir),
                "SAM3D_OBJ_ID": f"{object_id:03d}",
            }
        ):
            output = inference(image, object_mask, seed=seed)

        translation = _as_vector(output.get("translation"), 3, "translation")
        rotation = _as_vector(output.get("rotation"), 4, "rotation")
        scale = _as_vector(output.get("scale"), 3, "scale")
        poses[object_name] = {
            "translation": translation,
            "rotation": rotation,
            "scale": scale,
            "mask_value": mask_value,
            "object_index": object_id,
        }
        _, scene_mesh = _save_meshes(output, object_id, local_meshes_dir)
        scene_meshes.append(scene_mesh)
        scene_names.append(f"obj_{object_id}")
        label_rows.append(
            {
                "visible_index": visible_index,
                "mask_value": mask_value,
                "object_index": object_id,
                "pose_name": object_name,
                "local_mesh_name": f"obj_{object_id}_local.ply",
            }
        )

    from multi_object_decoder.mesh_io import export_mesh_scene

    complete_mesh_path = output_dir / "complete_multi_object_mesh.glb"
    export_mesh_scene(scene_meshes, scene_names, complete_mesh_path)
    label_map_path = output_dir / "object_label_map.json"
    label_map_path.write_text(
        json.dumps(
            {
                "background_value": int(background_value),
                "mask_values": labels,
                "objects": label_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    (output_dir / "time.txt").write_text(
        f"data_generation_seconds={elapsed:.3f}\n"
        f"data_generation_minutes={elapsed / 60:.3f}\n"
        f"num_objects={len(poses)}\n",
        encoding="utf-8",
    )
    return {
        "status": "success",
        "num_objects": len(poses),
        "poses": poses,
        "saved_files": {
            "complete_mesh": str(complete_mesh_path),
            "object_label_map": str(label_map_path),
        },
    }
