"""MessyKitchens Multi-Object Decoder (MOD).

The package contains the MOD pose refiner, checked scene loading for cached
SAM3D artifacts, pose utilities, training objectives, the optional SAM3D
backend, and the installed ``mod-*`` command-line entry points.
"""

from .models import (
    SAM3DPoseRefinementModel,
    SAM3DPoseRefiner,
    SimpleMLPRefiner,
    create_sam3d_pose_refinement_model,
    create_sam3d_pose_refiner,
    create_simple_mlp_refiner,
    DirectPoseRefinementModel,
    create_direct_pose_refiner,
)
from .object_ids import object_id_sort_key, parse_prefixed_object_id
from .data import ArtifactSchemaError, DecodeState, SceneDataset, SceneSample
from .poses import apply_pose_delta, load_pose_json, pose_loss, save_pose_json

__all__ = [
    "SAM3DPoseRefinementModel",
    "SAM3DPoseRefiner",
    "SimpleMLPRefiner",
    "create_sam3d_pose_refinement_model",
    "create_sam3d_pose_refiner",
    "create_simple_mlp_refiner",
    "DirectPoseRefinementModel",
    "create_direct_pose_refiner",
    "object_id_sort_key",
    "parse_prefixed_object_id",
    "ArtifactSchemaError",
    "DecodeState",
    "SceneDataset",
    "SceneSample",
    "apply_pose_delta",
    "load_pose_json",
    "pose_loss",
    "save_pose_json",
]

__version__ = "0.2.1"
