# Workflows on your own data

The three production stages accept explicit input and output paths. Set up the
[pinned SAM3D environment](SAM3D.md) for real processing and the `sam3d` backend.

## Data processing

Create a JSON list with stable scene/view IDs and paths relative to that JSON:

```json
[
  {
    "id": "scene_001_view_00",
    "scene_id": "scene_001",
    "image_path": "images/view_00.png",
    "instance_mask_path": "masks/view_00.png",
    "target_poses_path": "targets/view_00.json"
  }
]
```

Mask 0 is background. Positive label `k` maps to object ID `k-1`; absent labels
do not renumber later objects. A target file is optional for inference, but
required for supervised training. It must contain object-keyed MOD poses;
this command does not infer GT or automatically convert the public benchmark's
world-matrix JSON. See [DATA.md](DATA.md).

```bash
./scripts/prepare.sh \
  --data-json data/input_scenes.json \
  --output-dir outputs/prepared \
  --sam3d-root "${SAM3D_ROOT}" \
  --sam3d-config "${SAM3D_PIPELINE_CONFIG}"
```

Each scene writes a checked SAM3D cache, image condition, canonical object
meshes, baseline poses and object label mapping. The root `prepared_data.json`
is the next stage's input; `prepare_manifest.json` records success/failure.
`--overwrite` explicitly replaces an incomplete scene output; source inputs
are protected. `--continue-on-error` collects per-scene failures.

## Training

```bash
./scripts/train.sh \
  --config configs/pose_refiner_paper.yaml \
  --output-dir outputs/training \
  --device cuda \
  --allow-legacy-pickle \
  --sam3d-root "${SAM3D_ROOT}" \
  --sam3d-config "${SAM3D_PIPELINE_CONFIG}" \
  train.data_json=outputs/prepared/prepared_data.json
```

The paper profile selects the recovered SAM3D architecture and pose/Chamfer
objective. Full paper data and the selected paper checkpoint are separate
unpublished artifacts; this command alone is not a paper-score reproduction.
Resume a versioned checkpoint using `--resume`. Use `--max-train-steps` for a
bounded check without changing the paper scheduler horizon.

The included demo uses `configs/demo_sam3d.yaml`, which changes model capacity
and training duration for the small scene. It retains the same backend and
supervised objective. `configs/demo_direct.yaml` belongs to the separate CPU
interface fixture.

## Inference

```bash
./scripts/infer.sh \
  --checkpoint outputs/training/checkpoints/last.pt \
  --data-json outputs/prepared/prepared_data.json \
  --output-dir outputs/inference \
  --device cuda \
  --allow-legacy-pickle \
  --sam3d-root "${SAM3D_ROOT}" \
  --sam3d-config "${SAM3D_PIPELINE_CONFIG}" \
  --export-meshes
```

Versioned checkpoints carry their backend and model configuration. Legacy
`model_state_dict` files need explicit `--backend sam3d --model-config ...`.
Outputs contain a per-scene pose JSON, optional PLY/GLB exports, and an inference
manifest. Inspect scene statuses before aggregating results.

## Cache compatibility

Bundled demo numerical archives are pickle-free. Historical per-object SAM3D
NPZ caches may contain pickle-backed object arrays; loading those requires
`--allow-legacy-pickle` after verifying their source. Newly generated caches
are validated within the preparation process, but their legacy per-object
format requires the same explicit opt-in when loaded by a later command.
The production examples above assume a cache you generated locally. The
bundled numeric demo shards do not require this flag. Keep raw paths, IDs and target
poses with a cache when moving it to another computer.

## Geometry metrics

```bash
mod-metrics   --gt-mesh examples/aligned_gt.glb   --pred-mesh examples/aligned_prediction.glb   --output-json outputs/geometry.json --seed 0
```

Inputs must already share a coordinate frame. This local command does not
perform the full paper's Sim(3) alignment. See [IOU_METRICS.md](IOU_METRICS.md)
and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the distinction.
