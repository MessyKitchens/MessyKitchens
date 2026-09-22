# Data guide

The code repository includes one attributed GSO demo with RGB, instance masks,
SAM3D cache shards, canonical meshes and small demo weights under `assets/demo/`.
See [the bundled-data guide](../assets/demo/README.md). Full benchmark/training
assets remain separate. The public MessyKitchens dataset
is hosted separately at
[Hugging Face](https://huggingface.co/datasets/MessyKitchens/MessyKitchens)
under **CC BY-NC 4.0**. The repository's Apache-2.0 license does not change the
dataset license.

The public snapshot at revision
[`edc548246682daa6f4b23460e6317847a4541271`](https://huggingface.co/datasets/MessyKitchens/MessyKitchens/tree/edc548246682daa6f4b23460e6317847a4541271),
inspected on 2026-09-06, contains 100 scenes, 1,503 RGB
views, per-scene `poses.json` and `scene.glb` files, and approximately 22.9 GB
of data. The dataset card reports 130 object models, while the inspected file
tree contains 134 object-model PLY assets; per-scene RGB counts vary and should
be enumerated from the pinned revision rather than assumed to be exactly 15.
Its top-level layout is:

```text
MessyKitchens/
├── object_models/
└── scenes/
    ├── easy/
    ├── medium/
    └── hard/
```

## Download

Install the Hugging Face CLI and download into a directory ignored by Git:

```bash
python -m pip install --upgrade huggingface_hub
hf download MessyKitchens/MessyKitchens \
  --repo-type dataset \
  --revision edc548246682daa6f4b23460e6317847a4541271 \
  --local-dir data/MessyKitchens
```

Review the dataset card and its current file layout before constructing a
manifest. Large payloads should remain under `data/` or another directory
outside the source tree.

> **Instance-mask requirement:** no instance-mask artifact was present in the
> public file tree inspected on 2026-09-06. Real processing with
> `scripts/prepare.sh` requires both an RGB image and `instance_mask_path`.
> The current dataset download alone is therefore insufficient for this stage;
> the masks or a documented mask-generation procedure still need to be
> published. `scripts/demo.sh prepare` instead processes the separately bundled GSO RGB and
> labels. It does not claim to process the complete public benchmark.

## Raw input manifest

`scripts/prepare.sh` accepts a non-empty JSON list. Each item must contain:

```json
[
  {
    "id": "scene_001__view_000",
    "scene_id": "scene_001",
    "image_path": "images/scene_001/view_000.jpg",
    "instance_mask_path": "masks/scene_001/view_000.png",
    "target_poses_path": "targets/scene_001/view_000.json"
  }
]
```

Paths are resolved relative to the manifest. `id` should be unique for every
view; `scene_id` identifies the physical scene. Mask value `0` is background.
Positive labels preserve their numeric identity: label `1` becomes `obj_000`,
label `4` becomes `obj_003`, and missing labels are not renumbered.

`target_poses_path` (or its legacy alias `gt_poses_path`) is optional for cache
generation and inference but **required for training**. It must point to an
object-keyed MOD pose JSON aligned to the mask labels and the exact SAM3D
canonical meshes. By default, data processing only creates SAM3D base poses.
The optional [training-GT stage](#generate-training-gt) registers those meshes
against supplied GT geometry; it does not infer supervision from RGB/masks or
convert the public matrix-valued `poses.json` automatically.

[`configs/example_data.json`](../configs/example_data.json) is a schema-only
example. Its referenced images and masks are not bundled.

## Processing

With the external SAM3D environment configured as described in the root
README:

```bash
./scripts/prepare.sh \
  --data-json data/input_scenes.json \
  --output-dir outputs/prepared
```

Every successful record writes tokens, image conditioning, canonical meshes,
base poses, an object-label map, and a composed mesh. The output root contains
`prepared_data.json` for inference (and for training when the source record
supplied matched targets or GT generation succeeded) plus `prepare_manifest.json` with per-record
status. Generated paths are relative by default. The whole bundle is
relocatable only if every referenced source/target file is also kept under the
moved tree; external relative paths must remain valid.

## Generate training GT

`--generate-training-gt` adds scene-level and per-object GAPs registration to
preparation. It uses the actual prepared canonical meshes and base poses, maps
the GT scene back into the SAM3D prediction frame, and writes registered
translation/quaternion/scale targets to each scene's `target_poses.json`.
`metadata.output_pose_space` is `sam3d_pred`; `prepared_data.json` points both
`target_poses_path` and `gt_poses_path` at this generated file. Existing source
target files are not modified.

You must supply a GT scene with separate geometry-bearing scene nodes and an
explicit one-to-one correspondence for **every** prepared object. For example:

```json
[
  {
    "id": "scene_001__view_000",
    "scene_id": "scene_001",
    "image_path": "images/view_000.jpg",
    "instance_mask_path": "masks/view_000.png",
    "mesh_path": "geometry/scene.glb",
    "gt_object_map": {"0": "cup_node", "3": "plate_node"}
  }
]
```

Here mask labels `1` and `4` mean SAM3D object IDs `0` and `3`; `cup_node` and
`plate_node` are illustrative **scene-graph node names**, not geometry-array
indices or filenames. Use the exact names from your scene and verify their
mask correspondence. Neither node sorting nor nearest-neighbor guessing is
used. A single merged mesh without separate object nodes is insufficient.

```bash
bash scripts/prepare.sh \
  --data-json data/input_with_gt.json \
  --output-dir outputs/prepared-with-gt \
  --generate-training-gt \
  --gaps-binary /path/to/mshalign \
  --gt-timeout 120
```

GAPs is an external native dependency, not bundled with this release; see
[its provenance and license](../THIRD_PARTY_NOTICES.md#gaps). Set `GAPS_MSHALIGN`
instead of `--gaps-binary` if preferred. Timeout is per alignment call. Missing
executables, missing/duplicate object matches, failed or nonconverged alignment,
and invalid transforms fail the scene rather than silently producing identity
targets. Inspect `prepare_manifest.json` and the scene/object registration
diagnostics in each target JSON. Convergence does **not** prove semantic
correctness or label quality: visually review registrations before training.

For the single-scene demo, put just the mapping object in `gt_object_map.json`
(for its four labels, include IDs `0`, `1`, `2`, and `3`) and use:

```bash
bash scripts/demo.sh prepare --generate-training-gt \
  --gt-mesh /path/to/matching_gt_scene.glb \
  --gt-object-map /path/to/gt_object_map.json \
  --gaps-binary /path/to/mshalign \
  --output-dir outputs/demo/with-gt
```

The demo's raw GT geometry is not included. Supply the original matching scene,
or add `--data-json` with your own RGB/mask manifest. These two `--gt-*` input
overrides are single-scene conveniences; for multiple scenes use per-record
`mesh_path` and inline `gt_object_map` in `--data-json` instead. The demo writes a
derived manifest under its output directory, rebases input paths, and removes
historical target references; it never changes the bundled/source manifest.
If existing source targets are under the requested output directory, choose a
new output location to preserve them. Do not use that output directory's
`prepared_data.json` as the raw input; keep the original raw manifest separate.
`--dry-run` previews the command and derived manifest without writing files.
This stage is not available with the synthetic-tensor `--cpu-smoke` fixture.

To add GT to a cache you already prepared locally, use the same `--output-dir`
and add `--allow-legacy-pickle`, without `--overwrite`. This reuses the trusted
SAM3D cache instead of extracting latents again; keep `SAM3D_ROOT` and
`SAM3D_PIPELINE_CONFIG` configured for the wrapper. Never enable pickle loading
for untrusted caches. Training on the newly prepared bundle is explicit:

```bash
bash scripts/demo.sh train \
  --data-json outputs/demo/with-gt/prepared_data.json \
  --allow-legacy-pickle \
  --output-dir outputs/demo/training-with-gt
```

This adds registration-derived training labels, not the original real-data
capture pipeline, automatic mask generation, or paper-exact training splits.

## Paper split status

The exact ECCV manifests have been located internally but have not yet been
sanitized or added to this repository. Do not infer a paper split from
`configs/example_data.json`:

- the 9,090-record checkpoint training manifest uses synthetic GSO scenes, not
  the public MessyKitchens real-scene snapshot;
- all 9,090 prepared token/canonical/target records were found internally, but
  raw RGB/mask/mesh inputs exist for only 6,208; all 2,882 missing raw triples
  are from the GSO-hard subset, so the exact cache cannot currently be fully
  regenerated from recovered raw inputs;
- the recovered MessyKitchens evaluation manifest has 1,055 rows but only 660
  unique `(scene_id, subscene_id, image_id)` identities; 395 inputs occur under
  two legacy storage roots;
- those 660 unique RGB inputs can be mapped to public RGB files, but the
  required masks are absent publicly;
- the internal rectified GT meshes differ from the public `scene.glb` files,
  and public matrix-valued `poses.json` does not directly match MOD's
  object-keyed translation/quaternion/scale target schema.

The release must preserve the 1,055-row manifest if that weighting defines the
reported paper protocol, while also offering a separately named 660-image
deduplicated clean split. Each needs stable string IDs, relative paths, source
revision, checksums, and an explicit statement of whether duplicates count in
metric aggregation.

## Bundled numeric shards

A prepared record may use `scene_npz_paths`, a non-empty list of relative NPZ
paths, instead of `scene_npz_path`. Each shard carries a subset of objects in
one scene. The loader checks identical keys/dtypes/features, leading object
dimensions and unique IDs across shards, then orders the complete scene by ID.
No pickle is used for bundled numeric archives. See `assets/demo/real/prepared/`
for a complete four-object example with matched canonical and target poses.
