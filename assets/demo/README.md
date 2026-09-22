# Bundled demonstration data

The default demos use one complete **synthetic GSO training render** and its
original four-object SAM3D cache. This is processed research data from the MOD
training workflow, not a camera photograph from the public MessyKitchens test
set. The directory name `real/` distinguishes this original pipeline sample
from the optional generated-tensor CPU smoke test.

Sample: `train_0000_az_055_el_073_fr_0000`, from the `GSO_easy` training data.
The original RGB is 512 × 512. The background is mask label 0; labels 1, 2, 3,
and 4 correspond to cached object IDs 0, 1, 2, and 3. All four objects are
included in every prepared-scene load.

## Independent demo inputs

| Demo | Bundled input | Purpose |
|---|---|---|
| Data processing | `real/raw/input.json` | Run SAM3D on the original RGB and instance labels to create a fresh cache. |
| Training | `real/prepared/prepared_data.json` | Train using the historical cache, matching canonical meshes, and original registered target poses. |
| Inference | `real/prepared/prepared_data.json` | Refine the historical cached scene and export its geometry using a MOD checkpoint. |

The fresh data-processing manifest intentionally contains **no target-pose
path**. The historical targets were registered to the bundled canonical meshes.
If SAM3D is run again with another seed, implementation, or weights, its meshes
can differ. Those fresh meshes require their own ground-truth registration
before they become supervised training inputs. Completing the processing demo
does not establish that its new cache can reuse the historical training targets.

To generate targets matched to fresh meshes, use `scripts/demo.sh prepare
--generate-training-gt` with `--gt-mesh`, `--gt-object-map`, and `--gaps-binary`.
The raw matching GT scene is **not bundled**; you must provide it, or pass
`--data-json` for your own RGB/masks and GT geometry. The mapping JSON explicitly
associates each mask-derived object ID with a GT scene node (all four demo IDs
`0`–`3` are required). Registration writes new `sam3d_pred`-frame targets and a
training-ready manifest without attaching historical targets to fresh meshes.
See [the complete preparation/training example](../../docs/DATA.md#generate-training-gt).
The default preparation demo remains cache-only; `--cpu-smoke` does not support
GT registration.

## Layout and numeric contract

- `real/raw/image.png`: original RGB bytes, without resizing or recompression.
- `real/raw/instance_mask.png`: the complete original integer labels, converted
  losslessly from NPY to an 8-bit grayscale PNG. This is not the colored preview.
- `real/raw/input.json`: relative paths for fresh data processing.
- `real/prepared/scene_obj_000.npz` through `scene_obj_003.npz`: pickle-free
  compressed numeric shards of **one scene**, split by object for Git file limits.
- `real/prepared/local_meshes/obj_*_local.ply`: all original canonical vertices
  and triangle faces, with visual attributes omitted. Geometry is not sampled.
- `real/prepared/base_poses.json`: original SAM3D decoded base poses.
- `real/prepared/target_poses.json`: original GT-registered training targets;
  private filesystem metadata has been removed and pose values preserved.
- `real/PROVENANCE.json`: original-source hashes, label/object/target mapping,
  geometry counts, and conversion details.

Each object retains 4 pose tokens of dimension 1024, 4096 shape tokens of
dimension 1024, and 7528 image-condition tokens of dimension 1024. The latent
arrays remain float32 and image-condition arrays remain float16, as stored in
the original cache. No tokens were truncated and no numeric precision was
reduced. The converter checks exact array/dtype equality and exact mesh
vertices/faces after writing.

The manifest's `scene_npz_paths` loads the four shards as one numerically ordered
four-object scene. It rejects duplicate object IDs and mismatched fields,
dtypes, or feature dimensions. Existing single-file `scene_npz_path` manifests
remain supported. Runtime loading does not enable pickle.

Poses use translation `[x, y, z]`, quaternion `[w, x, y, z]`, and scale
`[sx, sy, sz]` in the decoded SAM3D scene frame used by the original registered
cache. These are canonical-specific training targets, not the public dataset's
world-coordinate pose matrices or a new calibrated metric-ground-truth export.

## Source and attribution

The source object dataset is Google Scanned Objects, provided by Google
Research under CC BY 4.0:
[official dataset announcement](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/),
[dataset paper](https://research.google/pubs/google-scanned-objects-a-high-quality-dataset-of-3d-scanned-household-items/).
The MessyKitchens authors generated the synthetic render, instance labels,
registered targets, and cached SAM3D outputs. These prepared data are released
under CC BY 4.0; see `LICENSE-DATA`. The data contain no original GSO model archive. The separate
`real/model.pt` file is a small SAM3D-backed MOD checkpoint and includes
SAM-derived components; it is distributed under `LICENSE-SAM`, not CC BY 4.0.
The full external SAM3D base weights are not bundled. See
[third-party notices](../../THIRD_PARTY_NOTICES.md).

The original synthetic scene manifest identifies this as a GSO render. Its
surviving GLB uses generic geometry names; individual original GSO catalog IDs
were not recovered and are not invented here. Object IDs in this bundle refer
to the recorded mask/cache/registered-target correspondence, not GSO catalog IDs.

## Rebuilding from the author's source artifacts

The one-time converter requires an explicitly trusted original cache because
its historical NPZ object mappings use pickle. It writes numeric-only NPZ
shards, removes private paths, verifies each conversion, and checks a 170 MiB
dataset budget and 50 MiB per-object limit:

```bash
python scripts/dev/bundle_demo_data.py \
  --source-manifest /path/to/source_training_manifest.json \
  --record-index 0 \
  --output-dir /path/to/new_empty_demo_bundle \
  --trust-legacy-cache
```

The public bundle records source checksums so the exact inputs can be verified;
the complete internal training dataset is not required to run the bundled demos.
The data are a single training example for exercising the workflow, not a
held-out evaluation set or evidence of model quality.

## Checkpoints and the offline fixture

`real/model.pt` contains the small SAM3D-backed MOD model after two supervised
training steps on this scene. It omits the optimizer, records the model
configuration and source-checkpoint checksum, and is not a paper-quality model.
Inference always selects this fixed checkpoint unless `--checkpoint` is explicit.

`portable/` contains two generated numerical scenes, box meshes, and a tiny
direct-backend checkpoint trained for eight CPU steps. These original test
fixtures are Apache-2.0 and are selected only with `--cpu-smoke`. They permit
CPU CI to run inference before training without network downloads.

`MANIFEST.json` records every bundled input artifact's exact SHA256, size,
license, and source. The release checker allows these specific binary files;
other caches and checkpoints remain excluded from the release payload.
