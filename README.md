# [ECCV 2026] MessyKitchens: Contact-rich Object-level 3D Scene Reconstruction

**Junaid Ahmed Ansari\*, Ran Ding\*, Fabio Pizzati, Ivan Laptev**<br>
Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI)<br>
\* Equal contribution

[Paper](https://arxiv.org/abs/2603.16868) · [Project and videos](https://messykitchens.github.io/) · [MessyKitchens-Real dataset](https://huggingface.co/datasets/MessyKitchens/MessyKitchens)

This repository contains the official Multi-Object Decoder (MOD), including
SAM3D feature preparation, optional training-target registration, training,
inference, mesh export, and aligned-geometry metrics. The commands needed to
run these stages are collected below. Run every command from the repository
root unless a command explicitly changes directory.

## Release status

_Last updated: September 21, 2026._

**Data and model weights**

- [x] Release the MessyKitchens-Real benchmark data.
- [x] Include small MOD demo weights.
- [ ] Release the pretrained MOD paper checkpoint.
- [ ] Release the MessyKitchens-Synthetic dataset.
- [ ] Release MOD-ready benchmark masks and split manifests.

**MOD preparation, training, inference, and evaluation**

- [x] Include the MOD network, preparation, training, inference, and aligned-geometry evaluation code.
- [x] Support latent extraction and training-GT registration from supplied geometry.
- [x] Provide preparation, training, and inference demos.
- [ ] Release physical/contact metrics and the paper-exact evaluation protocol.

**Synthetic generation and real-data annotation**

- [ ] Release the Blender synthetic-data generation pipeline.
- [ ] Release the real-data acquisition and annotation pipeline, including object-to-scene registration.

The bundled weights exercise the workflow; they are not the checkpoint used
for the paper. The full paper training split, selected checkpoint, complete
evaluator, and synthetic generation pipeline remain separate release items.

## Demo inputs and outputs

The default input is the four-object synthetic GSO render
`train_0000_az_055_el_073_fr_0000`. `real/` means an original pipeline sample,
not a photograph from the MessyKitchens-Real benchmark. `portable/` contains
small generated tensors for the optional CPU check.

```text
assets/demo/
├── real/
│   ├── raw/
│   │   ├── image.png                # Original 512 × 512 RGB bytes
│   │   ├── instance_mask.png        # Original integer labels, lossless NPY → PNG
│   │   └── input.json               # Relative RGB/mask paths for fresh preparation
│   ├── prepared/
│   │   ├── prepared_data.json       # Historical cache and its matching targets
│   │   ├── scene_obj_*.npz          # Four numeric, pickle-free object shards
│   │   ├── local_meshes/            # Original canonical meshes
│   │   ├── base_poses.json
│   │   └── target_poses.json
│   ├── model.pt                    # Compact SAM3D-backed MOD; two training steps
│   └── PROVENANCE.json
├── portable/                       # Generated CPU fixture and direct-backend weights
└── MANIFEST.json                    # File sizes, SHA256 checksums, and licenses
outputs/demo/                       # Generated locally; not source data
```

Preparation consumes **raw RGB and masks**, and writes a fresh cache under
`outputs/`. The bundled historical cache makes training and inference runnable
independently. Its targets match its own canonical meshes: do not attach those
targets to a newly extracted cache. To train from fresh preparation, supply the
matching original GT geometry and regenerate targets as described below.
The current raw bundle does not include that GT scene or its object map.

### Optional hard-scene raw demo (not bundled)

An additional raw-data bundle contains 10 GSO hard scenes, 60 RGB views,
60 original NPY label maps, 10 GT scene GLBs, and 10 scene JSON files.
Its 140 source files total 521,536,330 bytes; their source/destination SHA256
checksums matched during a local transfer on September 21, 2026. Existing
feature caches, registered poses, and metrics were excluded from that transfer.
This optional bundle is not included in a Git clone of this repository.
If you already have the bundle and its relative-path manifests, place its
`raw/` directory under `assets/demo/hard10/` before running this example.
Otherwise, use the bundled four-object demo in section 2.

After setting up the GPU environment in section 1:

```bash
# Use input_one.json for one view, or input.json for all 60 views.
# For another dataset, replace --data-json with its raw-input manifest.
bash scripts/prepare.sh \
  --data-json assets/demo/hard10/raw/input_one.json \
  --output-dir outputs/demo/hard10/prepared-one \
  --seed 42
```

Manifest paths are relative to `raw/`. The original metadata incorrectly
referenced `train_0000.json`; the imported manifest uses the existing scene
JSON matching the GT mesh basename (`train_000.glb` / `train_000.json`).
All 60 image/mask pairs and prepare-loader identities have been checked.
This transfer does not establish GPU preparation or end-to-end training on
these hard scenes. An explicit, verified GT object-to-node map is still
required for `--generate-training-gt`. The local bundle records its transfer hashes and checks in
`assets/demo/hard10/TRANSFER_RECEIPT.json`; that receipt is not distributed here.

## 1. Install the GPU environment

Real image preparation and the SAM3D-backed model require Linux, an NVIDIA
GPU, the SAM3D environment, and separately downloaded SAM3D base weights.
Upstream recommends at least 32 GB VRAM. The recorded demo runtime used an
A100 40 GB, Python 3.11, PyTorch 2.5.1, and CUDA 12.1. For the generated CPU
fixture only, use the installation in section 7 instead.

```bash
git clone https://github.com/MessyKitchens/MessyKitchens.git
cd MessyKitchens

git clone https://github.com/facebookresearch/sam-3d-objects.git external/sam-3d-objects
git -C external/sam-3d-objects checkout afdf6a31522d038c44c68a0bb57aa68827380797
mamba env create -f external/sam-3d-objects/environments/default.yml
mamba activate sam3d-objects
# Replace mamba with conda if that is your environment manager.

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
python -m pip install -e 'external/sam-3d-objects[dev]'
python -m pip install -e 'external/sam-3d-objects[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
python -m pip install -e 'external/sam-3d-objects[inference]'
python external/sam-3d-objects/patching/hydra
python -m pip install -e '.[train,iou]'
```

Obtain access to the `facebook/sam-3d-objects` gated model repository on
Hugging Face and accept its SAM License before downloading. The small bundled
MOD checkpoint does not replace these base weights.

```bash
python -m pip install 'huggingface-hub[cli]<1.0'
hf auth login
hf download facebook/sam-3d-objects \
  --repo-type model \
  --local-dir external/sam-3d-objects/checkpoints/hf-download \
  --max-workers 1

export SAM3D_ROOT="$(pwd)/external/sam-3d-objects"
export SAM3D_PIPELINE_CONFIG="${SAM3D_ROOT}/checkpoints/hf-download/checkpoints/pipeline.yaml"
export MOD_PYTHON="$(command -v python)"
export SAM3D_PYTHON="${MOD_PYTHON}"
# For an existing installation, replace these values with its checkout,
# pipeline.yaml, and environment interpreter. Keep model files beside that config.
python scripts/setup_sam3d.py --sam3d-root "${SAM3D_ROOT}"
```

The last command verifies the pinned upstream commit and six integration
files. It does not download weights or run the GPU. MOD captures the original
SAM3D solver tensors and applies a runtime compatibility fix that preserves
pose-decoder gradients; it does not edit the external SAM3D source files.

## 2. Prepare features from original RGB and masks

```bash
# For other data, replace --data-json with your raw manifest.
bash scripts/prepare.sh \
  --data-json assets/demo/real/raw/input.json \
  --output-dir outputs/demo/prepared \
  --seed 42
```

This runs SAM3D separately for each foreground object. Successful scenes write
`tokens/obj_*.npz`, image conditioning, `local_meshes/`, decoded baseline poses,
and an object-label map. `outputs/demo/prepared/prepared_data.json` is the
next-stage manifest; `prepare_manifest.json` records each scene's status.
This default command generates inference inputs, without supervised targets.

Use `--continue-on-error` to collect failures across a batch. To replace an
incomplete generated scene, use the same explicit output directory with
`--overwrite`. To reuse a previously generated cache, add
`--allow-legacy-pickle` without `--overwrite`. That flag is required only for
trusted historical/local SAM3D object-array caches; the bundled numeric shards
do not need it.

The equivalent convenience entrypoint is:

```bash
bash scripts/demo.sh prepare \
  --data-json assets/demo/real/raw/input.json \
  --output-dir outputs/demo/prepared-convenience \
  --seed 42
# Append --dry-run to inspect the selected command without loading models.
```

For another dataset, create a JSON list like this. File paths are relative to
**the manifest's directory**, and each `id` must identify a unique view:

```json
[
  {
    "id": "scene_001_view_00",
    "scene_id": "scene_001",
    "image_path": "images/view_00.png",
    "instance_mask_path": "masks/view_00.png"
  }
]
```

Mask value `0` is background; foreground label `k` maps to object ID `k-1`.
For example, labels `1` and `4` map to `obj_000` and `obj_003`; missing labels
do not renumber the remaining objects. Use integer-label PNG/NPY masks, not
colored previews. `target_poses_path` is optional for inference but required
for supervised training unless targets are embedded in the numeric fixture.
Targets use object-keyed translation `[x,y,z]`, quaternion `[w,x,y,z]`, and
scale `[sx,sy,sz]` in the matching decoded SAM3D frame. The public benchmark's
world-matrix `poses.json` is not accepted as a direct substitute.

### Optional: generate targets matched to fresh preparation

This stage additionally requires the original matching GT scene with a
separate mesh-bearing node for each object, an explicit object correspondence,
and the external GAPs `mshalign` executable. A merged mesh or RGB/masks alone
cannot supply these correspondences. The original matching GT files are not
currently bundled; the following command becomes runnable once they are
provided at the indicated demo-relative paths.

Build GAPs on Ubuntu, if it is not already available:

```bash
sudo apt-get install build-essential git mesa-common-dev libglu1-mesa-dev \
  libosmesa6-dev libxi-dev libgl1-mesa-dev libglew-dev libjpeg-dev libpng-dev
git clone https://github.com/tomfunkhouser/gaps.git external/gaps
git -C external/gaps checkout f78eb23e4b44378af1472c17a65d74b2c4e63c66
make -C external/gaps/pkgs -j8
make -C external/gaps/apps/mshalign -j8
export GAPS_MSHALIGN="$(pwd)/external/gaps/bin/x86_64/mshalign"
test -x "${GAPS_MSHALIGN}"
# With an existing build, set GAPS_MSHALIGN to that executable instead.
```

The commands follow the [pinned GAPs build files](https://github.com/tomfunkhouser/gaps/tree/f78eb23e4b44378af1472c17a65d74b2c4e63c66).
This identifies inspected source, not the paper's historical binary; a fresh
build has not been verified as part of the recorded demo runs. For this four-object demo, `gt_object_map.json` must map
all IDs `"0"`, `"1"`, `"2"`, `"3"` to the exact distinct nodes in `scene.glb`.
Inspect the supplied scene's names with:

```bash
# Replace scene.glb with your matching raw GT scene.
python -c 'import trimesh; s=trimesh.load("assets/demo/real/raw/scene.glb", force="scene", process=False); print(list(s.graph.nodes_geometry))'
```

Do not infer the mapping from node order. Verify which node each mask depicts.
For example, `{"0": "cup_node", "1": "plate_node"}` illustrates the format;
it is not a valid complete mapping for this four-object demo.

```bash
# Supply the matching raw scene and verified object map at these paths.
# For another single scene, replace --data-json and both --gt-* input paths.
bash scripts/demo.sh prepare \
  --data-json assets/demo/real/raw/input.json \
  --output-dir outputs/demo/prepared-with-gt \
  --generate-training-gt \
  --gt-mesh assets/demo/real/raw/scene.glb \
  --gt-object-map assets/demo/real/raw/gt_object_map.json \
  --gaps-binary "${GAPS_MSHALIGN}" \
  --gt-timeout 120
```

For multiple scenes, place `mesh_path` and inline `gt_object_map` in each raw
manifest record and invoke `scripts/prepare.sh --generate-training-gt`, without
the two single-scene `--gt-*` overrides. Registration writes new
`target_poses.json` files and a training-ready `prepared_data.json`. It registers
the GT scene and each object to the actual prepared geometry. Missing matches,
failed/nonconverged registration, and invalid transforms fail the scene;
inspect the status manifest and target diagnostics and visually review the
registrations before using them as supervision.

## 3. Train and resume MOD

The default training example uses the bundled cache and its matching historical
targets. The compact profile keeps the SAM3D backend and the recovered
pose/Chamfer objective, with a smaller network and a short training duration.
It requires PyTorch3D, installed with the GPU environment above.

```bash
# For other prepared data, replace --data-json with its prepared manifest.
bash scripts/train.sh \
  --config configs/demo_sam3d.yaml \
  --data-json assets/demo/real/prepared/prepared_data.json \
  --output-dir outputs/demo/training \
  --device cuda \
  --max-train-steps 2

# Continue the saved model/optimizer to a total of four steps.
bash scripts/train.sh \
  --config configs/demo_sam3d.yaml \
  --data-json assets/demo/real/prepared/prepared_data.json \
  --resume outputs/demo/training/checkpoints/last.pt \
  --output-dir outputs/demo/training-resumed \
  --device cuda \
  --max-train-steps 4
```

Outputs include `checkpoints/last.pt`, `metrics.jsonl`, `resolved_config.json`,
and a training summary. Use a new output directory for a new experiment.
`--max-train-steps` counts total optimizer steps, including resumed steps.
`--precision fp32|fp16|bf16` and `--seed` override their configuration values.

After **successful target registration** in the preceding optional stage,
train directly from its new cache:

```bash
bash scripts/train.sh \
  --config configs/demo_sam3d.yaml \
  --data-json outputs/demo/prepared-with-gt/prepared_data.json \
  --output-dir outputs/demo/training-from-raw \
  --device cuda \
  --allow-legacy-pickle \
  --max-train-steps 2
# For other data, use its prepared manifest, including matching targets.
```

To exercise the full-size recovered paper architecture on the demo, select
`--config configs/pose_refiner_paper.yaml` and keep the same demo `--data-json`.
For a longer run, omit `--max-train-steps`; optional configuration overrides
follow the flags, for example `train.epochs=20`. This profile uses AdamW at
`1e-5`, accumulation `2`, translation/quaternion/raw-scale weights `100/0.1/100`,
and symmetric PyTorch3D Chamfer weight `100` with 1,024 points. Its fixed
909,000-step schedule remains unchanged by a bounded diagnostic run. The
unreleased full training split and paper checkpoint are still needed to
reproduce paper scores.

`configs/pose_refiner.yaml` is a SAM3D pose-only template;
`configs/pose_refiner_direct.yaml` trains a direct pose head from cached
features without loading the SAM3D decoder. `configs/demo_direct.yaml` matches
the generated 8-dimensional CPU fixture. Select the configuration appropriate
to your feature dimensions and backend. Paths set inside YAML or with
`train.data_json=...`/`val.data_json=...` overrides are resolved relative to the
YAML directory; the explicit `--data-json` flag uses the current directory.

## 4. Infer poses and export meshes

Inference can run immediately with the bundled model, independently of training:

```bash
# For other data, replace --data-json; for your trained model, replace --checkpoint.
bash scripts/infer.sh \
  --checkpoint assets/demo/real/model.pt \
  --data-json assets/demo/real/prepared/prepared_data.json \
  --output-dir outputs/demo/inference \
  --device cuda \
  --export-meshes
```

The scene directory is
`outputs/demo/inference/gso__train_0000_az_055_el_073_fr_0000/`.
It contains `mod_pose.json`, `posed_meshes/obj_*.ply`, and
`posed_meshes/scene.glb`. Check the root `inference_manifest.json` for scene
statuses. Omit `--export-meshes` for pose-only output. Use `--overwrite` to
replace existing predictions and `--continue-on-error` for batch diagnostics.

Run your newly trained checkpoint on the fresh raw-data cache with:

```bash
bash scripts/infer.sh \
  --checkpoint outputs/demo/training-from-raw/checkpoints/last.pt \
  --data-json outputs/demo/prepared-with-gt/prepared_data.json \
  --output-dir outputs/demo/inference-from-raw \
  --device cuda \
  --allow-legacy-pickle \
  --export-meshes
# You can also infer on outputs/demo/prepared/prepared_data.json without targets.
```

Versioned `.pt` checkpoints store their model configuration and backend.
For an old `model.pth`/`model_state_dict` checkpoint, additionally supply
`--backend sam3d --model-config configs/pose_refiner_paper.yaml`, adjusting the
configuration to match that checkpoint exactly.

## 5. Compute geometry metrics

`python scripts/metrics.py` and the installed `mod-metrics` command compute
filled-voxel IoU, symmetric non-squared surface Chamfer distance, and F-score
for meshes **already in the same frame**. They do not align or rescale meshes.

For a runnable diagnostic using the bundled scene, export its registered
training-target poses on the same canonical meshes, then compare with section
4's prediction:

```bash
python - <<'PY'
from multi_object_decoder.data import SceneDataset
from multi_object_decoder.export import export_posed_meshes
sample = next(iter(SceneDataset(
    "assets/demo/real/prepared/prepared_data.json", require_target=True)))
export_posed_meshes(sample.record, sample.object_names, sample.target_poses,
                   "outputs/demo/target-meshes")
PY

python scripts/metrics.py \
  --gt-mesh outputs/demo/target-meshes/scene.glb \
  --pred-mesh outputs/demo/inference/gso__train_0000_az_055_el_073_fr_0000/posed_meshes/scene.glb \
  --output-json outputs/demo/aligned_metrics.json \
  --num-grids 64 --scale 2.0 --num-samples 10000 \
  --f-score-threshold 0.1 --seed 0
# For other data, replace both meshes with your already-aligned GT/prediction.
# Add --iou-only to skip Chamfer distance and F-score.
```

This checks consistency with the demo's registered training targets; its GT
mesh is reconstructed from SAM3D canonical meshes, not original scene geometry.
It is not a held-out benchmark or a paper-score reproduction. Voxel pitch is
`scale / num_grids` (default `2 / 64`); `scale` controls pitch, not input size.
The F-score threshold uses the meshes' own length unit.

### Legacy paper evaluator and result reports

`scripts/evaluate.py` calls an external
`SAM3D_ROOT/scripts/validate_predicted_poses_quick.py`. That validator and its
project-specific PartCrafter/SAM3 helpers are absent from clean upstream
checkouts, so the full evaluator is **not currently a clean-clone workflow**.
It also needs GAPs and matching raw GT geometry. Installing the geometry
Python dependencies alone does not supply this missing code.

When those dependencies and GT are supplied, an evaluation manifest such as
`outputs/demo/evaluation.json` contains:

```json
[
  {
    "id": "train_0000_az_055_el_073_fr_0000",
    "mesh_path": "assets/demo/real/raw/scene.glb",
    "sam3d_results_path": "outputs/demo/inference/gso__train_0000_az_055_el_073_fr_0000",
    "ours_results_path": "outputs/demo/inference/gso__train_0000_az_055_el_073_fr_0000",
    "sam3d_canonical_dir": "assets/demo/real/prepared/local_meshes",
    "predicted_poses_json": "outputs/demo/inference/gso__train_0000_az_055_el_073_fr_0000/mod_pose.json"
  }
]
```

For these legacy evaluator/report scripts, relative paths **inside records use
the current directory**, unlike preparation manifests. Run them from the root.
The commands below are conditional on those external inputs and evaluator:

```bash
# For other data, replace --config with its evaluation manifest and results key.
python scripts/evaluate.py \
  --config outputs/demo/evaluation.json \
  --validator-script "${SAM3D_ROOT}/scripts/validate_predicted_poses_quick.py" \
  --results-key sam3d_results_path --num-samples 10000 --workers 1

python -m pip install -e '.[reports]'
python scripts/compute_stats.py \
  --config outputs/demo/evaluation.json \
  --results-key sam3d_results_path \
  --metrics-glob 'metrics.txt' \
  --combined-results-keys '' \
  --out-json outputs/demo/metrics_stats.json \
  --out-excel outputs/demo/metrics_stats.xlsx

# Legacy ours-only reporting requires ours_results_path and metrics_*.json.
# The manifest above supplies that key; metrics.txt already contains JSON.
cp outputs/demo/inference/gso__train_0000_az_055_el_073_fr_0000/metrics.txt \
  outputs/demo/inference/gso__train_0000_az_055_el_073_fr_0000/metrics_evaluation.json
python scripts/compute_mean_metrics.py \
  --config outputs/demo/evaluation.json \
  --output-xlsx outputs/demo/mean_metrics.xlsx \
  --output-filtered-json outputs/demo/evaluation-filtered.json
```

The report scripts expect the evaluator's nested scene/object metric schema;
they do not directly consume section 5's flat local-metrics JSON.
`compute_stats.py` deduplicates result directories and reports missing/failed
inputs; `compute_mean_metrics.py` retains the old ours-specific interface.
The historical external alignment helper can silently fall back to identity
on GAPs failure, so verify successful alignment and inspect exported aligned
meshes before interpreting its numbers. Physical/contact metrics remain
unreleased.

## 6. Use the public benchmark or your own dataset

The public MessyKitchens-Real snapshot includes 100 scenes with RGB, object
models, scene geometry, and world-space object poses. The inspected revision
does not include instance masks or MOD-ready evaluation/split manifests.
Downloading it alone is insufficient for `prepare.sh`; supply corresponding
integer masks and construct the raw manifest described in section 2.

```bash
hf download MessyKitchens/MessyKitchens \
  --repo-type dataset \
  --revision edc548246682daa6f4b23460e6317847a4541271 \
  --local-dir data/MessyKitchens
# Create your raw manifest, then replace the demo --data-json argument.
# Supply GT scene-node correspondences to generate matched training targets.
```

Keep raw files immutable and write prepared caches/checkpoints/predictions to
separate output directories. Moving a prepared manifest requires keeping all
of its referenced raw inputs, canonical meshes, and target files reachable.
No synthetic scene-generation or real-data annotation command is supplied
because those pipelines have not yet been released.

## 7. CPU checks and development

The CPU fixture checks the interfaces using generated tensors and a direct
pose head. Its preparation command validates and copies a precomputed fixture;
it does not run RGB-to-SAM3D processing.

```bash
MOD_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash scripts/install.sh
export MOD_PYTHON="$(pwd)/.venv/bin/python"

# Each stage has independent bundled inputs and can run in any order.
bash scripts/demo.sh infer --cpu-smoke --output-dir outputs/demo/cpu-inference
bash scripts/demo.sh train --cpu-smoke --output-dir outputs/demo/cpu-training
bash scripts/demo.sh prepare --cpu-smoke --output-dir outputs/demo/cpu-prepared
# Choose fresh output directories when repeating training.

.venv/bin/python -m pip install -e '.[train,iou,test]'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache .
# To include the native registration test, also export GAPS_MSHALIGN.
```

After CPU checks, restore `MOD_PYTHON` to the SAM3D environment's interpreter
before returning to GPU commands. The `mod-prepare`, `mod-train`, `mod-infer`,
and `mod-metrics` console commands expose the same core interfaces as the
source wrappers; all scripts support `--help`.

The release checker expects a clean source payload, so do not run its default
check against a development tree containing `.venv/` or `outputs/`. After
reviewing and committing the intended changes, check a separate clean export:

```bash
# Maintainer check: exports committed files only, without local data or environments.
MOD_RELEASE_DIR=$(mktemp -d)
git archive HEAD | tar -x -C "${MOD_RELEASE_DIR}"
.venv/bin/python scripts/dev/check_release.py "${MOD_RELEASE_DIR}"
```

The checker verifies demo hashes and the allowed source payload.
`--require-git` is for a separate clean committed clone rather than an archive;
`--paper-exact` is expected to fail until the unreleased paper artifacts are
published. Maintainer-only fixture generation/conversion tools live in
`scripts/dev/`; normal preparation uses section 2 and never needs to rebundle
cached features.

## Verification and reproducibility limits

On September 21, 2026, the current checkout passed **128 tests**, with one
optional PyTorch3D parity test skipped in the CPU environment. This includes
native GAPs registration and actual training with CLI/config-relative data
paths. Independent CPU preparation, training, resume, inference, and mesh
metrics checks also passed. All 25 bundled assets retained their recorded
sizes and SHA256 hashes; the clean source export passed the release checker.

The bundled GSO RGB and masks were processed again with the clean pinned
SAM3D source on an A100-SXM4-40GB: fresh preparation succeeded for all four
objects (104.95 s), and compact training on the historical matched cache
completed two optimizer steps (58.31 s). Bundled-checkpoint inference passed
(48.37 s), and the newly trained checkpoint also processed the fresh cache
successfully (45.87 s). Both inference runs exported all four objects and a
scene GLB, with zero failed scenes. The section 5 target-mesh comparison also
completed (5.47 s). An initial training attempt was interrupted by an
unreachable compute node; the successful follow-up ran on another A100.
GPU dependencies and base weights were reused from an existing environment;
this does not establish a fresh dependency installation or an immutable
revision for every base-weight file.

The matching original GT scene remains required to verify fresh
RGB/masks → cache → registered GT → training end to end. Native registration
tests use synthetic geometry and do not establish real-data registration
quality. No full paper benchmark scores are claimed. Machine-readable
verification summaries, including historical runs, are retained in
`docs/release/verification.json`; demo source hashes and licenses are in
`assets/demo/MANIFEST.json` and `assets/demo/real/PROVENANCE.json`.

## Attribution, citation, and license

MOD builds on SAM 3D Objects. The geometry metrics include PartCrafter-derived
code. The demo render uses Google Scanned Objects (Google Research, CC BY 4.0).
Its image, labels, cached numerical features, canonical meshes, and registered
targets retain their original numeric content; source hashes and transformations
are recorded in the demo provenance. The full raw GSO archive is not bundled.

Source code and generated CPU fixtures are Apache-2.0. The synthetic GSO demo
data are CC BY 4.0 (`assets/demo/LICENSE-DATA`); the bundled SAM3D-backed MOD
checkpoint includes SAM-derived components and uses the SAM License
(`assets/demo/LICENSE-SAM`). External SAM3D source/base weights have their own
SAM License. MessyKitchens-Real is separately distributed under CC BY-NC 4.0.
The code license does not relicense any of these datasets or external weights.
`LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` retain the full attribution and
incorporated-code notices.

```bibtex
@inproceedings{ansari2026messykitchens,
  title     = {MessyKitchens: Contact-rich Object-level 3D Scene Reconstruction},
  author    = {Ansari, Junaid Ahmed and Ding, Ran and Pizzati, Fabio and Laptev, Ivan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.16868}
}
```
