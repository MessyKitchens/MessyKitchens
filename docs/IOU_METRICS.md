# Geometry metrics

This release has two distinct geometry-evaluation paths:

1. `mod-metrics` is self-contained and computes IoU, Chamfer distance (CD),
   and F-score for two meshes that are **already in the same coordinate
   frame**. It does not align, normalize, rotate, translate, or rescale them.
2. `scripts/evaluate.py` preserves the legacy orchestration interface for the
   paper protocol. It is not clean-clone runnable because some project-local
   validator/helper files have not yet been published.

Do not compare numbers from these paths unless the meshes, alignment protocol,
voxel settings, and sampling settings are identical.

## Local geometry metric code

All code needed for the already-aligned path is included in this release:

- `src/multi_object_decoder/geometry_metrics.py`
  - `load_mesh`: loads one triangle mesh; a GLB scene is transformed to world
    coordinates and concatenated.
  - `get_voxel_set` and `compute_iou`/`compute_IoU`: filled-voxel IoU.
  - `compute_chamfer_distance`: deterministic, non-squared, symmetric surface
    CD using SciPy `cKDTree`.
  - `compute_f_score`: symmetric surface F-score at a distance threshold.
  - `compute_geometry_metrics`: returns the combined JSON-ready result.
- `src/multi_object_decoder/commands/metrics.py`: argument parsing and JSON
  output for the installed `mod-metrics` command.
- `scripts/metrics.py`: source-tree wrapper for the same command.

The IoU convention matches the locally recovered paper-working-tree
implementation:

```text
pitch = scale / num_grids             # default: 2 / 64 = 0.03125
IoU = |filled_voxels_gt ∩ filled_voxels_pred|
      / |filled_voxels_gt ∪ filled_voxels_pred|
```

`scale` controls voxel pitch; it does **not** rescale either input mesh. Mesh
coordinates and the F-score threshold use the input files' length unit.

## Install metric dependencies

From the release root, install the local IoU/CD/F-score stack with:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[iou]'
.venv/bin/python -m pip check
.venv/bin/python -c 'import numpy, scipy, trimesh; print("geometry dependencies OK")'
```

Package roles are:

- `numpy`: voxel indices and numerical arrays.
- `trimesh`: GLB/PLY/OBJ loading, surface sampling, and filled voxelization.
- `scipy`: nearest-neighbor queries used by the local CD/F-score code and
  deterministic surface-distance calculations.

Install the optional Excel reporting dependency with:

```bash
.venv/bin/python -m pip install -e '.[reports]'
```

This adds `openpyxl` for the aggregation scripts' optional `.xlsx` output. It
does not install or imply availability of the incomplete paper evaluator.

## Run local metrics on aligned meshes

After editable installation, these two commands are equivalent:

```bash
.venv/bin/mod-metrics \
  --gt-mesh /path/to/aligned_gt.glb \
  --pred-mesh /path/to/aligned_pred.glb \
  --output-json outputs/aligned_metrics.json \
  --num-grids 64 \
  --scale 2.0 \
  --num-samples 10000 \
  --f-score-threshold 0.1 \
  --seed 0

.venv/bin/python scripts/metrics.py \
  --gt-mesh /path/to/aligned_gt.glb \
  --pred-mesh /path/to/aligned_pred.glb \
  --output-json outputs/aligned_metrics.json
```

Use `--iou-only` to skip CD and F-score. The JSON contains `iou`, voxel
intersection/union counts, pitch, and, unless skipped, CD and F-score.

When the missing evaluator implementation is supplied, it writes aligned mesh
pairs. They can be
recomputed with the local command as a useful audit:

```bash
.venv/bin/mod-metrics \
  --gt-mesh outputs/eval/scene_001/matched_aligned_gt.glb \
  --pred-mesh outputs/eval/scene_001/matched_aligned_pred.glb \
  --output-json outputs/eval/scene_001/local_aligned_metrics.json
```

This recomputes metrics on the saved aligned files; it does not prove that the
preceding alignment was valid. Inspect the GAPS checks and risks below.

## Paper geometry evaluator interface (incomplete release)

The release wrapper is `scripts/evaluate.py`. The numerical and alignment code
it loads is not fully included:

- `${SAM3D_ROOT}/scripts/validate_predicted_poses_quick.py`
  - scene loading, pose application, three alignment attempts, best-IoU
    selection, index/Hungarian object matching, and result export.
- sibling `partcrafter_ran/src/utils/metric_utils.py`
  - locally recovered `compute_chamfer_distance` and `compute_IoU` formulas.
- sibling `partcrafter_ran/src/utils/eval_utils.py`
  - GAPS alignment and transform extraction.
- sibling `sam3/examples/sam3d_ran_utils.py`
  - SAM3D pose transforms, mesh export, and visualization helpers.
- `SSR-code/external/ldif/gaps/bin/x86_64/mshalign`
  - native global Sim(3) alignment executable.

Important provenance boundary:

- `validate_predicted_poses_quick.py` is untracked in the authors' local
  SAM3D checkout and is absent from a clean upstream clone;
- official PartCrafter at the recorded inspected revision has
  `metric_utils.py` but no `eval_utils.py`;
- `sam3d_ran_utils.py` is a project-local helper with no public source location
  recorded in this release;
- the local `SSR-code` tree is ignored by the parent repository.

Consequently, reproducing the sibling directory layout below with stock
upstream clones is insufficient. Treat the command as documentation of the
interface until the narrow missing sources or a pinned evaluator fork are
published under compatible terms.

The authors' working-tree layout was similar to:

```text
submodules/
├── sam-3d-objects/
├── partcrafter_ran/
├── sam3/
└── SSR-code/
```

There is currently no supported clean-checkout package extra for this path.
The recovered external helpers import `torch`, Pillow, OmegaConf, Accelerate,
PyTorch3D, SAM3D, and additional project-local code, but installing those
packages still does not supply the missing source files. The `iou` extra is
only for the self-contained aligned-mesh command, and `reports` only adds Excel
output. GAPS is not a Python wheel.

A minimal evaluation manifest is:

```json
[
  {
    "id": "scene_001",
    "mesh_path": "/absolute/path/to/gt_scene.glb",
    "sam3d_results_path": "/absolute/path/to/inference/scene_001",
    "sam3d_canonical_dir": "/absolute/path/to/local_meshes",
    "predicted_poses_json": "/absolute/path/to/inference/scene_001/mod_pose.json"
  }
]
```

Run the wrapper from the release root:

```bash
export SAM3D_ROOT=/absolute/path/to/submodules/sam-3d-objects

.venv/bin/python scripts/evaluate.py \
  --config /absolute/path/to/eval.json \
  --validator-script "${SAM3D_ROOT}/scripts/validate_predicted_poses_quick.py" \
  --results-key sam3d_results_path \
  --num-samples 10000 \
  --workers 4
```

The config may be a list or contain a list under `items`, `data`, or `samples`.
For SAM3D/MOD results, each record needs a GT `mesh_path`, a results directory,
canonical local meshes, and a pose JSON. For MIDI or PartCrafter, select its
results key and place `completed_meshes.glb` in each results directory.

The external validator writes `metrics.json`, `GT.glb`,
`matched_aligned_gt.glb`, and `matched_aligned_pred.glb`. The release wrapper
also writes `metrics.txt` containing the same metrics dictionary in JSON
format. Aggregation uses `metrics*.json`, not `metrics.txt`.

## GAPS setup and alignment validity

Once the missing evaluator sources are available, GAPS can be built on Ubuntu
with the required native libraries:

```bash
sudo apt-get install build-essential git mesa-common-dev libglu1-mesa-dev \
  libosmesa6-dev libxi-dev libgl1-mesa-dev
cd /absolute/path/to/SSR-code/external
bash build_gaps.sh
```

Verify the exact executable before evaluation:

```bash
GAPS_BIN=/absolute/path/to/SSR-code/external/ldif/gaps/bin/x86_64/mshalign
test -x "${GAPS_BIN}"
file "${GAPS_BIN}"
if ldd "${GAPS_BIN}" | grep -q 'not found'; then
  echo "GAPS has unresolved shared libraries" >&2
  exit 1
fi
```

Important reproducibility warning: the current external
`partcrafter_ran/src/utils/eval_utils.py` contains a developer-machine absolute
`SSR-code` path inside `align_merged_meshes_with_gaps_and_get_transform`.
Patch or configure that external file to point to the checkout being evaluated;
setting `SAM3D_ROOT` alone does not change the hard-coded GAPS path.

That external alignment function returns the original meshes plus an identity
transform when GAPS is missing, times out, exits unsuccessfully, or transform
extraction fails. The calling validator may consequently write raw unaligned
metrics instead of terminating. Never report those numbers as paper-protocol
aligned IoU/CD. Check all of the following for every run:

- the log states that GAPS succeeded and a non-fallback transform was parsed;
- `matched_aligned_gt.glb` and `matched_aligned_pred.glb` are visually and
  numerically aligned;
- the wrapper's final `Finished: successful/total` count equals the requested
  sample count.

The batch wrapper returns process exit code `0` only when every selected record
succeeds, `1` when any record fails, and `2` when the manifest selects no work.
The final success count and per-scene files remain authoritative evidence for
what was actually evaluated.

## Aggregate geometry results

Use `scripts/compute_stats.py` as the primary geometry aggregation entry. It
reads one `metrics*.json` from each unique results directory, computes
scene-level and object-level predicted/refined CD and IoU statistics, and can
write JSON and Excel:

```bash
.venv/bin/python scripts/compute_stats.py \
  --config /absolute/path/to/eval.json \
  --results-key sam3d_results_path \
  --metrics-glob 'metrics*.json' \
  --prefer-latest \
  --out-json outputs/metrics_stats.json \
  --out-excel outputs/metrics_stats.xlsx
```

Without `--prefer-latest`, multiple matching files are resolved by sorted
filename. Excel export requires `openpyxl`, included in `.[reports]`. The
script can also build a combined per-sample Excel sheet across
`sam3d_results_path`, `ours_results_path`, `midi_results_path`, and
`partcrafter_results_path`.

`scripts/compute_mean_metrics.py` is a legacy, ours-specific aggregator. It
requires a top-level JSON list, reads only `ours_results_path`, and matches only
`metrics_*.json` (not plain `metrics.json`). Prefer `compute_stats.py` for new
evaluation runs.

The recovered physical/contact metric implementation is not included in this
release candidate because its contributor provenance and redistribution
license have not yet been confirmed. Publish it only after that review, with
its own tests, license statement, and exact paper settings.
