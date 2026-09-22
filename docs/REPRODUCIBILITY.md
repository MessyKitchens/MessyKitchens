# Reproducibility status

This document separates interfaces that can be verified from a clean checkout
from artifacts that are still required to reproduce the ECCV paper numbers.

## What is included

The default demos use a complete four-object synthetic GSO render and original
SAM3D cache, paired with registered target poses. The raw data, lossless numerical
shards, canonical meshes and a two-step small SAM3D-backed MOD checkpoint are
bundled. Each stage runs independently; [demo data notes](../assets/demo/README.md)
explain the historical-canonical GT boundary.

| Capability | Status | Verification |
|---|---|---|
| RGB/mask to fresh SAM3D cache | Included | A100 run: 1 scene, 4 objects succeeded |
| Small SAM3D-backed training | Included | A100 run: 2 optimizer steps with pose/Chamfer objective |
| Independent inference/export | Included | A100 run: 1 scene, 4 objects and GLB succeeded |
| CPU direct-backend fixture | Included | Bundled input/weights; independent-stage tests |
| Aligned-mesh IoU, CD, F-score | Included | CPU metric tests |
| Clean upstream integration | Included | Pinned source + scoped runtime observer; no source patch |
| Full public MessyKitchens snapshot | External | RGB/pose/mesh data; full masks are still missing |

The small model is not the paper checkpoint. The GSO demo is a training example,
not held-out evaluation. Exact demo artifacts and runtimes are recorded in
[VERIFICATION.md](release/VERIFICATION.md) and `assets/demo/MANIFEST.json`.

## Artifacts still required for paper-exact reproduction

| Artifact | Current public status | Required release metadata |
|---|---|---|
| ECCV MOD checkpoint | Exact internal candidate located; not public | URL, revision, SHA256, license |
| Exact train/evaluation splits | Internal candidates located; not sanitized | Versioned relative-path manifests and SHA256 |
| Instance masks or mask-generation recipe | Not found in public dataset | Versioned masks/procedure, mapping, and checksum |
| Camera-ready SAM3D changes | Local checkout differs from upstream | Public fork commit or redistributable patch |
| Full evaluator stack | Not publicly reconstructible | Publish the local validator/helpers, pinned revisions, and GAPS provenance |
| Physical/contact metric code | Withheld pending provenance review | Confirm contributor authorization, license, settings, and tests |
| Expected paper metrics | Not recorded here | Metric table, command, split, seed, tolerance |
| Clean GPU reproduction | Not yet recorded | OS, GPU, driver, CUDA, PyTorch, runtime, result |

## Recovered paper training objective

`configs/pose_refiner_paper.yaml` is a path-sanitized transcription of the
selected checkpoint run: AdamW at `1e-5`, gradient accumulation `2`, 200
epochs, 500 warmup steps, and a fixed 909,000-step cosine horizon. Its
translation, sign-invariant quaternion, and raw-scale losses have weights
`100/0.1/100`; PyTorch3D symmetric Chamfer distance has weight `100` and uses
1,024 points. The selected checkpoint is an early step-2,000 snapshot, not a
completed 909,000-step run.

The profile deliberately fails unless `workflow.backend=sam3d`, raw scale MSE,
and `chamfer.backend=pytorch3d` are explicit. Canonical NPY/PLY files are
matched to cached numeric object IDs without zero-filled fallbacks. PLY
sampling is deterministic and in-memory; a bounded 16-scene LRU prevents the
CPU point cache from growing with the dataset, and training never writes point
caches into the data directory. Validation retains the recovered pose-only
objective (raw scale MSE, no training CD term).

## Recovered internal candidates (not release artifacts)

The following provenance was recovered during the release audit. It is
recorded so that the files can be verified after they are sanitized and
uploaded; none of these artifacts is included in this checkout.

| Candidate | Records/size | SHA256 | Remaining work |
|---|---:|---|---|
| MOD checkpoint | 336,133,558 bytes | `aaf6e3fb5256ca7be8dd9f5472a85309b9194d868ea997e05556194f507ddf22` | Confirm license, convert or document safe loading, upload |
| Checkpoint training parameters | — | `31c667f0483908e9fab6055d5fd7becb354e9673f4157c196eeb02c58bafb29a` | Remove developer paths and compare with public config |
| Exact checkpoint training split | 9,090 synthetic GSO records | `04c69dc82b3b666287b2f17036a3c34d49a899bcb0cc65cf375b34748fbade4a` | Publish its prepared cache and rewrite resources as portable relative paths; raw inputs survive for only 6,208 |
| Training-time external validation split | 10 | `ea04c2a947e021a95c73d5ad5b94ec7c4fcd1b626c15ee43407ff84129206bf2` | Identify external dataset terms; do not call it the benchmark test split |
| MessyKitchens evaluation manifest | 1,055 rows / 660 unique image IDs | `e4fb98bd9e06c8af968ba373c49cad523b87a05b91f693bd65688133d622ee68` | Confirm duplicate weighting, assign stable IDs, rewrite paths, publish masks/rectified GT |

A real three-object RGB/mask/GT-mesh sample has also been located internally
as a possible end-to-end demo. It must be converted to a compact portable
bundle and its CC BY-NC 4.0 attribution/redistribution scope confirmed before
being added. The new bundled GSO demo is separate from this real-camera candidate. A later after-ECCV cleaning dataset exists, but it is
not the exact split used to train the checkpoint above and must remain labeled
separately.

The repository is a **code-and-small-data-demo release**, not a paper-exact
reproduction package. The unresolved rows apply to the full paper artifacts. `docs/release/manifest.json`
is the machine-readable source of the same status.

## External provenance

- SAM 3D Objects upstream:
  `https://github.com/facebookresearch/sam-3d-objects`, local baseline
  `afdf6a31522d038c44c68a0bb57aa68827380797`. The locally recovered checkout
  has unpublished modifications and is governed by Meta's SAM License; its
  exact relationship to the final paper runs still needs author confirmation.
- PartCrafter upstream: `https://github.com/wgsxm/PartCrafter`, inspected
  revision `3d773bf02fad51c7ab31a5615573fec93b287b30`, MIT.
- GAPS upstream: `https://github.com/tomfunkhouser/gaps`, current inspected
  revision `f78eb23e4b44378af1472c17a65d74b2c4e63c66`, MIT. The exact local GAPS
  build revision used for paper evaluation has not been recovered.

See `THIRD_PARTY_NOTICES.md` for licensing. Revisions described as
"inspected" are documentation anchors; they must not be presented as the
paper's exact revision unless the experiment provenance confirms that fact.

## Release validation

Follow [RELEASING.md](release/README.md) for clean source/export/distribution checks.
The ordinary release gate validates the bundled demos and source payload.
`--require-git` checks committed file coverage and cleanliness. `--paper-exact`
adds the independently unmet full-paper asset/protocol requirements.
