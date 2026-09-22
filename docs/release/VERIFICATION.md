# Demo verification

## Training-GT preparation update — 2026-09-15

The optional `demo prepare --generate-training-gt` workflow was checked with
Python 3.12.3 and PyTorch 2.5.1+cpu. The full suite passed **125 tests**, with
one optional PyTorch3D parity test skipped. This includes:

- fake-SAM3D preparation through fresh registered targets and the training
  loader, non-contiguous object IDs, trusted cache reuse without model reload,
  and exclusion of failed registrations from the output manifest;
- transform conventions with non-identity rotations and anisotropic base
  scales, explicit scene-node correspondences and hierarchical world transforms;
- missing/failed/timed-out/nonconverged GAPs calls, invalid matrices, atomic
  target writes, and source manifest/target overwrite protection;
- native GAPs scene and per-object registration on deterministic asymmetric
  synthetic geometry, as well as the independent existing CPU demos.

The tested `mshalign` SHA256 was
`4b35ae0e0045998cd677eb8f899c3343e6bc25c7a51f36131ee169f8f5aa9d28`.
Enable the optional native test with your own executable:

```bash
GAPS_MSHALIGN=/path/to/mshalign PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -q -p no:cacheprovider
```

The changed Python files passed Ruff, and the filesystem release gate passed
with 113 payload files and 51 Python files. This is a local working-tree
validation, not a clean committed/published release or a new distribution
build. The newly generated targets have **not** been used for a fresh real-GPU
SAM3D-to-training run. The native fixture verifies the registration interface,
not real-data correspondence quality, convergence to a correct semantic pose,
or paper-exact targets. Raw GT geometry and object correspondences remain
user-supplied; they are not inferred from RGB/masks.

## Historical GPU and release measurements

The following measurements were taken on 2026-09-07. They verify the small-data
workflows and do not reproduce paper benchmark scores.

## Real SAM3D workflows

The external source was a clean checkout of
`afdf6a31522d038c44c68a0bb57aa68827380797`; the source verifier checked all six
integration files against that commit. No local SAM3D patch was used. Hardware:
one NVIDIA A100-SXM4-40GB, driver 570.195.03, Python 3.11.0, PyTorch 2.5.1+cu121.

| Stage | Result | Wall time including process/model setup |
|---|---|---:|
| Training | Two optimizer steps, four objects, pose/Chamfer objective, checkpoint saved | 47.62 s |
| Inference | Bundled checkpoint, one scene/four objects, pose JSON, four PLYs and scene GLB | 43.33 s |
| Data processing | Original RGB and four instance labels, fresh cache and manifest, zero failures | 82.03 s |

These are independent runs: training/inference use the fixed historical cache
and its registered target poses; fresh processing uses RGB/mask without GT.
No benchmark claim is made about the two-step model. The training loss after
two steps was 0.553384; it is an in-sample workflow diagnostic.

The separate model-loading preflight measured 12.76 GiB of peak allocated
CUDA memory. This is **not** the peak for full processing or training, and no
minimum-VRAM claim is inferred from it. Test receipts are in
[verification.json](verification.json); input hashes and licenses are in
[the demo manifest](../../assets/demo/MANIFEST.json).

## CPU and release checks

The CPU regression run passed 62 tests, with one optional PyTorch3D parity test
skipped. Tests include inference before training, alternate working directories,
lossless four-object shard loading, object-ID/target consistency, immutable demo
inputs, malformed shards, checkpoint handling, and negative release-gate cases.
The small portable checkpoint was trained for eight CPU steps and contains no
SAM3D weights.

The release gate validates exact file hashes and committed payload boundaries;
see [RELEASING.md](README.md) for clean export and distribution verification.
The implementation was frozen as commit
`91891a29d60c9fac7ca81693cc54a7a39fe43167`. A fresh Git clone passed
`--require-git`; the source distribution and wheel passed build/Twine checks.
All 25 bundled artifacts matched their sizes and hashes after source-package
extraction. The extracted package ran CPU inference, training, and preparation
independently in that order, and the wheel installed/imported successfully.
The original required-source paths in that JSON record refer to the layout at
that commit. Current layout checks are recorded below.

GPU validation reused an existing SAM3D Python environment and base weights;
it did not reinstall all GPU dependencies from scratch. The actual pipeline
configuration hash is recorded in the JSON receipt. An exact immutable revision
for all downloaded base-weight files could not be established from surviving
cache metadata, so no such weights pin is claimed. Full paper assets remain separately gated by `--paper-exact`.

## Concise layout validation

The consolidated `scripts/demo.sh` entrypoint and relocated `assets/demo/`
inputs were checked on 2026-09-07. All 62 CPU tests passed, with the same one
optional PyTorch3D parity test skipped. Seven help invocations from outside the
repository, three SAM3D command dry runs, shell syntax, lint, local documentation
links, and the release gate passed.

A source snapshot was built into a source distribution and wheel; both passed
Twine checks. All 25 bundled payloads retained their original hashes and sizes
after extraction. The extracted source package independently ran CPU inference,
training, and preparation in that order. `layout_revision` in
[verification.json](verification.json) records the tested source-tree identity.
The final receipt updates only documentation relative to that tested tree.

The model and runtime logic were unchanged; the sole core-source edit updates
a script path in an error message. GPU workloads were not rerun for the layout
change; the original GPU measurements above remain associated with their
original implementation and environment.
