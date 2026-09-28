# Pinned SAM3D environment

The real demos require Linux, an NVIDIA GPU, SAM3D base weights and its CUDA
extensions. Start from the official upstream environment at the revision below.
The small MOD checkpoint does not replace the SAM3D base model.

## Source and dependencies

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git external/sam-3d-objects
cd external/sam-3d-objects
git checkout afdf6a31522d038c44c68a0bb57aa68827380797
mamba env create -f environments/default.yml
mamba activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
python -m pip install -e '.[dev]'
python -m pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
python -m pip install -e '.[inference]'
./patching/hydra
```

Follow that revision's official checkpoint-download instructions and SAM License.
The base checkpoint is obtained separately; it is not copied into this repository.
The reference runtime uses Python 3.11, PyTorch 2.5.1 and CUDA 12.1.

## Configure MOD

Return to the MOD repository, keeping the SAM3D environment active:

```bash
export SAM3D_ROOT=/path/to/MessyKitchens/external/sam-3d-objects
export SAM3D_PIPELINE_CONFIG="${SAM3D_ROOT}/checkpoints/hf/pipeline.yaml"
export MOD_PYTHON="$(command -v python)"
export SAM3D_PYTHON="${MOD_PYTHON}"
python -m pip install -e '.[train,iou]'
python scripts/setup_sam3d.py --sam3d-root "${SAM3D_ROOT}"
```

Set `SAM3D_PIPELINE_CONFIG` to the actual `pipeline.yaml` in the downloaded
checkpoint bundle. Some downloads add a `checkpoints/` subdirectory. The config's
relative model paths must resolve within that bundle.

`setup_sam3d.py` checks the upstream revision and the exact integration files.
It does not download weights or prove GPU execution. The real run receipts
and tested hardware are recorded separately in [VERIFICATION.md](release/VERIFICATION.md).

## Included runtime integration

`sam3d_capture.py` observes the existing sparse-structure Euler solver,
the transformer's split output before latent projection, image conditioning,
and pointmap normalization. It records actual tensors from the last original
solver step, preserves original return values, and restores scoped instance
methods even on errors. It does not sample random replacement tensors, re-run
the backbone, override solver states, or apply local research optimizations.
Only one object may use a pipeline instance at a time during preparation.

The legacy per-object files written during preparation use trusted local NumPy
object arrays. Published demo caches use the separate pickle-free scene format.

For differentiable MOD training, the external pose-target module's dataclass
conversion is replaced at runtime by a tensor-preserving equivalent. Upstream
`dataclasses.asdict` deep-copies tensors, which fails on non-leaf tensors;
preserving tensor objects retains the same pose calculation and its gradients.
The external source files are unchanged. The standard upstream initializer is
skipped through its documented `LIDRA_SKIP_INIT` environment switch.

The observer and compatibility helper are original MOD integration code under
the repository's Apache-2.0 license. They do not relicense SAM3D source or weights.
The exact paper's historical local checkout included additional changes and
is a separate provenance question; this integration does not claim their parity.

Full paper equivalence is tracked in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
