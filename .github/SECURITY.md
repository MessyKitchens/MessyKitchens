# Security policy

## Supported version

Security fixes are applied to the latest release and the current `main`
branch. Research snapshots and unpublished experiment branches are not
supported.

## Reporting a vulnerability

Please use the repository's **Security → Report a vulnerability** workflow so
the report is not public before a fix is available. Include affected versions,
reproduction steps, impact, and any proposed mitigation. Do not include real
credentials or private datasets.

## Checkpoint safety

PyTorch checkpoint files can contain pickle payloads. MessyKitchens loads
checkpoints with `torch.load(..., weights_only=True)` and refuses the unsafe
fallback used by older PyTorch versions. Still download weights only from a
documented source and verify their published SHA256 before use.

## Legacy cache safety

SAM3D's legacy `tokens/obj_*.npz` files contain pickle-backed object arrays.
Training, inference, and reuse during data preparation refuse those files by
default. Only for a cache you generated yourself or otherwise verified, pass
`--allow-legacy-pickle` to `scripts/train.sh`, `scripts/infer.sh`, or
`scripts/prepare.sh`. The portable scene NPZ format and all bundled demos are
pickle-free and do not require this opt-in.
