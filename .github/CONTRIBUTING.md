# Contributing

Thank you for improving MessyKitchens. Please keep changes focused, portable,
and reproducible.

## Before opening a pull request

1. Create a branch from the current `main` branch.
2. Add or update tests for behavior changes.
3. Run `python scripts/dev/check_release.py` and `python -m pytest -q`.
4. Run the relevant demo stage when changing data, training, or inference.
5. Update the README and release manifest when an interface or artifact
   contract changes.

Do not commit datasets, checkpoints, generated outputs, credentials, private
paths, or code whose redistribution terms are unclear. Keep optional research
dependencies behind lazy imports so the portable workflow remains usable.

By submitting a contribution, you agree that it may be distributed under the
Apache License 2.0 in this repository. If a change derives from third-party
code, preserve its attribution and update `THIRD_PARTY_NOTICES.md`.

Bug reports should include the exact command, the smallest usable input or
schema example, OS/Python/PyTorch/CUDA versions, and the complete error. Never
attach private data or access tokens.
