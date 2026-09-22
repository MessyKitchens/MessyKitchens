# Clean release procedure

Run the static payload check on a fresh checkout before installing packages or
creating outputs inside it. Development environments and experiment outputs
are not release artifacts.

## Source and tests

```bash
python scripts/dev/check_release.py
python -m pip install -e '.[train,iou,test]'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
python -m ruff check --no-cache .
```

Installation can create `src/*.egg-info`, and local environments/output folders
may exist in a developer checkout. Do not include those in a source release.
After reviewing and committing the intended code and demo files, validate a
separate clean checkout with `python scripts/dev/check_release.py --require-git`.
It checks both tracked files and the clean index/worktree.

## Export a release payload

```bash
RELEASE_DIR=$(mktemp -d)
git archive HEAD | tar -x -C "${RELEASE_DIR}"
python "${RELEASE_DIR}/scripts/dev/check_release.py" "${RELEASE_DIR}"
```

The exported tree has no `.git` and contains only committed files. The default
check validates it without Git. All binary demo inputs must match
`assets/demo/MANIFEST.json`; a missing/unlisted/modified asset fails the check.
Record the originating `git rev-parse HEAD` in the publication receipt.

## Build and inspect distributions

In a packaging environment with `build` and `twine`:

```bash
DIST_DIR=$(mktemp -d)
python -m build --outdir "${DIST_DIR}" "${RELEASE_DIR}"
python -m twine check "${DIST_DIR}"/*
```

Extract the source distribution into another temporary directory and confirm it
contains the consolidated `scripts/demo.sh` entrypoint, `scripts/`, `configs/`, `assets/demo/`,
integration notes, tests, and license texts. The wheel provides the Python
package and console commands; the repository/source distribution provides the
standalone demo assets and shell entrypoints. Run CPU demos from the extracted
source using an external environment so no development files are packaged.

## GPU and paper gates

Use the pinned external SAM3D revision and separately obtained weights. Run
training, inference, and processing against the committed demo inputs with
fresh output directories; preserve scene success/error manifests and the
source/asset checksums. Full paper claims additionally require:

```bash
python scripts/dev/check_release.py --paper-exact
```

That stricter gate is expected to fail for this code-and-demo release because
the full paper checkpoint, split/GT protocol and evaluation stack are not all
published. Do not relabel the demo verification as paper-result reproduction.

Publishing is a separate step: confirm the final repository URL, push the
reviewed commit, set the intended visibility, and link that same repository
from the project website. Update the website only after public access is verified.
