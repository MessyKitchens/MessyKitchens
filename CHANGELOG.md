# Changelog

All notable changes to this repository are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-09-28

### Fixed

- `hydra-core` is now part of the `test` extra. The fake-SAM3D preparation
  test instantiates its pipeline through Hydra, so the first public CI run
  failed in a clean CPU environment that did not have it installed.
- Inference rejects non-finite predicted poses instead of writing invalid pose
  JSON; the scene is reported as `failed` in `inference_manifest.json`.
  `poses_to_json` and `save_pose_json` now raise `ValueError` on NaN or
  infinity.
- Removed a misleading NaN early return from the pose refiner. Non-finite
  activations still propagate and are caught by the existing training-loss and
  inference checks; finite outputs are unchanged.
- `scripts/compute_mean_metrics.py` writes its default Excel report next to
  the manifest instead of into `scripts/`.

### Changed

- Rewrote `README.md` around a CPU quick start, a method overview, a repository
  map, a configuration table, and a documentation index. Internal verification
  narrative now lives in `docs/release/`.
- Translated the remaining non-English source comments, removed a debugging
  `__main__` block from the pose refiner, and cleaned up legacy script
  docstrings.
- Fixed a duplicated paragraph in `docs/SAM3D.md` and a mangled command in
  `docs/WORKFLOWS.md`.
- Added issue and pull request templates, this changelog, and `testpaths` for
  pytest.

## [0.2.0] - 2026-09-22

### Added

- First public release: the Multi-Object Decoder network, SAM3D feature
  preparation, optional training-target registration, training, inference,
  mesh export, aligned-geometry metrics, bundled demos with small weights,
  documentation, and the CPU test suite.
