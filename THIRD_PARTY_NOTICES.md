# Third-Party Notices

This file records third-party code incorporated into this repository and
external software required by optional reproduction workflows. It is provided
for attribution and scope clarification; it does not replace the license text
distributed by any third party.

## License scope

The original MessyKitchens source code in this repository is licensed under
Apache License 2.0; see [`LICENSE`](LICENSE).

That license does **not** apply to separately downloaded datasets, model
weights, or third-party repositories. In particular:

- The MessyKitchens dataset is distributed separately under CC BY-NC 4.0 at
  <https://huggingface.co/datasets/MessyKitchens/MessyKitchens>.
- SAM 3D Objects code and checkpoints are distributed separately under Meta's
  custom SAM License, described below.
- Other model artifacts remain subject to the terms stated on their respective
  download pages.

## Code incorporated or adapted in this repository

### PartCrafter

- Project: <https://github.com/wgsxm/PartCrafter>
- Source-lineage commit:
  `347e9b6d4093bae0e886772dd9d41e555848cb94`
- Audited upstream snapshot:
  `3d773bf02fad51c7ab31a5615573fec93b287b30`
- License: MIT
- Copyright: Copyright (c) 2025 Yuchen Lin

The following MessyKitchens implementation areas contain code adapted from
PartCrafter:

- `src/multi_object_decoder/geometry_metrics.py`: voxel IoU, Chamfer distance,
  and F-score routines, adapted from `src/utils/metric_utils.py`.
The relevant PartCrafter files were introduced at the source-lineage commit
above and remain present in the audited snapshot. The MIT notice applying to
those portions follows.

```text
MIT License

Copyright (c) 2025 Yuchen Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Optional external software not included in this repository

The items in this section are not relicensed by MessyKitchens. Users obtain
them separately and are responsible for complying with their original terms.

### SAM 3D Objects

- Project: <https://github.com/facebookresearch/sam-3d-objects>
- Compatibility baseline:
  `afdf6a31522d038c44c68a0bb57aa68827380797`
- License: custom SAM License, last updated November 19, 2025
- License text at the baseline:
  <https://github.com/facebookresearch/sam-3d-objects/blob/afdf6a31522d038c44c68a0bb57aa68827380797/LICENSE>

The full SAM 3D Objects source code and base checkpoints are not included in this
repository. The small SAM3D-backed MOD demo checkpoint in
`assets/demo/real/model.pt` includes SAM-derived components and is distributed
under the complete SAM License copied to `assets/demo/LICENSE-SAM`. They are **not** covered by the MessyKitchens Apache-2.0 license.
Use of either requires separate acceptance of and compliance with the complete
SAM License. Among other provisions, that license governs redistribution of
SAM Materials and derivative works, requires distribution under its own terms
with a copy of the agreement, requires research acknowledgement, and includes
trade-control and use restrictions. Refer to the complete upstream agreement,
not this summary, for the controlling terms.

The small-data demos use the pinned upstream baseline with the repository
owned runtime observer documented in `docs/SAM3D.md`. Their compatibility
is checked separately from equivalence with the historical paper working tree. If
such a fork or patch is distributed, it must carry the SAM License separately;
the repository-level Apache-2.0 license must not be presented as covering the
SAM Materials or their derivatives.

### PyTorch3D

- Project: <https://github.com/facebookresearch/pytorch3d>
- Commit pinned by the SAM 3D Objects baseline:
  `75ebeeaea0908c5527e7b1e305fbc7681382db47`
- License: BSD 3-Clause
- Copyright: Copyright (c) Meta Platforms, Inc. and affiliates
- License text:
  <https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/LICENSE>

PyTorch3D is installed as part of the external SAM 3D Objects environment; its
source and binaries are not bundled here.

### GAPS

- Project: <https://github.com/tomfunkhouser/gaps>
- License: MIT
- Copyright: Copyright (c) 2007 Thomas Funkhouser
- License text: <https://github.com/tomfunkhouser/gaps/blob/master/LICENSE.txt>

The optional `prepare --generate-training-gt` stage and the full paper
evaluator use the native `mshalign` executable for Sim(3) alignment. The
training-target stage uses the release-owned, checked subprocess wrapper in
`src/multi_object_decoder/training_gt.py`, not the private evaluation helpers.
Neither GAPS source nor a GAPS binary is bundled here. The README pins an
inspected source revision for the optional build instructions; it does not
identify the paper's historical binary. A release claiming exact metric
reproducibility should record the tested GAPS commit or binary checksum
alongside the evaluator configuration.

### External PartCrafter evaluation utilities

The full paper evaluator also imports a locally modified PartCrafter-derived
source tree (`partcrafter_ran`) that is not included in this repository. The
standalone aligned-mesh metrics implementation does not require that tree.
Cloning upstream PartCrafter is not, by itself, a substitute for the modified
paper-evaluation utilities. Any future distribution of those utilities must
preserve the PartCrafter MIT notice above and identify project-authored
modifications.

## Package-manager dependencies

The Python packages declared in `pyproject.toml` are installed separately by a
package manager and retain their own licenses and bundled third-party notices.
This repository does not vendor those distributions. Because most dependency
requirements are version ranges rather than exact pins, the complete resolved
license inventory can vary with installation time and platform. For a binary
release or archived reproduction environment, preserve the exact lock file and
the license metadata shipped by every resolved distribution.

## Bundled demonstration assets

The original GSO-derived render, labels, registered targets, canonical geometry
and numerical caches are CC BY 4.0; see `assets/demo/LICENSE-DATA` and the detailed
source/transform record in `assets/demo/real/PROVENANCE.json`. The actual
SAM3D-backed small MOD model in `assets/demo/real/model.pt` is distributed under
the SAM License, with its complete agreement in `assets/demo/LICENSE-SAM`.
It includes learned MOD adapters and copied SAM components; it is not the
full SAM3D base model or the paper checkpoint. Synthetic CPU fixtures and their
small independently initialized direct-backend model are Apache-2.0.
