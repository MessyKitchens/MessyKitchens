# Open-source release progress

_Last updated: September 15, 2026._

This page tracks the staged release of MessyKitchens and Multi-Object Decoder
(MOD). A method described in the paper is not marked as released until its
code, configuration, required metadata, license, and a public download location
are available.

Status key: ✅ public · 🧪 included in the repository release candidate ·
📄 described in the paper, release package pending · 🚧 in preparation

## Release snapshot

| Track | Status | Available now | Remaining release work |
|---|---|---|---|
| Paper and project page | ✅ Public | [Paper](https://arxiv.org/abs/2603.16868), [project page](https://messykitchens.github.io/), and qualitative videos | Keep release links synchronized with this tracker. |
| MessyKitchens-Real data | ✅ Public | [Hugging Face snapshot](https://huggingface.co/datasets/MessyKitchens/MessyKitchens) with RGB views, object models, registered scene geometry, and object poses | Instance masks, MOD-ready manifests, and the exact evaluation protocol remain pending. |
| MOD network and runnable workflow | 🧪 Release candidate | Network source, data preparation, training, inference, mesh export, aligned-geometry metrics, and small workflow checkpoints | Freeze and verify the exact paper architecture/configuration and publish the paper checkpoint with a model card, license, revision, and checksum. |
| Real-data acquisition | 📄 Method documented | The paper describes the scanner, capture rig, object/scene scanning, and RGB acquisition procedure | Publish a standalone protocol, rig details, raw-data provenance, and any redistributable acquisition assets. |
| Object-to-scene registration | 📄 Method documented | The paper describes manual initialization followed by distance- and normal-aware refinement | Publish the registration source, environment, configs, example inputs/outputs, initialization format, and verification tests. |
| MessyKitchens-Synthetic generation | 📄 Method documented | The paper describes GSO preprocessing, controlled Blender physics, Cycles rendering, and instance-map export; this repository includes one processed GSO demo | Publish the Blender source, pinned environment, asset manifest, exact generation/render configs, seed policy, quality-control rules, split manifests, and full dataset. |
| Paper-exact evaluation | 🚧 In preparation | Aligned-mesh IoU, Chamfer distance, and F-score utilities | Publish the full evaluator, physical/contact metrics, expected results, pinned dependencies, and reproduction commands. |

The current package is a **code-and-demo release candidate**, not yet a
paper-exact reproduction package. Artifact-level audit results are recorded in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md), and completed validation runs are in
[release/VERIFICATION.md](release/VERIFICATION.md).

## 1. MOD network and model release

At a high level, the paper builds MOD on top of SAM 3D Objects:

1. An RGB image and an instance-label mask are processed object by object by
   SAM3D to obtain canonical geometry, a base pose, pose tokens, and shape
   tokens.
2. MOD processes all objects belonging to one scene together. Its attention
   blocks combine within-object pose reasoning, cross-object pose reasoning,
   pose-to-shape conditioning, and an MLP.
3. The decoder predicts residual object transforms. The canonical geometry is
   kept fixed while object placement is refined for better scene consistency.
4. Inference exports object-keyed poses, individual posed meshes, and an
   optional composed scene mesh.

The release candidate contains the MOD network and the preparation, training,
and inference entrypoints. It also contains a small SAM3D-backed checkpoint and
a portable CPU checkpoint for workflow validation. Neither checkpoint is the
model used to report the paper results.

Preparation also supports `--generate-training-gt`: given posed GT geometry,
explicit mask-object-to-scene-node correspondence, and an external GAPs
`mshalign` executable, it registers fresh SAM3D geometry and writes training
targets in SAM3D prediction space. See [the training-GT guide](DATA.md#generate-training-gt).
This does not release the original real-data acquisition/scan-registration
pipeline or establish paper-exact target quality.

Before the model track is marked paper-exact, the final release must include:

- the verified camera-ready network/configuration and pinned SAM3D integration;
- the official MOD checkpoint, model card, license, revision, and SHA256;
- the exact training split and target-pose provenance;
- the paper training objective and schedule, verified against the released
  implementation;
- reference inference outputs and expected evaluation values.

See [WORKFLOWS.md](WORKFLOWS.md) for the currently runnable commands and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the known boundary between the
release candidate and the paper artifacts.

## 2. MessyKitchens-Real acquisition and registration

The paper's real-data pipeline can be summarized as follows:

1. **Object scanning.** Each kitchen object is scanned from the top and bottom
   on a transparent acrylic plate mounted on a manual turntable. Back-to-back
   reflective markers provide shared correspondences through the plate, while
   marker-equipped blocks add non-coplanar constraints.
2. **Object-model assembly.** The complementary scans are aligned and merged in
   the scanner software to form a complete object model.
3. **Scene scanning.** Contact-rich easy, medium, and hard arrangements are
   scanned at high resolution. The paper reports decimating the raw scene mesh
   by 80% while retaining a point-to-mesh error below 0.05 mm.
4. **RGB acquisition.** The target protocol captures 15 handheld-camera images
   per scene using three elevations and five azimuths under varied backgrounds
   and lighting.
5. **Object-to-scene registration.** Starting from a manual coarse alignment,
   each object is refined first with a robust surface-distance objective and
   then with a normal-aware objective to resolve ambiguities for thin or
   concave objects.

For registration, the paper samples 500 object-surface points, uses closest
points on the scene mesh, and optimizes a soft-L1 objective for 20 iterations.
The second 20-iteration stage starts from that solution and uses surface-normal
agreement, retaining a normal weight only when the dot product is at least 0.7.

The public dataset contains the final RGB images, object models, scene meshes,
and registered poses. It does **not** currently constitute a source release of
the acquisition and registration pipeline: raw scanner projects, rig assets,
camera metadata, manual initializations, registration source code, and instance
masks are not present in the inspected public package. The repository's
`scripts/prepare.sh` creates SAM3D caches from RGB and supplied masks, and can
optionally register those predictions against already-posed GT geometry for
training. It is not the original object-scan-to-scene-scan registration program.

The registration track will be marked released only after a clean example can
be reproduced from documented input meshes and a coarse initialization through
to registered object transforms and a verification report.

## 3. MessyKitchens-Synthetic Blender generation

The paper describes the following Blender pipeline:

1. Select 42 kitchenware assets from Google Scanned Objects (GSO), remesh them
   for stable simulation, and correct centers of mass where needed.
2. Construct easy, medium, and hard scenes with controlled object placement,
   stacking, and nesting. Support compatibility uses object volume and upright
   top-surface area, with an explicit list of objects that should not be
   stacked.
3. Simulate with concave mesh collision. The reported settings include a
   0.01 mm object-object margin, a 1.0 mm object-plane margin, zero restitution,
   disabled velocity-based deactivation, and a fully constrained active plane.
4. Render settled scenes with Blender Cycles using the original object textures,
   randomized azimuth and elevation, and export RGB images together with
   instance maps and scene ground truth.

The paper reports 600 scenes for each difficulty level, or 1,800 scenes total.
Its rendering section states ten views per generated scene, while the training
setup states six images per scene, or 10,800 training images. The public release
will explicitly distinguish the generated views from the views selected for
training in versioned manifests.

The matching Blender generator is **not included** in the current release
candidate.

The planned Blender release contains:

- a pinned Blender/Python environment and command-line entrypoint;
- GSO asset IDs, preprocessing code, attribution, and download instructions;
- scene templates and exact easy/medium/hard construction rules;
- rigid-body, collision, gravity, settling, and quality-control parameters;
- camera, lighting, background, resolution, and Cycles render settings;
- deterministic seeds and per-scene metadata, object transforms, RGB images,
  instance maps, and integrity checks;
- versioned train/validation manifests and checksums for the released assets.

Until these items are available and a clean generation run has been recorded,
the Blender pipeline and full MessyKitchens-Synthetic dataset remain marked
**in preparation**.
