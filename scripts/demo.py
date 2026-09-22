#!/usr/bin/env python3
"""Run an independent preparation, training, or inference demo."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("prepare", "train", "infer"))
    result.add_argument("--demo-dir", type=Path, default=ROOT / "outputs" / "demo")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--data-json", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--device", default="cuda")
    result.add_argument("--max-train-steps", type=int, default=2)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--sam3d-root", type=Path, default=os.environ.get("SAM3D_ROOT"))
    result.add_argument("--sam3d-config", type=Path,
                        default=os.environ.get("SAM3D_PIPELINE_CONFIG"))
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--allow-legacy-pickle", action="store_true",
                        help="Trust SAM3D caches generated locally with pickle-backed arrays.")
    result.add_argument("--generate-training-gt", action="store_true",
                        help="During preparation, register fresh SAM3D geometry to supplied GT.")
    result.add_argument("--gt-mesh", type=Path,
                        help="GT scene GLB for a single-record preparation input (not bundled).")
    result.add_argument("--gt-object-map", type=Path,
                        help="JSON mapping SAM3D object IDs to GT scene node names.")
    result.add_argument("--gaps-binary", type=Path,
                        help="External GAPs mshalign executable; alternatively set GAPS_MSHALIGN.")
    result.add_argument("--gt-timeout", type=float,
                        help="Seconds per GT alignment call (default: 120).")
    result.add_argument("--cpu-smoke", action="store_true",
                        help="Use the bundled synthetic direct-backend fixture; no SAM3D.")
    result.add_argument("--dry-run", action="store_true",
                        help="Print the resolved command and inputs without loading models.")
    return result


def _training_gt_input(
    args: argparse.Namespace, source: Path, output: Path,
) -> tuple[list[dict], list[Path]]:
    """Rebase raw inputs without ever modifying the bundled/source manifest."""
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
        raise ValueError("--data-json must contain a non-empty list of scene records")
    override = args.gt_mesh is not None or args.gt_object_map is not None
    if override and (args.gt_mesh is None or args.gt_object_map is None):
        raise ValueError("provide --gt-mesh and --gt-object-map together")
    if override and len(records) != 1:
        raise ValueError("--gt-mesh/--gt-object-map require a single-record input; "
                         "for multiple scenes set mesh_path and gt_object_map in each record")
    required = []
    mapping = None
    if override:
        map_path = args.gt_object_map.expanduser().resolve()
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        required.append(map_path)
    derived = []
    for row in records:
        record = dict(row)
        for field in ("target_poses_path", "gt_poses_path"):
            value = record.get(field)
            if value is None or not str(value).strip():
                continue
            target = Path(str(value)).expanduser()
            target = (source.parent / target).resolve() if not target.is_absolute() else target.resolve()
            # Stripping these references must not hide protected source targets
            # from preparation's overwrite checks or its generated-target writer.
            if target.is_relative_to(output):
                raise ValueError(f"source {field} is inside the GT output directory; "
                                 "choose a new --output-dir to preserve existing targets")
        if override:
            record["mesh_path"] = str(args.gt_mesh.expanduser().resolve())
            record["gt_object_map"] = mapping
        object_map = record.get("gt_object_map")
        if (not isinstance(object_map, dict) or not object_map
                or any(not isinstance(key, str) or not key.isascii() or not key.isdigit()
                       or str(int(key)) != key or not isinstance(value, str) or not value.strip()
                       for key, value in object_map.items())
                or len(set(object_map.values())) != len(object_map)):
            raise ValueError("gt_object_map must map distinct numeric SAM3D IDs (e.g. '0') "
                             "one-to-one to non-empty GT scene node names")
        for field in ("image_path", "instance_mask_path", "mesh_path", "scene_json_path"):
            value = record.get(field)
            if value is None and field == "scene_json_path":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"GT preparation needs {field}; supply your raw GT scene "
                                 "with --gt-mesh/--gt-object-map or a custom --data-json")
            path = Path(value).expanduser()
            path = (source.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise ValueError(f"required GT preparation input is missing: {path}")
            record[field] = str(path)
            required.append(path)
        # Historical targets only match their historical canonical meshes.
        record.pop("target_poses_path", None)
        record.pop("gt_poses_path", None)
        derived.append(record)
    return derived, required


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    if args.max_train_steps < 1 or args.seed < 0:
        cli.error("training steps must be positive and seed must be non-negative")
    if args.cpu_smoke and args.stage == "prepare" and args.data_json is not None:
        cli.error("CPU preparation uses the fixed bundled fixture; --data-json is not supported")
    if args.checkpoint is not None and args.stage != "infer":
        cli.error("--checkpoint is an inference option")
    gt_options = (args.gt_mesh, args.gt_object_map, args.gaps_binary, args.gt_timeout)
    if (args.generate_training_gt or any(value is not None for value in gt_options)) and (
            args.stage != "prepare" or args.cpu_smoke):
        cli.error("training-GT options require the SAM3D prepare stage (without --cpu-smoke)")
    if any(value is not None for value in gt_options) and not args.generate_training_gt:
        cli.error("GT options require --generate-training-gt")
    if args.gt_timeout is not None and (not math.isfinite(args.gt_timeout) or args.gt_timeout <= 0):
        cli.error("--gt-timeout must be a positive finite number")
    if args.allow_legacy_pickle and args.cpu_smoke and args.stage == "prepare":
        cli.error("CPU preparation has no legacy SAM3D cache to trust")
    bundle = ROOT / "assets/demo" / ("portable" if args.cpu_smoke else "real")
    names = {"prepare": "data", "train": "training", "infer": "inference"}
    output = (args.output_dir or args.demo_dir / names[args.stage]).expanduser().resolve()
    if output == ROOT or output in ROOT.parents or output.is_relative_to(ROOT / "assets/demo"):
        cli.error("demo output must be outside source and bundled demo inputs")
    data = (args.data_json or bundle / ("raw/input.json" if args.stage == "prepare"
                                       else "prepared/prepared_data.json")).expanduser().resolve()
    checkpoint = (args.checkpoint or bundle / "model.pt").expanduser().resolve()
    command = [sys.executable]
    required: list[Path] = []
    generated_input = None
    if args.generate_training_gt:
        if data in {(output / name).resolve() for name in (
                "prepared_data.json", "prepare_manifest.json", "demo_run.json")}:
            cli.error("the source manifest must not be a generated output manifest; "
                      "choose a separate input manifest or a new --output-dir")
        if args.overwrite and data.is_relative_to(output):
            cli.error("source manifest is inside --output-dir; choose a new output "
                      "directory to preserve it during --overwrite")
        try:
            generated_records, gt_inputs = _training_gt_input(args, data, output)
        except (OSError, ValueError) as error:
            cli.error(str(error))
        required.extend([data, *gt_inputs])
        data = output / "_inputs" / "training_gt_input.json"
        if data.resolve() in required or not data.resolve().is_relative_to(output):
            cli.error("generated GT manifest must remain inside the output and not replace a source input")
        generated_input = {"path": str(data), "records": generated_records}
    if args.cpu_smoke and args.stage == "prepare":
        command += [str(ROOT / "scripts" / "prepare_portable_demo.py"),
                    "--source", str(bundle / "prepared"), "--output-dir", str(output)]
        required.append(bundle / "prepared" / "prepared_data.json")
        if args.overwrite:
            command.append("--overwrite")
    else:
        entry = {"prepare": "prepare.py", "train": "train.py", "infer": "infer.py"}
        command.append(str(ROOT / "scripts" / entry[args.stage]))
        if generated_input is None:
            required.append(data)
        if args.stage == "train":
            config = ROOT / "configs" / ("demo_direct.yaml" if args.cpu_smoke
                                        else "demo_sam3d.yaml")
            command += ["--config", str(config), "--output-dir", str(output),
                        "--device", "cpu" if args.cpu_smoke else args.device,
                        "--precision", "fp32", "--seed", str(args.seed),
                        "--max-train-steps", str(args.max_train_steps)]
        else:
            command += ["--data-json", str(data), "--output-dir", str(output)]
            if args.stage == "prepare":
                command += ["--seed", str(args.seed)]
                if args.generate_training_gt:
                    command += ["--generate-training-gt", "--gt-timeout",
                                str(args.gt_timeout if args.gt_timeout is not None else 120.0)]
                    if args.gaps_binary is not None:
                        command += ["--gaps-binary", str(args.gaps_binary.expanduser().resolve())]
            if args.stage == "infer":
                required.append(checkpoint)
                command += ["--checkpoint", str(checkpoint), "--device",
                            "cpu" if args.cpu_smoke else args.device, "--export-meshes"]
            if args.overwrite:
                command.append("--overwrite")
        if not args.cpu_smoke:
            if not args.sam3d_root or not args.sam3d_config:
                cli.error("set SAM3D_ROOT and SAM3D_PIPELINE_CONFIG (see docs/SAM3D.md), "
                          "or select --cpu-smoke for the small offline interface test")
            command += ["--sam3d-root", str(args.sam3d_root.expanduser().resolve()),
                        "--sam3d-config", str(args.sam3d_config.expanduser().resolve())]
        if args.allow_legacy_pickle:
            command.append("--allow-legacy-pickle")
        if args.stage == "train":
            command.append("train.data_json=" + str(data))
    for path in required:
        if not path.is_file():
            cli.error(f"required demo input is missing: {path}")
    if args.stage == "train" and output.exists() and any(output.iterdir()):
        cli.error("training output is not empty; choose another --output-dir to keep runs separate")
    receipt = {"stage": args.stage, "backend": "direct" if args.cpu_smoke else "sam3d",
               "inputs": [str(path) for path in required], "output_dir": str(output),
               "command": command}
    if generated_input is not None:
        receipt["generated_input"] = generated_input
    print(json.dumps(receipt, indent=2), flush=True)
    if args.dry_run:
        return 0
    if generated_input is not None:
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(json.dumps(generated_records, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["LIDRA_SKIP_INIT"] = "true"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(key, "1")
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if result.returncode == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "demo_run.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Demo {args.stage} completed: {output}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
