#!/usr/bin/env python3
"""Export model-only demo weights, retaining safe reproducibility metadata."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from multi_object_decoder.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", required=True,
                        choices=("synthetic_cpu_fixture", "synthetic_gso_small_sam3d"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; use --overwrite to replace it")
    checkpoint = load_checkpoint(args.input)
    expected = "direct" if args.scope == "synthetic_cpu_fixture" else "sam3d"
    if checkpoint["version"] < 1 or checkpoint["backend"] != expected:
        parser.error("a versioned checkpoint matching the requested backend is required")
    state = checkpoint["training_state"]
    metadata = {"epoch": int(state.get("epoch", 0)),
                "global_step": int(state.get("global_step", 0)),
                "demo_scope": args.scope, "paper_checkpoint": False,
                "source_checkpoint_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    save_checkpoint(args.output, backend=checkpoint["backend"],
                    model_config=checkpoint["model_config"], state_dict=checkpoint["state_dict"],
                    training_state=metadata)
    print(f"Exported {args.output.stat().st_size} bytes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
