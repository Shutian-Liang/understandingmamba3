"""Evaluate a local training checkpoint on the fixed PG-19 length probe."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate.common import (ROOT, atomic_json, exclusive_lock, load_model,
                             read_json, sha256, signature, software_identity)
from pretrain.pg19 import evaluate_pg19_fixed_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--probe", type=Path,
        default=ROOT / "data/pg19_test/mamba3_llama31_fixed_probe",
    )
    parser.add_argument("--results-root", type=Path, default=ROOT / "evaluate/results")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.run_name:
        args.run_name = args.checkpoint.expanduser().resolve().parent.name

    model, checkpoint = load_model(args.checkpoint, args.allow_incomplete)
    manifest_path = args.probe / "manifest.json" if args.probe.is_dir() else args.probe
    manifest = read_json(manifest_path)
    spec = {
        "version": 1,
        "protocol": "pg19_fixed_length_extrapolation",
        "checkpoint": checkpoint,
        "probe_manifest": str(manifest_path.resolve()),
        "probe_manifest_sha256": sha256(manifest_path),
        "probe_protocol": manifest["protocol"],
        "batch_size": args.batch_size,
        "precision": "fp32_weights_bf16_autocast",
        "software": software_identity(),
    }
    output = args.results_root / args.run_name / "pg19_fixed" / "result.json"
    expected_signature = signature(spec)
    with exclusive_lock(output.with_suffix(".lock")):
        if output.exists():
            existing = read_json(output)
            if existing.get("signature") != expected_signature:
                raise ValueError(f"Result path belongs to another protocol/checkpoint: {output}")
            print(f"DONE ALREADY PG19: {output}", flush=True)
            return
        result = {
            "signature": expected_signature,
            "spec": spec,
            "metrics": evaluate_pg19_fixed_probe(
                model, args.probe, device="cuda", batch_size=args.batch_size,
                autocast_dtype="bfloat16",
            ),
        }
        atomic_json(output, result)
    print(f"PG19 COMPLETE: {output}", flush=True)
    for key, value in result["metrics"].items():
        if "/perplexity_" in key:
            print(f"{key}={value:.6f}", flush=True)


if __name__ == "__main__":
    main()
