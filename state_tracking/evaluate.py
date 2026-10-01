from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent

import torch

from state_tracking.train import build_model, evaluate


def comma_separated_ints(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("expected positive comma-separated lengths")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lengths", type=comma_separated_ints, default=[192, 224])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional two-record JSONL with validation and test metrics.",
    )
    parser.add_argument(
        "--log-step",
        type=int,
        default=None,
        help="Curve step for the supplemental validation/test records.",
    )
    return parser.parse_args()


def resolved_file(path: Path) -> Path:
    result = path if path.is_absolute() else REPO_ROOT / path
    result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def resolved_output(path: Path) -> Path:
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def main() -> None:
    cli = parse_args()
    if not cli.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("State-tracking checkpoint evaluation requires CUDA")
    checkpoint_path = resolved_file(cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"Not a state-tracking checkpoint: {checkpoint_path}")

    config = dict(checkpoint["config"])
    config.setdefault("d_intermediate", 0)
    config.setdefault("tie_embeddings", False)
    config.setdefault("mimo_rank", 1)
    config.setdefault("chunk_size", 64)
    config.setdefault("precision", "fp32")
    config.setdefault("allow_tf32", False)
    config["device"] = cli.device
    config["eval_lengths"] = list(cli.lengths)
    if cli.micro_batch_size is not None:
        config["micro_batch_size"] = cli.micro_batch_size
    args = SimpleNamespace(**config)

    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    device = torch.device(cli.device)
    # Triton launches against the current CUDA device, not only tensor.device.
    torch.cuda.set_device(device)
    model = build_model(args, int(args.n_layers)).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"], strict=True)
    validation = evaluate(
        model,
        args,
        samples=cli.validation_samples,
        seed=int(args.seed) + 1_000_000_007,
    )
    test = evaluate(
        model,
        args,
        samples=cli.test_samples,
        seed=int(args.seed) + 2_000_000_011,
    )

    checkpoint_step = int(checkpoint.get("step", 0))
    log_step = checkpoint_step if cli.log_step is None else cli.log_step
    output_path = (
        resolved_output(cli.output)
        if cli.output is not None
        else checkpoint_path.parents[1]
        / (
            f"supplemental_eval_step{checkpoint_step}_lengths_"
            + "_".join(str(length) for length in cli.lengths)
            + ".json"
        )
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "curve_log_step": log_step,
        "task": str(args.task),
        "model": str(args.model),
        "lengths": list(cli.lengths),
        "validation_samples_per_length": cli.validation_samples,
        "test_samples_per_length": cli.test_samples,
        "validation": validation,
        "test": test,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if cli.metrics_output is not None:
        metrics_path = resolved_output(cli.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        records = (
            {"type": "validation", "step": log_step, "results": validation},
            {
                "type": "test",
                "step": log_step,
                "checkpoint": f"checkpoint_step_{checkpoint_step}",
                "results": test,
            },
        )
        metrics_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        payload["metrics_output"] = str(metrics_path)
    payload["output"] = str(output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
