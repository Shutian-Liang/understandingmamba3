"""Run the standard zero-shot LM Evaluation Harness tasks on a checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from evaluate.common import atomic_json


TASKS = (
    "lambada_openai",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "openbookqa",
)

TASK_METRICS = {
    "lambada_openai": (("lambada_ppl", "perplexity"), ("lambada_acc", "acc")),
    "hellaswag": (("hellaswag_acc_norm", "acc_norm"),),
    "piqa": (("piqa_acc", "acc"),),
    "arc_easy": (("arc_easy_acc", "acc"),),
    "arc_challenge": (("arc_challenge_acc_norm", "acc_norm"),),
    "winogrande": (("winogrande_acc", "acc"),),
    "openbookqa": (("openbookqa_acc", "acc"),),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(results: dict, task: str, name: str) -> float:
    task_metrics = results["results"][task]
    for key in (f"{name},none", name):
        if key in task_metrics:
            return float(task_metrics[key])
    raise KeyError(f"Missing {task}/{name}: {sorted(task_metrics)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--results-root", type=Path, default=ROOT / "evaluate/results")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bucket-lengths", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Allow a checkpoint saved before target_tokens, including an earlier best.pt")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    run_name = args.run_name or checkpoint.parent.name
    output = args.results_root / "harness" / run_name / "results.json"
    if output.exists() and not args.force and args.limit is None:
        previous = json.loads(output.read_text())
        saved_checkpoint = previous.get("checkpoint", {})
        if (not args.allow_incomplete and
                saved_checkpoint.get("tokens_seen", 0) < saved_checkpoint.get("target_tokens", 0)):
            parser.error("Checkpoint predates the full token budget; use --allow-incomplete")
        if (previous.get("checkpoint", {}).get("sha256") != file_sha256(checkpoint)
                or set(previous.get("harness_results", {})) != set(args.tasks)
                or previous.get("batch_size") != args.batch_size
                or previous.get("bucket_lengths", False) != args.bucket_lengths):
            parser.error(f"Different checkpoint or evaluation options at {output}; use another --run-name or --force")
        print(f"Already complete: {output}")
        return

    from lm_eval import evaluator
    from evaluate._harness_model import Mamba3HarnessLM

    started = time.time()
    lm = Mamba3HarnessLM(
        checkpoint,
        batch_size=args.batch_size,
        max_length=2048,
        bucket_lengths=args.bucket_lengths,
        allow_incomplete=args.allow_incomplete,
    )
    harness = evaluator.simple_evaluate(
        model=lm,
        tasks=list(args.tasks),
        num_fewshot=0,
        batch_size=args.batch_size,
        device="cuda",
        limit=args.limit,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
        log_samples=False,
    )
    scores = {}
    for task in args.tasks:
        for output_name, metric_name in TASK_METRICS[task]:
            scores[output_name] = metric(harness, task, metric_name)
    accuracy_keys = [key for key in scores if key.endswith("_acc") or key.endswith("_acc_norm")]
    if accuracy_keys:
        scores["average_acc"] = sum(scores[key] for key in accuracy_keys) / len(accuracy_keys)
    payload = {
        "protocol": "mamba3_lm_harness_zero_shot",
        "lm_eval_version": "0.4.3",
        "tasks": list(args.tasks),
        "run": run_name,
        "checkpoint": lm.checkpoint_identity,
        "tokenizer": {
            "name": "meta-llama/Meta-Llama-3.1-8B",
            "path": str(ROOT / "evaluate/assets/tokenizer/tokenizer.json"),
            "sha256": file_sha256(ROOT / "evaluate/assets/tokenizer/tokenizer.json"),
        },
        "zero_shot": True,
        "max_length": 2048,
        "batch_size": args.batch_size,
        "bucket_lengths": args.bucket_lengths,
        "limit": args.limit,
        "scores": scores,
        "harness_results": harness["results"],
        "harness_versions": harness.get("versions", {}),
        "elapsed_seconds": time.time() - started,
    }
    if args.limit is None:
        atomic_json(output, payload)
    else:
        smoke_output = output.with_name(f"smoke_limit_{args.limit}.json")
        atomic_json(smoke_output, payload)
        output = smoke_output
    print(json.dumps({"output": str(output), "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
