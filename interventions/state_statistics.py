"""Reproduce the PG-19 recurrent-state statistics reported in Table 6."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import time

import torch

from evaluate.common import atomic_json, exclusive_lock, sha256, signature
from interventions.pg19 import (
    ROOT,
    intervention_software_identity,
    load_books,
    load_model,
)
from interventions.similarity.run import StateObserver, ZeroAngleProjection
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.utils.paper_kernel_policy import apply_paper_kernel_policy


CONDITIONS = ("native", "phase_removal")


def new_cache(model, length):
    params = InferenceParams(max_seqlen=length + 1, max_batch_size=1, seqlen_offset=1)
    params.key_value_memory_dict = model.allocate_inference_cache(1, length + 1)
    return params


@torch.inference_mode()
def trace(model, ids, model_name, condition, lengths, warmup, use_graph):
    count = ids.shape[0]
    params = new_cache(model, count)
    caches = [value for states in params.key_value_memory_dict.values() for value in states]
    observer = StateObserver(model, model_name == "mimo")
    phase = (
        ZeroAngleProjection(model)
        if condition == "phase_removal"
        else contextlib.nullcontext()
    )
    token = ids[:1].view(1, 1).clone()

    def clear():
        for value in caches:
            value.zero_()
        observer.reset()

    with phase, observer:
        replay = None
        if use_graph:
            # Warm up before capture, then discard all mutated cache/statistic state.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model.backbone(token, inference_params=params)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                model.backbone(token, inference_params=params)
            clear()
            replay = graph.replay

        started = time.monotonic()
        rows = []
        for position in range(count):
            if replay is None:
                model.backbone(ids[position:position + 1].view(1, 1), inference_params=params)
            else:
                token.copy_(ids[position:position + 1].view(1, 1))
                replay()
            if position + 1 == warmup:
                observer.reset()  # Keep recurrent state; exclude only accumulated measurements.
            if position + 1 in lengths:
                expected = position + 1 - warmup
                if not torch.equal(observer.counts, torch.full_like(observer.counts, expected)):
                    raise RuntimeError(f"observer count mismatch at {position + 1}")
                summary = observer.summary()
                cosine = float(torch.quantile(summary["cosine"][..., 0].flatten(), 0.5))
                rms = math.sqrt(float(torch.quantile(summary["state"][..., 0].flatten(), 0.5)))
                rows.append({
                    "length": position + 1,
                    "temporal_tokens": expected,
                    "cosine": cosine,
                    "state_rms": rms,
                })
            if (position + 1) % 2000 == 0:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                print(
                    f"condition={condition} tokens={position + 1}/{count} "
                    f"tokens_per_second={(position + 1) / elapsed:.1f}", flush=True,
                )
    return rows, time.monotonic() - started


def aggregate(rows, lengths, expected_books):
    result = []
    for length in lengths:
        selected = [row for row in rows if row["length"] == length]
        result.append({
            "length": length,
            "books": len(selected),
            "expected_books": expected_books,
            "complete": len(selected) == expected_books,
            "cosine": sum(row["cosine"] for row in selected) / len(selected) if selected else None,
            "state_rms": (
                sum(row["state_rms"] for row in selected) / len(selected) if selected else None
            ),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("siso", "mimo"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "evaluate/assets/tokenizer")
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data/eval/pg19/validation-00000-of-00001.parquet",
    )
    parser.add_argument("--lengths", default="16000,64000")
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--max-books", type=int, default=0, help="0 uses every eligible book")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eager", action="store_true", help="Disable CUDA graph replay")
    parser.add_argument("--no-kernel-policy", action="store_true")
    args = parser.parse_args()

    lengths = tuple(sorted({int(value) for value in args.lengths.split(",")}))
    if not lengths or not 0 < args.warmup < min(lengths):
        parser.error("require 0 < warmup < every measurement length")
    if args.max_books < 0:
        parser.error("--max-books must be nonnegative")
    model_path = args.model_path or (
        ROOT / f"checkpoints/officialpretrained/mamba3-{args.model}-1.5b"
    )
    output = args.output or (
        ROOT / "interventions/results/table6" / args.model / args.condition
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)

    model, checkpoint = load_model(args.model, model_path, device)
    kernel_policy = None if args.no_kernel_policy else apply_paper_kernel_policy(args.model)
    books = [book for book in load_books(args.dataset, args.tokenizer, 0)
             if len(book["token_ids"]) >= lengths[-1]]
    if args.max_books:
        books = books[:args.max_books]
    if not books:
        raise ValueError("no books are long enough")

    spec = {
        "version": 1,
        "paper_table": 6,
        "model": args.model,
        "condition": args.condition,
        "checkpoint": checkpoint,
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "split": "validation",
        "book_selection": "all books with at least max(lengths) tokens, source order",
        "lengths": list(lengths),
        "warmup_tokens_excluded": args.warmup,
        "trajectory": "one uninterrupted teacher-forced decode from zero state per book/condition",
        "per_book": "temporal mean per layer/head; median over layer/head",
        "state_rms": "sqrt(median layer/head of temporal mean squared Frobenius norm)",
        "aggregation": "equal-weight arithmetic mean across books",
        "kernel_policy": kernel_policy,
        "cuda_graph": not args.eager,
        "max_books": args.max_books,
        "software": intervention_software_identity(),
    }
    run_signature = signature(spec)
    metadata_path = output / "metadata.json"
    rows_path = output / "books.jsonl"
    with exclusive_lock(output / "run.lock"):
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["signature"] != run_signature:
                raise ValueError(f"output belongs to a different run: {output}")
        else:
            atomic_json(
                metadata_path,
                {"signature": run_signature, "spec": spec, "complete": False},
            )
        rows = []
        if rows_path.exists():
            rows = [
                json.loads(line)
                for line in rows_path.read_text().splitlines()
                if line.strip()
            ]
        completed = {row["book_index"] for row in rows}
        rows_are_valid = all(
            row.get("model") == args.model
            and row.get("condition") == args.condition
            and {value["length"] for value in row.get("measurements", [])} == set(lengths)
            and all(
                math.isfinite(value[field])
                for value in row.get("measurements", [])
                for field in ("cosine", "state_rms")
            )
            for row in rows
        )
        if (
            len(rows) != len(completed)
            or not completed <= {book["book_index"] for book in books}
            or not rows_are_valid
        ):
            raise ValueError("output contains duplicate or unexpected books")
        with rows_path.open("a", encoding="utf-8", buffering=1) as stream:
            for book in books:
                if book["book_index"] in completed:
                    continue
                values, seconds = trace(
                    model, book["token_ids"][:lengths[-1]].to(device), args.model,
                    args.condition, lengths, args.warmup, not args.eager,
                )
                row = {
                    "model": args.model, "condition": args.condition,
                    "book_index": book["book_index"], "seconds": seconds,
                    "measurements": values,
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
                completed.add(book["book_index"])
                flattened = [
                    {"book_index": item["book_index"], **measurement}
                    for item in rows for measurement in item["measurements"]
                ]
                atomic_json(output / "summary.json", {
                    "signature": run_signature,
                    "complete": len(completed) == len(books),
                    "completed_books": len(completed), "expected_books": len(books),
                    "measurements": aggregate(flattened, lengths, len(books)),
                })
                print(
                    f"BOOK COMPLETE {book['book_index']}: "
                    f"{len(completed)}/{len(books)}",
                    flush=True,
                )
        atomic_json(metadata_path, {
            "signature": run_signature, "spec": spec,
            "complete": len(completed) == len(books),
            "completed_books": len(completed), "expected_books": len(books),
        })
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
