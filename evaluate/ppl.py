"""Evaluate PPL on SlimPajama, CodeParrot, MATH-Hard, and TriviaQA.

The paper protocol appends EOS to every source row, concatenates rows inside
each 1000-row source batch, discards the incomplete tail, splits the stream
into fixed-length sequences, and averages NLL by absolute position.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

from evaluate.common import (ROOT, atomic_json, exclusive_lock, load_model,
                             read_json, signature, software_identity)


PAPER_DATASETS = ("codeparrot", "math_hard", "trivia_qa", "slimpajama")
PAPER_SEQUENCE_LENGTHS = (1024, 2048, 4096)
EOS_TOKEN_ID = 128001


def source_cache_name(dataset):
    return f"paper_{dataset}" if dataset in {"math_hard", "trivia_qa"} else dataset


def packing_stats(documents, sequence_length, source_batch_size):
    sequences = packed_tokens = dropped_tokens = 0
    for start in range(0, len(documents), source_batch_size):
        group = documents[start:start + source_batch_size]
        tokens = sum(document["length"] + 1 for document in group)
        complete = tokens // sequence_length
        sequences += complete
        packed_tokens += complete * sequence_length
        dropped_tokens += tokens - complete * sequence_length
    return {"sequences": sequences, "packed_tokens": packed_tokens,
            "dropped_tokens": dropped_tokens}


def iter_packed_sequences(tokens, documents, sequence_length, source_batch_size):
    """Yield fixed arrays with EOS between rows and a dropped tail per source batch."""
    import numpy as np

    for group_start in range(0, len(documents), source_batch_size):
        group = documents[group_start:group_start + source_batch_size]
        output = np.empty(sequence_length, dtype="<u4")
        filled = 0
        for document in group:
            pieces = (tokens[document["offset"]:document["offset"] + document["length"]],
                      (EOS_TOKEN_ID,))
            for piece in pieces:
                position = 0
                while position < len(piece):
                    take = min(sequence_length - filled, len(piece) - position)
                    output[filled:filled + take] = piece[position:position + take]
                    filled += take
                    position += take
                    if filled == sequence_length:
                        yield output
                        output = np.empty(sequence_length, dtype="<u4")
                        filled = 0
        # Historical datasets.map preprocessing truncated this tail separately
        # for every default 1000-row source batch.


def packed_bucket_nll(model, sequences, sequence_length, bucket_size, head_rows):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from evaluate.forward import hidden_forward

    values = np.stack(sequences).astype(np.int64, copy=False)
    input_ids = torch.from_numpy(values).to(device="cuda", dtype=torch.long)
    targets = input_ids[:, 1:]
    count = math.ceil((sequence_length - 1) / bucket_size)
    losses = [0.0] * count
    counts = [0] * count
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = hidden_forward(model, input_ids)
        for bucket, start in enumerate(range(0, sequence_length - 1, bucket_size)):
            end = min(start + bucket_size, sequence_length - 1)
            for row in range(0, len(sequences), head_rows):
                row_end = min(row + head_rows, len(sequences))
                logits = model.lm_head(hidden[row:row_end, start:end])
                labels = targets[row:row_end, start:end]
                losses[bucket] += F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    reduction="sum").double().item()
                counts[bucket] += labels.numel()
    if not all(math.isfinite(value) for value in losses):
        raise FloatingPointError("Nonfinite packed position-bucket NLL")
    return losses, counts


def evaluate_one(model, checkpoint, software, args, dataset):
    import numpy as np
    import torch

    cache = args.cache_root / source_cache_name(dataset)
    manifest = read_json(cache / "manifest.json")
    tokens = np.memmap(cache / "tokens.bin", mode="r", dtype="<u4")
    documents = [json.loads(line) for line in
                 (cache / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    stats = packing_stats(documents, args.sequence_length, args.source_batch_size)
    total_sequences = stats["sequences"]
    if args.max_sequences:
        total_sequences = min(total_sequences, args.max_sequences)
    spec = {
        "version": 1,
        "protocol": "paper_figure_ppl_early_fla",
        "reference": "fla-org/flash-linear-attention@353cbf1:evaluate/evals/ppl.py",
        "checkpoint": checkpoint,
        "software": software,
        "dataset": dataset,
        "source_cache": source_cache_name(dataset),
        "cache_signature": signature(manifest),
        "sequence_length": args.sequence_length,
        "bucket_size": args.bucket_size,
        "source_batch_size": args.source_batch_size,
        "append_eos_per_document": True,
        "score_eos_targets": True,
        "add_bos": False,
        "document_packing": True,
        "state_reset": "packed_sequence_only",
        "drop_incomplete_tail_per_source_batch": True,
        "batch_size": args.batch_size,
        "head_rows": args.head_rows,
        "max_sequences": args.max_sequences,
        "packing": stats,
    }
    key = signature(spec)
    label = f"seq_{args.sequence_length}_bucket_{args.bucket_size}"
    out = (args.results_root / args.run_name / "ppl" / dataset /
           label / "result.json")
    with exclusive_lock(out.with_suffix(".lock")):
        if out.exists():
            result = read_json(out)
            if result["signature"] != key:
                raise ValueError(f"Result belongs to another protocol/checkpoint: {out}")
            if result["status"] == "complete":
                print(f"DONE ALREADY paper_figure_ppl {dataset}", flush=True)
                return
        else:
            result = {
                "signature": key,
                "spec": spec,
                "status": "running",
                "cursor": 0,
                "batches": 0,
                "seconds": 0.0,
                "peak_allocated_gib": 0.0,
                "overall_ppl": None,
                "buckets": [
                    {"index": i, "predictor_start": i * args.bucket_size,
                     "predictor_end": min((i + 1) * args.bucket_size,
                                          args.sequence_length - 1),
                     "nll_sum": 0.0, "scored_tokens": 0, "ppl": None}
                    for i in range(math.ceil((args.sequence_length - 1) /
                                             args.bucket_size))
                ],
            }
        started = last_save = time.monotonic()
        base_seconds = result["seconds"]
        torch.cuda.reset_peak_memory_stats()

        def save():
            result["seconds"] = base_seconds + time.monotonic() - started
            result["peak_allocated_gib"] = max(
                result["peak_allocated_gib"], torch.cuda.max_memory_allocated() / 2**30)
            total_nll = total_tokens = 0
            for bucket in result["buckets"]:
                count = bucket["scored_tokens"]
                bucket["ppl"] = math.exp(bucket["nll_sum"] / count) if count else None
                total_nll += bucket["nll_sum"]
                total_tokens += count
            result["overall_ppl"] = math.exp(total_nll / total_tokens) if total_tokens else None
            atomic_json(out, result)

        save()
        stream = iter_packed_sequences(tokens, documents, args.sequence_length,
                                       args.source_batch_size)
        stream = itertools.islice(stream, result["cursor"], total_sequences)
        while result["cursor"] < total_sequences:
            sequences = list(itertools.islice(stream, args.batch_size))
            if not sequences:
                raise AssertionError("Packed stream ended before its counted length")
            nlls, counts = packed_bucket_nll(
                model, sequences, args.sequence_length, args.bucket_size, args.head_rows)
            for bucket, nll, count in zip(result["buckets"], nlls, counts):
                bucket["nll_sum"] += nll
                bucket["scored_tokens"] += count
            result["cursor"] += len(sequences)
            result["batches"] += 1
            if time.monotonic() - last_save > 45:
                save()
                print(f"PAPER_FIGURE {dataset} sequences={result['cursor']}/{total_sequences} "
                      f"ppl={result['overall_ppl']:.4f}", flush=True)
                last_save = time.monotonic()
        expected = total_sequences * (args.sequence_length - 1)
        observed = sum(bucket["scored_tokens"] for bucket in result["buckets"])
        if observed != expected:
            raise AssertionError(f"Scoring coverage mismatch: {observed} != {expected}")
        result["status"] = "smoke_complete" if args.max_sequences else "complete"
        save()
        print(f"{result['status'].upper()} paper_figure_ppl {dataset}: "
              f"sequences={total_sequences} ppl={result['overall_ppl']:.4f}", flush=True)


def main():
    def lengths(value):
        parsed = tuple(int(item) for item in value.split(",") if item.strip())
        if not parsed:
            raise argparse.ArgumentTypeError("provide at least one sequence length")
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "evaluate/cache")
    parser.add_argument("--results-root", type=Path, default=ROOT / "evaluate/results")
    parser.add_argument("--datasets", nargs="+", choices=PAPER_DATASETS,
                        default=list(PAPER_DATASETS))
    parser.add_argument("--sequence-length", type=int,
                        help="Run one length instead of the 1K/2K/4K paper grid")
    parser.add_argument("--sequence-lengths", type=lengths,
                        help="Comma-separated custom grid; cannot be combined with --sequence-length")
    parser.add_argument("--bucket-size", type=int, default=128)
    parser.add_argument("--source-batch-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--head-rows", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not args.run_name:
        args.run_name = args.checkpoint.expanduser().resolve().parent.name
    if args.sequence_length is not None and args.sequence_lengths is not None:
        parser.error("Use either --sequence-length or --sequence-lengths, not both")
    selected_lengths = (args.sequence_lengths or
                        ((args.sequence_length,) if args.sequence_length is not None
                         else PAPER_SEQUENCE_LENGTHS))
    if (any(length < 2 or length % args.bucket_size for length in selected_lengths) or
            len(set(selected_lengths)) != len(selected_lengths) or
            min(args.bucket_size, args.source_batch_size, args.batch_size,
                args.head_rows) < 1 or args.max_sequences < 0):
        parser.error("Invalid sequence, bucket, batch, head, or smoke setting")
    model, checkpoint = load_model(args.checkpoint, args.allow_incomplete)
    software = software_identity()
    for sequence_length in selected_lengths:
        args.sequence_length = sequence_length
        for dataset in args.datasets:
            evaluate_one(model, checkpoint, software, args, dataset)


if __name__ == "__main__":
    main()
