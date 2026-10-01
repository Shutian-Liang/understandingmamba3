"""LongBench-E generation with pinned official prompts and official metrics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

from evaluate.common import (ROOT, TASKS, atomic_json, exclusive_lock, load_model,
                             load_tokenizer, read_json, read_jsonl, sha256,
                             signature, software_identity)

ASSETS = ROOT / "evaluate/assets/longbench"


def length_bucket(length):
    return "0-4k" if length < 4000 else "4-8k" if length < 8000 else "8k+"


def metric_functions():
    spec = importlib.util.spec_from_file_location("official_longbench_metrics", ASSETS / "metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    names = {task: "qa_f1_score" for task in TASKS}
    names.update({task: "rouge_score" for task in ("gov_report", "multi_news", "samsum")})
    names.update(trec="classification_score", passage_count="count_score",
                 passage_retrieval_en="retrieval_score", lcc="code_sim_score")
    names["repobench-p"] = "code_sim_score"
    return {task: getattr(metrics, name) for task, name in names.items()}


def score_prediction(task, prediction, answers, all_classes, metrics=None):
    metrics = metric_functions() if metrics is None else metrics
    if task in ("trec", "triviaqa", "samsum", "lsht"):
        prediction = prediction.lstrip("\n").split("\n")[0]
    if not answers:
        raise ValueError(f"{task}: missing gold answers")
    score = max(metrics[task](prediction, answer, all_classes=all_classes) for answer in answers)
    if not math.isfinite(score):
        raise FloatingPointError("Nonfinite LongBench score")
    return float(score)


def generate_batch(model, prompt_ids_batch, max_new_tokens, eos_ids, min_new_tokens=0):
    import torch
    from evaluate.forward import hidden_forward
    sequences = [list(ids) for ids in prompt_ids_batch]
    generated = [[] for _ in sequences]
    active = list(range(len(sequences)))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range(max_new_tokens):
            if not active:
                break
            lengths = [len(sequences[index]) for index in active]
            ids = torch.full((len(active), max(lengths)), 128001, dtype=torch.long, device="cuda")
            for row, index in enumerate(active):
                ids[row, :lengths[row]] = torch.tensor(sequences[index], dtype=torch.long, device="cuda")
            hidden = hidden_forward(model, ids)
            last = torch.stack([hidden[row, length-1] for row, length in enumerate(lengths)])
            logits = model.lm_head(last)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite generation logits")
            if step < min_new_tokens:
                logits[:, list(eos_ids)] = -float("inf")
            tokens = logits.argmax(dim=-1).tolist()
            remaining = []
            for index, token in zip(active, tokens):
                if token not in eos_ids:
                    generated[index].append(token)
                    sequences[index].append(token)
                    remaining.append(index)
            active = remaining
            # Full-forward decoding recomputes the prompt on every step. Drop
            # references now so the next forward does not overlap two complete
            # sets of 64K activations in memory.
            del ids, hidden, last, logits
    return generated


def generate(model, prompt_ids, max_new_tokens, eos_ids, min_new_tokens=0):
    return generate_batch(model, [prompt_ids], max_new_tokens, eos_ids, min_new_tokens)[0]


def evaluate_task(model, tokenizer, checkpoint, software, args, task, metrics):
    import torch
    source = args.data_root / f"{task}_e.jsonl"
    rows = read_jsonl(source)
    if args.max_examples:
        rows = rows[:args.max_examples]
    template = read_json(ASSETS / "config/dataset2prompt.json")[task]
    max_new = read_json(ASSETS / "config/dataset2maxlen.json")[task]
    if args.max_new_tokens:
        max_new = args.max_new_tokens
    batch_size = getattr(args, "batch_size", 1)
    spec = {"version": 2, "checkpoint": checkpoint, "software": software,
            "task": task, "source_sha256": sha256(source), "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": max_new, "template": template, "chat_template": None,
            "add_special_tokens": False, "decoding": "greedy_full_forward", "truncate": "official_middle_decode_reencode",
            "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
            "official_assets": {str(p.relative_to(ASSETS)): sha256(p) for p in sorted(ASSETS.rglob("*")) if p.is_file() and "__pycache__" not in str(p)},
            "rank": args.rank, "world_size": args.world_size, "max_examples": args.max_examples,
            "batch_size": batch_size}
    directory = args.results_root / args.run_name / "longbench_e" / str(args.max_prompt_tokens) / task
    output = directory / f"shard_{args.rank:03d}.jsonl"
    meta_path = directory / f"shard_{args.rank:03d}.meta.json"
    with exclusive_lock(output.with_suffix(".lock")):
        key = signature(spec)
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta["signature"] != key:
                raise ValueError(f"Different checkpoint/protocol at {meta_path}; choose another --run-name")
            if meta["status"] == "complete":
                print(f"DONE ALREADY LongBench-E {task} rank={args.rank}", flush=True)
                return
        else:
            if output.exists() and output.stat().st_size:
                raise ValueError(f"Prediction file without metadata: {output}")
            meta = {"signature": key, "spec": spec, "status": "running", "dataset_examples": len(rows)}
            atomic_json(meta_path, meta)
        completed = set()
        if output.exists():
            # Recover only an interrupted final JSONL record; preserve valid records.
            with output.open("rb+") as stream:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        stream.truncate(position)
                        break
                    row = json.loads(line)
                    index = row["index"]
                    if index in completed or index % args.world_size != args.rank:
                        raise ValueError(f"Duplicate or wrong-rank prediction: {output}")
                    completed.add(index)
        eos_ids = {tokenizer.token_to_id("<|end_of_text|>")}
        if task == "samsum":
            eos_ids.add(tokenizer.encode("\n", add_special_tokens=False).ids[-1])
        if None in eos_ids:
            raise ValueError("Missing EOS token")
        torch.cuda.reset_peak_memory_stats()
        with output.open("a", encoding="utf-8") as stream:
            pending = [index for index in range(args.rank, len(rows), args.world_size) if index not in completed]
            for begin in range(0, len(pending), batch_size):
                indexes = pending[begin:begin+batch_size]
                prepared = []
                for index in indexes:
                    item = rows[index]
                    ids = tokenizer.encode(template.format(**item), add_special_tokens=False).ids
                    original_length = len(ids)
                    if len(ids) > args.max_prompt_tokens:
                        # Match official pred.py: decode the two halves, concatenate,
                        # then tokenize the text again (can differ slightly in length).
                        half = args.max_prompt_tokens // 2
                        prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(ids[-half:], skip_special_tokens=True)
                        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                    prepared.append((index, item, ids, original_length))
                started = time.monotonic()
                outputs = generate_batch(model, [entry[2] for entry in prepared], max_new, eos_ids,
                                         min_new_tokens=int(task == "samsum"))
                batch_seconds = time.monotonic() - started
                for (index, item, ids, original_length), output_ids in zip(prepared, outputs):
                    prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
                    score = score_prediction(task, prediction, item["answers"], item["all_classes"], metrics)
                    record = {"index": index, "_id": item.get("_id"), "pred": prediction,
                              "answers": item["answers"], "all_classes": item["all_classes"],
                              "length": item["length"], "length_bucket": length_bucket(item["length"]),
                              "score": score, "original_prompt_tokens": original_length,
                              "prompt_tokens": len(ids), "truncated": original_length > len(ids),
                              "generated_tokens": len(output_ids), "batch_seconds": batch_seconds,
                              "batch_examples": len(prepared)}
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed.add(index)
                    print(f"LBE {task} rank={args.rank} example={index}/{len(rows)} score={100*score:.2f} gen={len(output_ids)} batch_seconds={batch_seconds:.1f}", flush=True)
        if len(completed) != len(range(args.rank, len(rows), args.world_size)):
            raise AssertionError("LongBench prediction coverage mismatch")
        meta.update(status="smoke_complete" if args.max_examples or args.max_new_tokens else "complete",
                    completed_examples=len(completed), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(meta_path, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/eval/longbench_e/data")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "evaluate/assets/tokenizer")
    parser.add_argument("--results-root", type=Path, default=ROOT / "evaluate/results")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--max-prompt-tokens", type=int, default=65536)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="Smoke only; 0=official per-task limits")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not args.run_name:
        args.run_name = args.checkpoint.expanduser().resolve().parent.name
    if (not 0 <= args.rank < args.world_size or args.max_prompt_tokens < 2
            or min(args.max_examples, args.max_new_tokens) < 0 or args.batch_size < 1):
        parser.error("Invalid rank, prompt budget or limit")
    tokenizer = load_tokenizer(args.tokenizer)
    metrics = metric_functions()
    model, checkpoint = load_model(args.checkpoint, args.allow_incomplete)
    software = software_identity()
    for task in args.tasks:
        evaluate_task(model, tokenizer, checkpoint, software, args, task, metrics)


if __name__ == "__main__":
    main()
