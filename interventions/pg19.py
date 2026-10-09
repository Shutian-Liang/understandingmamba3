"""Reproduce the pretrained-model PG-19 interventions in Tables 5 and 7."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from evaluate.common import (
    atomic_json,
    exclusive_lock,
    sha256,
    signature,
    software_identity,
)
from interventions.phase_final_100.run import TailBCPhase, predictor_bounds
from interventions.phase_interventions.run import paper_table5_intervention
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.paper_kernel_policy import apply_paper_kernel_policy


ROOT = Path(__file__).resolve().parents[1]
TABLE5 = ("native", "decay_permutation", "phase_shuffle", "phase_removal", "phase_reversal")
TABLE7 = ("native", "remove_b", "remove_c", "remove_both")
TABLE7_INTERNAL = {"remove_b": "b_only", "remove_c": "c_only", "remove_both": "both"}


def starts(length: int, context: int, windows: int = 10) -> range:
    """LongMamba window positions used by the paper; exact fits are excluded."""
    if length <= context:
        return range(0)
    return range(0, length - context, max(10, (length - context) // windows))


def intervention_software_identity():
    identity = software_identity()
    identity["code_sha256"].update({
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted((ROOT / "interventions").rglob("*.py"))
    })
    return identity


def load_model(name: str, path: Path, device: torch.device):
    config_path = path / "config.json"
    weights_path = path / "pytorch_model.bin"
    config = MambaConfig(**json.loads(config_path.read_text()))
    if config.ssm_cfg.get("layer") != "Mamba3":
        raise ValueError(f"expected a Mamba3 checkpoint: {path}")
    if bool(config.ssm_cfg.get("is_mimo", False)) != (name == "mimo"):
        raise ValueError(f"checkpoint type does not match --model {name}: {path}")
    model = MambaLMHeadModel(config, device=device, dtype=torch.bfloat16)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval().requires_grad_(False)
    return model, {
        "path": str(path.resolve()),
        "config_sha256": sha256(config_path),
        "weights_sha256": sha256(weights_path),
    }


def load_books(dataset: Path, tokenizer_path: Path, max_books: int):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.model_max_length = 10**12
    texts = pq.read_table(dataset, columns=["text"])["text"]
    count = min(len(texts), max_books) if max_books else len(texts)
    books = []
    for index in range(count):
        ids = tokenizer.encode(str(texts[index].as_py()), add_special_tokens=False)
        books.append({"book_index": index, "token_ids": torch.tensor(ids, dtype=torch.long)})
        print(f"TOKENIZED book={index} tokens={len(ids)}", flush=True)
    return books


@torch.inference_mode()
def score(model, ids: torch.Tensor, prediction_length: int, context_manager, device):
    window = ids.unsqueeze(0).to(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with context_manager:
        logits = model(window, num_last_tokens=prediction_length + 1).logits[:, :-1].float()
    targets = window[:, -prediction_length:]
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum",
    )
    torch.cuda.synchronize(device)
    return float(nll), time.monotonic() - started, int(logits.shape[-1])


def summarize(rows, expected_by_context):
    output = []
    for context, expected in expected_by_context.items():
        selected = [row for row in rows if row["context_length"] == context]
        if not selected:
            continue
        tokens = sum(row["target_tokens"] for row in selected)
        nll = math.fsum(row["nll"] for row in selected)
        output.append({
            "context_length": context,
            "windows": len(selected),
            "expected_windows": expected,
            "complete": len(selected) == expected,
            "target_tokens": tokens,
            "nll": nll,
            "mean_nll": nll / tokens,
            "perplexity": math.exp(nll / tokens),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=("5", "7"), required=True)
    parser.add_argument("--model", choices=("siso", "mimo"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "evaluate/assets/tokenizer")
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data/eval/pg19/test-00000-of-00001.parquet",
    )
    parser.add_argument("--contexts", default="2000,16000,64000")
    parser.add_argument("--prediction-length", type=int, default=100)
    parser.add_argument("--windows-per-book", type=int, default=10)
    parser.add_argument("--max-books", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run one window per context")
    parser.add_argument("--no-kernel-policy", action="store_true")
    args = parser.parse_args()

    choices = TABLE5 if args.table == "5" else TABLE7
    if args.condition not in choices:
        parser.error(f"--condition for Table {args.table} must be one of {choices}")
    contexts = tuple(sorted({int(value) for value in args.contexts.split(",")}))
    if args.table == "7":
        contexts = (64000,)
    if not contexts or min(contexts) <= args.prediction_length:
        parser.error("contexts must exceed --prediction-length")
    if args.prediction_length < 1 or args.windows_per_book < 1 or args.max_books < 1:
        parser.error("prediction length, windows per book, and max books must be positive")

    model_path = args.model_path or (
        ROOT / f"checkpoints/officialpretrained/mamba3-{args.model}-1.5b"
    )
    output = args.output or (
        ROOT / "interventions/results" / f"table{args.table}" / args.model / args.condition
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)

    model, checkpoint = load_model(args.model, model_path, device)
    kernel_policy = None if args.no_kernel_policy else apply_paper_kernel_policy(args.model)
    books = load_books(args.dataset, args.tokenizer, args.max_books)
    expected = []
    for context in contexts:
        for book in books:
            positions = list(starts(len(book["token_ids"]), context, args.windows_per_book))
            if args.smoke:
                positions = positions[:1]
            expected.extend((context, book["book_index"], begin) for begin in positions)
        if args.smoke and expected:
            # One deterministic window per context, from the first eligible book.
            candidates = [key for key in expected if key[0] == context]
            expected = [key for key in expected if key[0] != context] + candidates[:1]
    expected = sorted(expected)
    if not expected:
        raise ValueError("no eligible windows")
    expected_set = set(expected)
    expected_by_context = {c: sum(key[0] == c for key in expected) for c in contexts}

    spec = {
        "version": 1,
        "paper_table": int(args.table),
        "model": args.model,
        "condition": args.condition,
        "checkpoint": checkpoint,
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "contexts": list(contexts),
        "prediction_length": args.prediction_length,
        "candidate_books": args.max_books,
        "windows_per_book": args.windows_per_book,
        "sampling": "range(0, L-C, max(10, floor((L-C)/K))); exact fits excluded",
        "scoring": "final prediction_length next-token targets; FP32 summed NLL",
        "aggregation": "exp(sum NLL / sum target tokens)",
        "state": "fresh zero recurrent state for every window",
        "batch_size": 1,
        "seed": args.seed,
        "kernel_policy": kernel_policy,
        "smoke": args.smoke,
        "software": intervention_software_identity(),
    }
    run_signature = signature(spec)
    metadata_path = output / "metadata.json"
    rows_path = output / "windows.jsonl"
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
        completed = {(r["context_length"], r["book_index"], r["start_token"]) for r in rows}
        rows_are_valid = all(
            row.get("table") == int(args.table)
            and row.get("model") == args.model
            and row.get("condition") == args.condition
            and row.get("target_tokens") == args.prediction_length
            and math.isfinite(row.get("nll", math.nan))
            for row in rows
        )
        if len(completed) != len(rows) or not completed <= expected_set or not rows_are_valid:
            raise ValueError("output contains duplicate or unexpected windows")
        book_map = {book["book_index"]: book for book in books}
        with rows_path.open("a", encoding="utf-8", buffering=1) as stream:
            for context, book_index, begin in expected:
                key = (context, book_index, begin)
                if key in completed:
                    continue
                ids = book_map[book_index]["token_ids"][begin:begin + context]
                if args.condition == "native":
                    manager = contextlib.nullcontext()
                elif args.table == "5":
                    manager = paper_table5_intervention(
                        model, args.condition, context, seed=args.seed,
                    )
                else:
                    phase_start, phase_end = predictor_bounds(context, args.prediction_length)
                    manager = TailBCPhase(
                        args.model, TABLE7_INTERNAL[args.condition], phase_start, phase_end,
                    )
                nll, seconds, vocabulary = score(
                    model, ids, args.prediction_length, manager, device,
                )
                row = {
                    "table": int(args.table), "model": args.model,
                    "condition": args.condition, "context_length": context,
                    "book_index": book_index, "start_token": begin,
                    "end_token": begin + context,
                    "target_begin": begin + context - args.prediction_length,
                    "target_end": begin + context,
                    "target_tokens": args.prediction_length,
                    "nll": nll, "mean_nll": nll / args.prediction_length,
                    "perplexity": math.exp(nll / args.prediction_length),
                    "seconds": seconds, "logit_vocabulary": vocabulary,
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
                completed.add(key)
                atomic_json(output / "summary.json", {
                    "signature": run_signature,
                    "complete": completed == expected_set,
                    "completed_windows": len(completed),
                    "expected_windows": len(expected),
                    "contexts": summarize(rows, expected_by_context),
                })
                print(
                    f"{len(completed)}/{len(expected)} table={args.table} model={args.model} "
                    f"condition={args.condition} context={context} book={book_index} "
                    f"start={begin} ppl={row['perplexity']:.6f}", flush=True,
                )
        atomic_json(metadata_path, {
            "signature": run_signature, "spec": spec,
            "complete": completed == expected_set,
            "completed_windows": len(completed), "expected_windows": len(expected),
        })
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
