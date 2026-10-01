from __future__ import annotations

import argparse
import glob
import json
from itertools import islice
from pathlib import Path

import numpy as np

from evaluate.common import ROOT, atomic_json, load_tokenizer, sha256


class TokenWriter:
    def __init__(self, directory, split, shard_tokens):
        self.directory, self.split, self.shard_tokens = directory, split, shard_tokens
        self.total = self.in_shard = 0
        self.files = []
        self.stream = None

    def write(self, ids):
        values = np.asarray(ids, dtype="<u4")
        while len(values):
            if self.stream is None:
                path = self.directory / f"{self.split}_{len(self.files):05d}.bin"
                self.files.append(path)
                self.stream = path.open("wb")
                self.in_shard = 0
            count = min(len(values), self.shard_tokens - self.in_shard)
            values[:count].tofile(self.stream)
            values = values[count:]
            self.total += count
            self.in_shard += count
            if self.in_shard == self.shard_tokens:
                self.close()

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def parquet_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            yield from batch.to_pylist()


def staging_directory(destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; select another --output-dir")
    staging = destination.with_name(destination.name + ".preparing")
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def tokenized_documents(rows, tokenizer):
    texts = (row["text"] for row in rows
             if isinstance(row.get("text"), str) and row["text"].strip())
    while batch := list(islice(texts, 256)):
        for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
            yield encoding.ids


def prepare_tokens(args):
    tokenizer = load_tokenizer(args.tokenizer)
    eos = tokenizer.token_to_id("<|end_of_text|>")
    if eos != 128001:
        raise ValueError("Expected Llama-3.1 EOS 128001")
    if args.input:
        paths = [Path(p).resolve() for p in sorted(glob.glob(args.input))]
        if not paths:
            raise FileNotFoundError(args.input)
        source = {"files": [{"path": str(p), "sha256": sha256(p)} for p in paths]}
        rows = parquet_rows(paths)
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
        source = {"dataset": args.dataset, "subset": args.subset, "revision": revision, "split": "train"}
        rows = load_dataset(args.dataset, args.subset, revision=revision, split="train", streaming=True)
    output = staging_directory(args.output_dir)
    train = TokenWriter(output, "train", args.shard_tokens)
    val = TokenWriter(output, "val", args.shard_tokens)
    documents = {"train": 0, "val": 0}
    try:
        for ids in tokenized_documents(rows, tokenizer):
            ids.append(eos)
            # A document is assigned to one split only. If validation reaches its
            # token budget midway through a document, discard its remaining tail.
            split = "val" if val.total < args.val_tokens else "train"
            writer, budget = (val, args.val_tokens) if split == "val" else (train, args.train_tokens)
            writer.write(ids[:budget - writer.total])
            documents[split] += 1
            if sum(documents.values()) % 10000 == 0:
                print(f"tokens: train={train.total:,} val={val.total:,}", flush=True)
            if train.total == args.train_tokens and val.total == args.val_tokens:
                break
    finally:
        train.close()
        val.close()
    if train.total != args.train_tokens or val.total != args.val_tokens:
        raise ValueError(f"Source exhausted: train={train.total}, val={val.total}; incomplete files: {output}")
    files = train.files + val.files
    atomic_json(output / "meta.json", {
        "dtype": "uint32", "numpy_dtype": "<u4", "vocab_size": 128256,
        "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "source": source, "shuffle": False, "append_eos": True, "add_bos": False,
        "split_rule": "first documents to validation budget; all later documents to training",
        "documents": documents, "train_tokens": train.total, "val_tokens": val.total,
        "shard_tokens": args.shard_tokens,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    })
    output.rename(args.output_dir.resolve())
    print(f"Prepared training shards: {args.output_dir}", flush=True)


def prepare_probe(args):
    tokenizer = load_tokenizer(args.tokenizer)
    maximum = max(args.contexts)
    candidates = []
    for index, row in enumerate(parquet_rows([args.input])):
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        if len(ids) >= maximum:
            candidates.append((index, row.get("short_book_title", ""), ids))
    if len(candidates) < args.num_windows:
        raise ValueError(f"Only {len(candidates)} eligible books for {args.num_windows} windows")
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(len(candidates), args.num_windows, replace=False).tolist())
    rows, windows = [], []
    for cache_row, selected_index in enumerate(selected):
        book_index, title, ids = candidates[selected_index]
        begin = int(rng.integers(0, len(ids) - maximum + 1))
        end = begin + maximum
        rows.append(ids[begin:end])
        windows.append({
            "cache_row": cache_row, "book_index": book_index, "title": title,
            "book_window_begin": begin, "book_window_end": end,
            "book_target_begin": end - args.prediction_length, "book_target_end": end,
            "cache_target_begin": maximum - args.prediction_length, "cache_target_end": maximum,
        })
    output = staging_directory(args.output_dir)
    values = np.asarray(rows, dtype="<u4")
    tokens = output / "probe_tokens.uint32.bin"
    values.tofile(tokens)
    atomic_json(output / "manifest.json", {
        "format": "mamba3_pg19_fixed_probe", "format_version": 1,
        "source": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "seed": args.seed, "sampling": "uniform distinct eligible books; uniform window start per book",
        "protocol": {
            "context_lengths": args.contexts, "maximum_context_length": maximum,
            "prediction_length": args.prediction_length, "num_windows": len(rows),
            "one_window_per_distinct_book": True, "shared_target_endpoints_across_context_lengths": True,
        },
        "tokenizer": {"length": 128256, "sha256": sha256(args.tokenizer / "tokenizer.json")},
        "storage": {
            "tokens_file": tokens.name, "logical_dtype": "uint32", "numpy_dtype": "<u4",
            "shape": list(values.shape), "size_bytes": tokens.stat().st_size,
            "sha256": sha256(tokens), "minimum_token_id": int(values.min()),
            "maximum_token_id": int(values.max()),
        },
        "windows": windows,
    })
    from pretrain.pg19 import load_fixed_pg19_probe
    load_fixed_pg19_probe(output)
    output.rename(args.output_dir.resolve())
    print(f"Prepared PG-19 probe: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    tokens = subparsers.add_parser("tokens", help="Stream FineWeb-Edu or local Parquet into uint32 shards")
    tokens.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    tokens.add_argument("--subset", default="sample-350BT")
    tokens.add_argument("--revision", default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
    tokens.add_argument("--input", help="Quoted glob of local Parquet files instead of a Hub dataset")
    tokens.add_argument("--output-dir", type=Path, default=ROOT / "data/fineweb_edu_llama31_100B")
    tokens.add_argument("--train-tokens", type=int, default=100_000_000_000)
    tokens.add_argument("--val-tokens", type=int, default=10_000_000)
    tokens.add_argument("--shard-tokens", type=int, default=100_000_000)
    probe = subparsers.add_parser("pg19", help="Build a fixed probe with shared targets across context lengths")
    probe.add_argument("--input", type=Path, default=ROOT / "data/eval/pg19/test-00000-of-00001.parquet")
    probe.add_argument("--output-dir", type=Path, default=ROOT / "data/pg19_test/mamba3_llama31_fixed_probe")
    probe.add_argument("--contexts", type=lambda s: [int(x) for x in s.split(",")], default=[2000, 16000, 64000])
    probe.add_argument("--prediction-length", type=int, default=100)
    probe.add_argument("--num-windows", type=int, default=32)
    probe.add_argument("--seed", type=int, default=42)
    for child in (tokens, probe):
        child.add_argument("--tokenizer", type=Path, default=ROOT / "evaluate/assets/tokenizer")
    args = parser.parse_args()
    if args.action == "tokens":
        if min(args.train_tokens, args.val_tokens, args.shard_tokens) < 2049:
            parser.error("Token budgets/shards must contain at least 2049 tokens")
        if any(0 < budget % args.shard_tokens < 2049 for budget in (args.train_tokens, args.val_tokens)):
            parser.error("The final shard must contain at least 2049 tokens; change --shard-tokens")
        prepare_tokens(args)
    else:
        if (not args.contexts or args.contexts != sorted(set(args.contexts))
                or args.prediction_length < 1 or args.contexts[0] <= args.prediction_length
                or args.num_windows < 1):
            parser.error("Use increasing unique contexts > prediction length and a positive window count")
        prepare_probe(args)


if __name__ == "__main__":
    main()
