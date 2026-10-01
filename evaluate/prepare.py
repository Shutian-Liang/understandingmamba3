"""Prepare local SlimPajama, CodeParrot, MATH-Hard, and TriviaQA PPL caches."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
import time
from pathlib import Path

from evaluate.common import ROOT, DATASETS, atomic_json, exclusive_lock, load_tokenizer, read_json, sha256, signature
from evaluate._prepare_reconstructed import (
    PAPER_PREP_DATASETS,
    prepare_one as prepare_reconstructed,
)


def sources(name, root):
    patterns = {
        "codeparrot": "eval/codeparrot_clean_valid/*.json.gz",
        "slimpajama": "eval/slimpajama_test/data/*.parquet",
    }
    paths = sorted(Path(root).glob(patterns[name]))
    if not paths:
        raise FileNotFoundError(f"No {name} sources: {Path(root) / patterns[name]}")
    return paths


def records(name, paths):
    for path in paths:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                yield from batch.to_pylist()
        else:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)


def text_fields(name, row):
    return "", row["content"] if name == "codeparrot" else row["text"]


def prepare_one(name, args, tokenizer):
    import numpy as np
    paths = sources(name, args.data_root)
    source_files = [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths]
    spec = {"version": 1, "dataset": name, "sources": source_files,
            "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
            "add_special_tokens": False, "append_eos": False, "document_packing": False,
            "score": "body_next_token", "text_format": "original_text",
            "max_documents": args.max_documents}
    dest = args.cache_root / name
    with exclusive_lock(args.cache_root / (name + ".lock")):
        if (dest / "manifest.json").exists():
            old = read_json(dest / "manifest.json")
            if old["spec"] != spec:
                raise ValueError(f"Different cache already exists at {dest}; select another --cache-root")
            for filename, expected in old["files"].items():
                if sha256(dest / filename) != expected["sha256"]:
                    raise ValueError(f"Corrupt cache: {dest / filename}")
            print(f"CACHE VERIFIED {name}: {old['documents']} documents, {old['tokens']} tokens", flush=True)
            return
        if dest.exists():
            raise ValueError(f"Incomplete directory exists: {dest}; inspect it before retrying")
        tmp = Path(tempfile.mkdtemp(prefix=name + ".preparing-", dir=args.cache_root))
        total = scored = count = skipped = 0
        max_len = 0
        long_counts = {str(c): 0 for c in (1024, 2048, 4096)}
        started = time.monotonic()
        with (tmp / "tokens.bin").open("wb") as binary, (tmp / "index.jsonl").open("w", encoding="utf-8") as index:
            for row_i, row in enumerate(records(name, paths)):
                if args.max_documents and row_i >= args.max_documents:
                    break
                prefix, text = text_fields(name, row)
                if not isinstance(text, str):
                    raise ValueError(f"{name} row {row_i} text is not a string")
                encoding = tokenizer.encode(prefix + text, add_special_tokens=False)
                ids = encoding.ids
                score_start = 1
                if prefix:
                    # Include the boundary token if it contains any answer character.
                    score_start = next((i for i, (_, end) in enumerate(encoding.offsets) if end > len(prefix)), len(ids))
                    score_start = max(1, score_start)
                if len(ids) <= score_start:
                    skipped += 1
                    continue
                np.asarray(ids, dtype="<u4").tofile(binary)
                index.write(json.dumps({"row": row_i, "offset": total, "length": len(ids),
                                        "score_start": score_start}) + "\n")
                count += 1
                total += len(ids)
                scored += len(ids) - score_start
                max_len = max(max_len, len(ids))
                for c in long_counts:
                    long_counts[c] += len(ids) > int(c)
                if count % 10000 == 0:
                    print(f"PREPARE {name}: docs={count} tokens={total} seconds={time.monotonic()-started:.1f}", flush=True)
        if not count:
            raise ValueError(f"No scorable documents in {name}")
        files = {f: {"bytes": (tmp/f).stat().st_size, "sha256": sha256(tmp/f)} for f in ("tokens.bin", "index.jsonl")}
        manifest = {"spec": spec, "signature": signature(spec), "documents": count, "skipped_empty": skipped,
                    "tokens": total, "scored_tokens": scored, "maximum_document_tokens": max_len,
                    "documents_longer_than_context": long_counts, "files": files,
                    "seconds": time.monotonic()-started}
        atomic_json(tmp / "manifest.json", manifest)
        os.rename(tmp, dest)
        print(f"PREPARED {name}: {count} documents, {total} tokens, {scored} scored", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "evaluate/assets/tokenizer")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "evaluate/cache")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--max-documents", type=int, default=0, help="Smoke only; 0 means all")
    args = parser.parse_args()
    if args.max_documents < 0:
        parser.error("--max-documents must be nonnegative")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    for name in args.datasets:
        if name in PAPER_PREP_DATASETS:
            prepare_reconstructed(name, args, tokenizer)
        else:
            prepare_one(name, args, tokenizer)


if __name__ == "__main__":
    main()
