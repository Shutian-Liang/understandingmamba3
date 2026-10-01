"""Prepare reconstructed MATH-Hard and TriviaQA text caches for paper PPL."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from evaluate.common import atomic_json, exclusive_lock, read_json, sha256, signature


PAPER_PREP_DATASETS = ("math_hard", "trivia_qa")


def trivia_evidence_text(row):
    contexts = []
    for nested, field in ((row.get("entity_pages"), "wiki_context"),
                          (row.get("search_results"), "search_context")):
        if isinstance(nested, dict):
            values = nested.get(field) or []
        else:
            values = [item.get(field) for item in (nested or []) if isinstance(item, dict)]
        for text in values:
            if isinstance(text, str) and text.strip():
                contexts.append(text)
    return "\n\n".join(contexts)


def sources_for(dataset, data_root):
    pattern = ("eval/math_hard/train/*.jsonl" if dataset == "math_hard" else
               "eval/trivia_qa_rc/rc/validation-*.parquet")
    sources = sorted(data_root.glob(pattern))
    if not sources:
        raise FileNotFoundError(f"Missing {dataset} sources: {data_root / pattern}")
    return sources


def rows_for(dataset, sources):
    if dataset == "math_hard":
        for path in sources:
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream)
            yield from value
        return
    import pyarrow.parquet as pq
    for path in sources:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            yield from batch.to_pylist()


def text_for(dataset, row):
    if dataset == "math_hard":
        problem, solution = row.get("problem"), row.get("solution")
        if not isinstance(problem, str) or not isinstance(solution, str):
            return ""
        return f"Problem: {problem}\nSolution:\n{solution}"
    return trivia_evidence_text(row)


def spec_for(dataset, args, sources):
    source_files = [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                     "sha256": sha256(path)} for path in sources]
    common = {
        "version": 2,
        "dataset": dataset,
        "sources": source_files,
        "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "add_special_tokens": False,
        "virtual_bos_at_evaluation": True,
        "document_packing": False,
        "max_documents": args.max_documents,
    }
    if dataset == "math_hard":
        common.update({
            "huggingface_id": "lighteval/MATH-Hard",
            "revision": "cf0716b8bafa192bcb6b455b0679538787dc43f0",
            "config": "default",
            "split": "train",
            "text_reconstruction": "Problem: {problem}\\nSolution:\\n{solution}",
            "reproducibility_note": "The paper cites MATH-Hard and its evaluator defaults to split=train, but does not release the conversion to its required text column; this is a documented reconstruction.",
        })
    else:
        common.update({
            "huggingface_id": "mandarjoshi/trivia_qa",
            "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
            "config": "rc",
            "split": "validation",
            "text_reconstruction": "join entity_pages.wiki_context then search_results.search_context per question with two newlines",
            "reproducibility_note": "The paper cites TriviaQA but does not release the conversion to its required text column; this is an evidence-based reconstruction.",
        })
    return common


def prepare_one(dataset, args, tokenizer):
    import numpy as np

    sources = sources_for(dataset, args.data_root)
    spec = spec_for(dataset, args, sources)
    cache_name = "paper_" + dataset
    destination = args.cache_root / cache_name
    with exclusive_lock(args.cache_root / (cache_name + ".lock")):
        if destination.exists():
            old = read_json(destination / "manifest.json")
            if old["spec"] != spec:
                raise ValueError(f"Different cache exists at {destination}")
            for filename, expected in old["files"].items():
                if sha256(destination / filename) != expected["sha256"]:
                    raise ValueError(f"Corrupt cache: {destination / filename}")
            print(f"CACHE VERIFIED {cache_name}: {old['documents']} documents", flush=True)
            return

        temp = Path(tempfile.mkdtemp(prefix=cache_name + ".preparing-", dir=args.cache_root))
        total = documents = skipped = 0
        max_length = 0
        longer = {str(value): 0 for value in (1024, 2048, 4096)}
        started = time.monotonic()
        with (temp / "tokens.bin").open("wb") as binary, \
             (temp / "index.jsonl").open("w", encoding="utf-8") as index:
            for row_index, row in enumerate(rows_for(dataset, sources)):
                if args.max_documents and documents >= args.max_documents:
                    break
                text = text_for(dataset, row)
                if not text:
                    skipped += 1
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not ids:
                    skipped += 1
                    continue
                np.asarray(ids, dtype="<u4").tofile(binary)
                index.write(json.dumps({"row": row_index, "offset": total,
                                        "length": len(ids)}) + "\n")
                documents += 1
                total += len(ids)
                max_length = max(max_length, len(ids))
                for threshold in longer:
                    longer[threshold] += len(ids) + 1 > int(threshold)
                if documents % 1000 == 0:
                    print(f"PREPARE {cache_name} docs={documents} tokens={total}", flush=True)
        if not documents:
            raise ValueError(f"No {dataset} documents were prepared")
        files = {name: {"bytes": (temp / name).stat().st_size,
                        "sha256": sha256(temp / name)}
                 for name in ("tokens.bin", "index.jsonl")}
        manifest = {"spec": spec, "signature": signature(spec), "documents": documents,
                    "skipped_empty": skipped, "tokens": total,
                    "maximum_document_tokens": max_length,
                    "documents_longer_than_position": longer, "files": files,
                    "seconds": time.monotonic() - started}
        atomic_json(temp / "manifest.json", manifest)
        os.rename(temp, destination)
        print(f"PREPARED {cache_name}: {documents} documents, {total} tokens", flush=True)
