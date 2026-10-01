from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = (1024, 2048, 4096)
DATASETS = ("codeparrot", "math_hard", "trivia_qa", "slimpajama")
TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
         "multi_news", "trec", "triviaqa", "samsum", "passage_count",
         "passage_retrieval_en", "lcc", "repobench-p")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    # JSON strings may contain literal U+0085/U+2028/U+2029. str.splitlines()
    # incorrectly treats those as record separators; iterate physical lines.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def exclusive_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {path}") from exc
        yield


def load_tokenizer(path):
    from tokenizers import Tokenizer
    path = Path(path)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.get_vocab_size(with_added_tokens=True) != 128256:
        raise ValueError("Expected the training Llama-3.1 vocabulary (128256 entries)")
    return tokenizer


def load_model(checkpoint, allow_incomplete=False):
    """Only load locally trained checkpoints. FP32 weights + BF16 autocast as in training."""
    import argparse
    import torch
    from pretrain.train import build_model
    checkpoint = Path(checkpoint).resolve()
    before = checkpoint.stat()
    # mmap avoids copying unused optimizer states into RAM.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    args = argparse.Namespace(**state["args"])
    args.gradient_checkpointing = False
    tokens = int(state["tokens_seen"])
    if not allow_incomplete and tokens < int(args.target_tokens):
        raise ValueError(f"Checkpoint incomplete: {tokens} < {args.target_tokens}: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("Mamba3 evaluation requires a CUDA compute node")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_model(args, torch.device("cpu"))
    model.load_state_dict(state["model"], strict=True)
    model.eval().to("cuda")
    identity = {"path": str(checkpoint), "sha256": sha256(checkpoint),
                "step": int(state["step"]), "tokens_seen": tokens,
                "target_tokens": int(args.target_tokens), "args": vars(args),
                "precision": "fp32_weights_bf16_autocast"}
    after = checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed during loading; retry after training finishes")
    del state
    return model, identity


def software_identity():
    import importlib.metadata
    import platform
    versions = {"python": platform.python_version()}
    for package in ("torch", "numpy", "triton", "tilelang", "tokenizers", "pyarrow", "rouge", "fuzzywuzzy", "python-Levenshtein", "quack-kernels", "nvidia-cutlass-dsl", "apache-tvm-ffi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    files = list((ROOT / "evaluate").glob("*.py"))
    files += list((ROOT / "mamba_ssm").rglob("*.py"))
    files += [ROOT / "pretrain/train.py"]
    return {"versions": versions, "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}}
