from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


FORMAT_NAME = "mamba3_pg19_fixed_probe"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedPG19Probe:
    """Validated manifest and memory-mapped fixed PG19 windows."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    tokens_path: Path
    tokens: np.memmap
    context_lengths: tuple[int, ...]
    prediction_length: int
    num_windows: int
    maximum_context_length: int
    tokenizer_length: int


def _resolve_tokens_path(root: Path, filename: str) -> Path:
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"probe tokens path escapes its cache directory: {filename}") from error
    return candidate


def _positive_int(value: Any, name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"probe manifest field {name!r} must be an integer") from error
    if converted < 1:
        raise ValueError(f"probe manifest field {name!r} must be positive")
    return converted


def load_fixed_pg19_probe(
    path: str | Path,
    *,
    verify_sha256: bool = True,
) -> FixedPG19Probe:
    """Load and validate a probe directory (or its ``manifest.json`` path)."""
    supplied_path = Path(path).expanduser().resolve()
    manifest_path = supplied_path if supplied_path.is_file() else supplied_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PG19 probe manifest does not exist: {manifest_path}")
    root = manifest_path.parent.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("PG19 probe manifest must contain a JSON object")
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(
            f"unsupported PG19 probe format {manifest.get('format')!r}; expected {FORMAT_NAME!r}"
        )
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"unsupported PG19 probe version {manifest.get('format_version')!r}"
        )

    protocol = manifest.get("protocol")
    storage = manifest.get("storage")
    tokenizer = manifest.get("tokenizer")
    windows = manifest.get("windows")
    if not isinstance(protocol, dict) or not isinstance(storage, dict):
        raise ValueError("probe manifest is missing protocol/storage objects")
    if not isinstance(tokenizer, dict) or not isinstance(windows, list):
        raise ValueError("probe manifest is missing tokenizer/windows records")

    raw_contexts = protocol.get("context_lengths")
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise ValueError("probe context_lengths must be a non-empty list")
    contexts = tuple(_positive_int(value, "context_lengths") for value in raw_contexts)
    if contexts != tuple(sorted(set(contexts))):
        raise ValueError("probe context lengths must be unique and strictly increasing")
    prediction_length = _positive_int(protocol.get("prediction_length"), "prediction_length")
    num_windows = _positive_int(protocol.get("num_windows"), "num_windows")
    maximum_context = _positive_int(
        protocol.get("maximum_context_length"), "maximum_context_length"
    )
    if maximum_context != contexts[-1]:
        raise ValueError("maximum_context_length does not match the largest context")
    if contexts[0] <= prediction_length:
        raise ValueError("every probe context must exceed prediction_length")
    if protocol.get("one_window_per_distinct_book") is not True:
        raise ValueError("probe must contain one window per distinct book")
    if protocol.get("shared_target_endpoints_across_context_lengths") is not True:
        raise ValueError("probe must use shared target endpoints across context lengths")

    tokenizer_length = _positive_int(tokenizer.get("length"), "tokenizer.length")
    if tokenizer_length != 128_256:
        raise ValueError(
            f"probe tokenizer has {tokenizer_length} entries; expected Mamba3 Llama-3.1's 128256"
        )
    if storage.get("logical_dtype") != "uint32":
        raise ValueError("PG19 Mamba3 probe storage must use uint32 token ids")
    numpy_dtype = np.dtype(storage.get("numpy_dtype"))
    if numpy_dtype.kind != "u" or numpy_dtype.itemsize != 4:
        raise ValueError(f"invalid probe numpy dtype: {numpy_dtype}")
    raw_shape = storage.get("shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != 2:
        raise ValueError("probe storage shape must have two dimensions")
    shape = tuple(int(value) for value in raw_shape)
    if shape != (num_windows, maximum_context):
        raise ValueError(
            f"probe storage shape {shape} does not match {(num_windows, maximum_context)}"
        )
    tokens_filename = storage.get("tokens_file")
    if not isinstance(tokens_filename, str) or not tokens_filename:
        raise ValueError("probe storage is missing tokens_file")
    tokens_path = _resolve_tokens_path(root, tokens_filename)
    if not tokens_path.is_file():
        raise FileNotFoundError(f"PG19 probe tokens do not exist: {tokens_path}")
    expected_bytes = num_windows * maximum_context * numpy_dtype.itemsize
    actual_bytes = tokens_path.stat().st_size
    if actual_bytes != expected_bytes or int(storage.get("size_bytes", -1)) != expected_bytes:
        raise ValueError(
            f"probe token file has {actual_bytes} bytes; expected exactly {expected_bytes}"
        )
    expected_hash = storage.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("probe storage is missing a valid sha256")
    if verify_sha256:
        actual_hash = sha256_file(tokens_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"probe token hash mismatch: manifest={expected_hash}, actual={actual_hash}"
            )

    if len(windows) != num_windows:
        raise ValueError(f"manifest has {len(windows)} windows; expected {num_windows}")
    cache_rows: set[int] = set()
    book_indices: set[int] = set()
    for position, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"window {position} is not an object")
        cache_row = int(window.get("cache_row", -1))
        book_index = int(window.get("book_index", -1))
        cache_rows.add(cache_row)
        book_indices.add(book_index)
        if int(window.get("book_window_end", -1)) - int(
            window.get("book_window_begin", -1)
        ) != maximum_context:
            raise ValueError(f"window {position} does not span maximum_context_length")
        if int(window.get("book_target_end", -1)) - int(
            window.get("book_target_begin", -1)
        ) != prediction_length:
            raise ValueError(f"window {position} has an invalid book target span")
        if int(window.get("cache_target_begin", -1)) != maximum_context - prediction_length:
            raise ValueError(f"window {position} has an invalid cache target begin")
        if int(window.get("cache_target_end", -1)) != maximum_context:
            raise ValueError(f"window {position} has an invalid cache target end")
    if cache_rows != set(range(num_windows)):
        raise ValueError("probe cache_row values must be a permutation of all rows")
    if len(book_indices) != num_windows or min(book_indices, default=-1) < 0:
        raise ValueError("probe windows must come from distinct, non-negative book indices")

    tokens = np.memmap(tokens_path, mode="r", dtype=numpy_dtype, shape=shape)
    actual_min = int(tokens.min())
    actual_max = int(tokens.max())
    if actual_min < 0 or actual_max >= tokenizer_length:
        raise ValueError(
            f"probe token range [{actual_min}, {actual_max}] exceeds tokenizer length {tokenizer_length}"
        )
    if int(storage.get("minimum_token_id", -1)) != actual_min:
        raise ValueError("probe minimum_token_id does not match token data")
    if int(storage.get("maximum_token_id", -1)) != actual_max:
        raise ValueError("probe maximum_token_id does not match token data")

    return FixedPG19Probe(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        tokens_path=tokens_path,
        tokens=tokens,
        context_lengths=contexts,
        prediction_length=prediction_length,
        num_windows=num_windows,
        maximum_context_length=maximum_context,
        tokenizer_length=tokenizer_length,
    )


def partition_probe_indices(num_windows: int, rank: int, world_size: int) -> tuple[int, ...]:
    """Deterministically assign every probe row to exactly one DDP rank."""
    if num_windows < 1:
        raise ValueError("num_windows must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    return tuple(range(rank, num_windows, world_size))


def _unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "module"):
            current = current.module  # type: ignore[assignment,union-attr]
        elif hasattr(current, "_orig_mod"):
            current = current._orig_mod  # type: ignore[assignment,union-attr]
        else:
            break
    return current


def _distributed_info(process_group: Any | None) -> tuple[int, int, bool]:
    initialized = dist.is_available() and dist.is_initialized()
    if not initialized:
        if process_group is not None:
            raise RuntimeError("a process_group was supplied but torch.distributed is not initialized")
        return 0, 1, False
    return (
        dist.get_rank(group=process_group),
        dist.get_world_size(group=process_group),
        True,
    )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_dtype(value: torch.dtype | str | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return choices[value]
    except KeyError as error:
        raise ValueError(f"unsupported PG19 autocast dtype: {value!r}") from error


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_logits(output: Any) -> Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output must expose a Tensor in .logits")
    return logits


def evaluate_pg19_fixed_probe(
    model: nn.Module,
    probe: FixedPG19Probe | str | Path,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 1,
    autocast_dtype: torch.dtype | str | None = torch.bfloat16,
    metric_prefix: str = "pg19_fixed",
    process_group: Any | None = None,
    verify_sha256: bool = True,
) -> dict[str, float | int]:
    """Evaluate a live model and return flat, TensorBoard-ready scalar metrics.

    Every process in ``process_group`` must call this function in the same
    context-length order.  DDP itself is bypassed for forward calls because
    each rank evaluates different examples; only explicit aggregate
    collectives are used.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    prefix = metric_prefix.rstrip("/")
    if not prefix:
        raise ValueError("metric_prefix must not be empty")
    fixed_probe = (
        probe
        if isinstance(probe, FixedPG19Probe)
        else load_fixed_pg19_probe(probe, verify_sha256=verify_sha256)
    )
    rank, world_size, distributed = _distributed_info(process_group)
    local_indices = partition_probe_indices(fixed_probe.num_windows, rank, world_size)
    raw_model = _unwrap_model(model)
    evaluation_device = torch.device(device) if device is not None else _model_device(raw_model)
    selected_autocast_dtype = _autocast_dtype(autocast_dtype)
    autocast_enabled = (
        evaluation_device.type == "cuda"
        and selected_autocast_dtype is not None
        and selected_autocast_dtype != torch.float32
    )
    was_training = model.training
    metrics: dict[str, float | int] = {}
    mean_nll_by_context: dict[int, float] = {}
    total_elapsed = 0.0

    if distributed:
        dist.barrier(group=process_group)
    model.eval()
    try:
        with torch.inference_mode():
            for context_length in fixed_probe.context_lengths:
                _synchronize_device(evaluation_device)
                started = time.perf_counter()
                totals = torch.zeros(3, dtype=torch.float64, device=evaluation_device)
                context_offset = fixed_probe.maximum_context_length - context_length
                for offset in range(0, len(local_indices), batch_size):
                    batch_indices = local_indices[offset : offset + batch_size]
                    input_array = np.array(
                        fixed_probe.tokens[list(batch_indices), context_offset:],
                        dtype=np.int64,
                        copy=True,
                    )
                    input_ids = torch.from_numpy(input_array).to(
                        device=evaluation_device,
                        non_blocking=evaluation_device.type == "cuda",
                    )
                    autocast_context = (
                        torch.autocast(
                            device_type="cuda",
                            dtype=selected_autocast_dtype,
                        )
                        if autocast_enabled
                        else nullcontext()
                    )
                    with autocast_context:
                        output = raw_model(
                            input_ids,
                            num_last_tokens=fixed_probe.prediction_length + 1,
                        )
                    logits = _model_logits(output)
                    expected_shape = (
                        len(batch_indices),
                        fixed_probe.prediction_length + 1,
                    )
                    if tuple(logits.shape[:2]) != expected_shape:
                        raise ValueError(
                            f"model returned logits shape {tuple(logits.shape)}; first two "
                            f"dimensions must be {expected_shape}"
                        )
                    targets = input_ids[:, -fixed_probe.prediction_length :]
                    if int(targets.max()) >= logits.shape[-1]:
                        raise ValueError(
                            f"probe target id {int(targets.max())} exceeds model logits "
                            f"vocabulary {logits.shape[-1]}"
                        )
                    prediction_logits = logits[:, :-1, :].float()
                    batch_nll = F.cross_entropy(
                        prediction_logits.reshape(-1, prediction_logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    if not bool(torch.isfinite(batch_nll)):
                        raise FloatingPointError(
                            f"non-finite PG19 NLL at context={context_length}, rank={rank}"
                        )
                    totals[0] += batch_nll.double()
                    totals[1] += targets.numel()
                    totals[2] += len(batch_indices)

                _synchronize_device(evaluation_device)
                elapsed = torch.tensor(
                    time.perf_counter() - started,
                    dtype=torch.float64,
                    device=evaluation_device,
                )
                if distributed:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=process_group)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=process_group)
                global_token_count = int(totals[1].item())
                global_window_count = int(totals[2].item())
                expected_tokens = fixed_probe.num_windows * fixed_probe.prediction_length
                if global_window_count != fixed_probe.num_windows:
                    raise RuntimeError(
                        f"DDP probe covered {global_window_count} windows; "
                        f"expected {fixed_probe.num_windows}"
                    )
                if global_token_count != expected_tokens:
                    raise RuntimeError(
                        f"DDP probe scored {global_token_count} tokens; expected {expected_tokens}"
                    )
                mean_nll = float(totals[0].item() / global_token_count)
                if not math.isfinite(mean_nll):
                    raise FloatingPointError(
                        f"non-finite aggregate PG19 NLL at context={context_length}"
                    )
                perplexity = math.exp(mean_nll)
                context_elapsed = float(elapsed.item())
                mean_nll_by_context[context_length] = mean_nll
                metrics[f"{prefix}/nll_{context_length}"] = mean_nll
                metrics[f"{prefix}/perplexity_{context_length}"] = perplexity
                metrics[f"{prefix}/scored_tokens_{context_length}"] = global_token_count
                metrics[f"{prefix}/windows_{context_length}"] = global_window_count
                metrics[f"{prefix}/seconds_{context_length}"] = context_elapsed
                total_elapsed += context_elapsed
    finally:
        model.train(was_training)

    base_context = fixed_probe.context_lengths[0]
    base_nll = mean_nll_by_context[base_context]
    for context_length in fixed_probe.context_lengths[1:]:
        metrics[f"{prefix}/delta_nll_{context_length}_vs_{base_context}"] = (
            mean_nll_by_context[context_length] - base_nll
        )
    metrics[f"{prefix}/eval_seconds"] = total_elapsed
    metrics[f"{prefix}/world_size"] = world_size
    if distributed:
        dist.barrier(group=process_group)
    return metrics


__all__ = [
    "FixedPG19Probe",
    "evaluate_pg19_fixed_probe",
    "load_fixed_pg19_probe",
    "partition_probe_indices",
]
