from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class DataPlan:
    train_files: tuple[str, ...]
    val_files: tuple[str, ...]
    test_files: tuple[str, ...]
    stored_train_tokens: int
    stored_val_tokens: int
    stored_test_tokens: int
    usable_train_tokens: int
    usable_val_tokens: int
    usable_test_tokens: int
    train_samples: int
    val_samples: int
    test_samples: int
    metadata: dict[str, Any]


class TensorBoardLogger:
    def __init__(self, log_dir, config, *, resume_step=None, resume_tokens=None):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard is enabled but unavailable; install the training extra "
                "or use --no-tensorboard."
            ) from error
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(
            str(self.log_dir),
            purge_step=resume_step + 1 if resume_step is not None else None,
            flush_secs=30,
        )
        self.token_writer = SummaryWriter(
            str(self.log_dir / "by_tokens"),
            purge_step=resume_tokens + 1 if resume_tokens is not None else None,
            flush_secs=30,
        )
        self.writer.add_text(
            "run/config", "```json\n" + json.dumps(config, indent=2) + "\n```", 0
        )

    def log(self, metrics, *, step):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, global_step=step)
            if "train/tokens_seen" in metrics:
                self.token_writer.add_scalar(
                    name, value, global_step=int(metrics["train/tokens_seen"])
                )

    def flush(self):
        self.writer.flush()
        self.token_writer.flush()

    def finish(self, *, state="success", error=None):
        self.writer.add_text("run/status", state + (f": {error}" if error else ""))
        self.writer.close()
        self.token_writer.close()


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default="")
    known, _ = bootstrap.parse_known_args()
    config = _load_json(known.config) if known.config else {}

    parser = argparse.ArgumentParser(
        description="Train a Mamba-1, Mamba-2, or Mamba-3 causal LM on flat token shards."
    )
    parser.add_argument("--config", type=str, default=known.config)

    # Data.
    parser.add_argument("--train-data", type=str, default="")
    parser.add_argument("--val-data", type=str, default="")
    parser.add_argument("--test-data", type=str, default="")
    parser.add_argument("--token-dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000)
    parser.add_argument(
        "--lr-schedule-tokens",
        type=int,
        default=0,
        help="Token budget for LR scheduling; 0 uses target_tokens.",
    )
    parser.add_argument("--num-workers", type=int, default=4)

    # Model.  d_model=768 and n_layer=16 produce 98.95M parameters for
    # Mamba-1 and 98.85M for Mamba-2 with the defaults below.
    parser.add_argument(
        "--ssm-layer",
        choices=("Mamba1", "Mamba2", "Mamba3"),
        default="Mamba2",
    )
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--pad-vocab-size-multiple", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-intermediate", type=int, default=0)
    parser.add_argument("--n-layer", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--ngroups", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--rope-fraction",
        type=float,
        choices=(0.0, 0.5),
        default=0.5,
        help="Fraction of Mamba-3 state values rotated; 0 disables RoPE.",
    )
    parser.add_argument(
        "--is-outproj-norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--is-mimo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument(
        "--fuse-pregate-headwise-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-add-norm", action=argparse.BooleanOptionalAction, default=True
    )

    # Optimization.  batch_size is per rank.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--loss-chunk-size", type=int, default=256)
    parser.add_argument(
        "--full-logits-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Materialize logits for every token in a micro-batch and use the "
              "standard cross-entropy path instead of chunked loss recomputation."),
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.06554)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)

    # Runtime and checkpointing.
    parser.add_argument("--output-dir", type=str, default="training_outputs/mamba2_98m")
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--test-eval-batches", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable only for short profiling/smoke jobs; production runs should keep it enabled.",
    )
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-log-dir", type=str, default="",
                        help=("Defaults to <output-dir-parent>/tensorboard/<run-name>; "
                              "writes local event files only."))
    parser.add_argument(
        "--pg19-manifest",
        type=str,
        default="",
        help="Fixed PG19 probe directory or manifest.json; empty disables periodic PG19.",
    )
    parser.add_argument(
        "--pg19-eval-every-tokens",
        type=int,
        default=0,
        help="Evaluate the fixed PG19 probe at this token cadence; 0 disables it.",
    )
    parser.add_argument("--dry-run", action="store_true")

    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown keys in {known.config}: {unknown_keys}")
    parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.test_data:
        args.test_data = args.val_data
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "sequence_length",
        "target_tokens",
        "vocab_size",
        "d_model",
        "n_layer",
        "d_state",
        "d_conv",
        "expand",
        "batch_size",
        "grad_accum_steps",
        "eval_batch_size",
        "loss_chunk_size",
        "pad_vocab_size_multiple",
        "learning_rate",
        "log_every",
        "eval_every",
        "save_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.d_intermediate < 0:
        raise ValueError("--d-intermediate must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    # argparse choices do not validate JSON defaults.
    if args.ssm_layer not in {"Mamba1", "Mamba2", "Mamba3"}:
        raise ValueError("ssm_layer must be Mamba1, Mamba2, or Mamba3")
    if args.rope_fraction not in (0.0, 0.5):
        raise ValueError("rope_fraction must be 0.0 or 0.5")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.eval_batches < 0 or args.test_eval_batches < 0:
        raise ValueError("--eval-batches and --test-eval-batches must be non-negative")
    if args.lr_schedule_tokens < 0:
        raise ValueError("--lr-schedule-tokens must be non-negative")
    if args.pg19_eval_every_tokens < 0:
        raise ValueError("--pg19-eval-every-tokens must be non-negative")
    if bool(args.pg19_manifest) != bool(args.pg19_eval_every_tokens):
        raise ValueError(
            "--pg19-manifest and --pg19-eval-every-tokens must be enabled together"
        )
    if args.ssm_layer in {"Mamba2", "Mamba3"}:
        if args.headdim <= 0 or args.ngroups <= 0 or args.chunk_size <= 0:
            raise ValueError("Mamba-2/3 headdim, ngroups, and chunk-size must be positive")
        if (args.d_model * args.expand) % args.headdim:
            raise ValueError("expand * d_model must be divisible by headdim")
        if (args.d_model * args.expand // args.headdim) % args.ngroups:
            raise ValueError("number of heads must be divisible by ngroups")
    if args.ssm_layer == "Mamba3" and args.mimo_rank <= 0:
        raise ValueError("--mimo-rank must be positive")
    if args.rope_fraction == 0.0 and args.ssm_layer != "Mamba3":
        raise ValueError("--rope-fraction 0 is currently supported only for Mamba3")
    if not args.train_data or not args.val_data:
        raise ValueError("--train-data and --val-data are required")


def resolve_files(specification: str) -> tuple[str, ...]:
    resolved: list[str] = []
    for piece in (part.strip() for part in specification.split(",")):
        if not piece:
            continue
        path = Path(piece).expanduser()
        if path.is_file():
            # Keep the logical symlink path so a zero-copy dataset view can
            # carry its own sibling meta.json. np.memmap follows the link when
            # opening the payload.
            resolved.append(str(path.absolute()))
        elif path.is_dir():
            resolved.extend(str(item.absolute()) for item in sorted(path.glob("*.bin")))
        else:
            resolved.extend(str(Path(item).absolute()) for item in sorted(glob.glob(piece)))
    files = tuple(dict.fromkeys(resolved))
    if not files:
        raise FileNotFoundError(f"No files matched {specification!r}")
    return files


def dtype_from_name(name: str) -> np.dtype:
    return np.dtype({"uint16": np.uint16, "uint32": np.uint32}[name])


def _stored_tokens(files: Sequence[str], dtype: np.dtype) -> int:
    total = 0
    for filename in files:
        size = os.path.getsize(filename)
        if size % dtype.itemsize:
            raise ValueError(
                f"{filename} has {size} bytes, not divisible by {dtype.itemsize}"
            )
        total += size // dtype.itemsize
    return int(total)


def _sample_count(files: Sequence[str], dtype: np.dtype, sequence_length: int) -> int:
    total = 0
    width = sequence_length + 1
    for filename in files:
        tokens = os.path.getsize(filename) // dtype.itemsize
        if tokens < width:
            raise ValueError(f"{filename} has only {tokens} tokens; need at least {width}")
        total += (tokens - width) // sequence_length + 1
    return int(total)


def _read_sibling_metadata(files: Sequence[str]) -> dict[str, Any]:
    parents = {str(Path(filename).parent) for filename in files}
    if len(parents) != 1:
        return {}
    meta_path = Path(next(iter(parents))) / "meta.json"
    return _load_json(meta_path) if meta_path.exists() else {}


def inspect_data(args: argparse.Namespace) -> DataPlan:
    train_files = resolve_files(args.train_data)
    val_files = resolve_files(args.val_data)
    test_files = resolve_files(args.test_data)
    dtype = dtype_from_name(args.token_dtype)
    stored_train = _stored_tokens(train_files, dtype)
    stored_val = _stored_tokens(val_files, dtype)
    stored_test = _stored_tokens(test_files, dtype)
    train_samples = _sample_count(train_files, dtype, args.sequence_length)
    val_samples = _sample_count(val_files, dtype, args.sequence_length)
    test_samples = _sample_count(test_files, dtype, args.sequence_length)
    metadata = _read_sibling_metadata(train_files)

    if metadata:
        meta_dtype = metadata.get("dtype")
        if meta_dtype and meta_dtype != args.token_dtype:
            raise ValueError(
                f"meta.json says dtype={meta_dtype}, but --token-dtype={args.token_dtype}"
            )
        meta_vocab = metadata.get("vocab_size")
        if meta_vocab and int(meta_vocab) != args.vocab_size:
            raise ValueError(
                f"meta.json says vocab_size={meta_vocab}, but --vocab-size={args.vocab_size}"
            )

    return DataPlan(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        stored_train_tokens=stored_train,
        stored_val_tokens=stored_val,
        stored_test_tokens=stored_test,
        usable_train_tokens=train_samples * args.sequence_length,
        usable_val_tokens=val_samples * args.sequence_length,
        usable_test_tokens=test_samples * args.sequence_length,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        metadata=metadata,
    )


class ShardedTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Fixed-length, non-overlapping causal-LM samples over flat token shards."""

    def __init__(
        self, files: Sequence[str], sequence_length: int, dtype_name: str
    ) -> None:
        super().__init__()
        self.files = tuple(files)
        self.sequence_length = int(sequence_length)
        self.dtype = dtype_from_name(dtype_name)
        self.sample_counts: list[int] = []
        cumulative: list[int] = []
        total = 0
        for filename in self.files:
            tokens = os.path.getsize(filename) // self.dtype.itemsize
            samples = (tokens - (self.sequence_length + 1)) // self.sequence_length + 1
            if samples <= 0:
                raise ValueError(f"{filename} is too short for the configured sequence length")
            self.sample_counts.append(int(samples))
            total += int(samples)
            cumulative.append(total)
        self.cumulative_samples = cumulative
        self._arrays: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.cumulative_samples[-1]

    def _array(self, shard_index: int) -> np.memmap:
        array = self._arrays.get(shard_index)
        if array is None:
            array = np.memmap(self.files[shard_index], dtype=self.dtype, mode="r")
            self._arrays[shard_index] = array
        return array

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.cumulative_samples, index)
        previous = 0 if shard == 0 else self.cumulative_samples[shard - 1]
        local_index = index - previous
        start = local_index * self.sequence_length
        tokens = np.asarray(
            self._array(shard)[start : start + self.sequence_length + 1],
            dtype=np.int64,
        )
        tensor = torch.from_numpy(tokens.copy())
        return tensor[:-1], tensor[1:]


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that can begin at a batch-aligned rank-local offset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return max(0, super().__len__() - self.start_index)


def initialize_distributed() -> DistributedContext:
    # torchrun exports WORLD_SIZE/RANK/LOCAL_RANK, while a native Slurm srun
    # launch exports the equivalent SLURM_* variables. Supporting both keeps
    # login-node tests simple and avoids wrapping every rank in another shell.
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; use --dry-run for a CPU-only check")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def padded_vocab_size(vocab_size: int, multiple: int) -> int:
    return int(math.ceil(vocab_size / multiple) * multiple)


def estimate_parameter_count(args: argparse.Namespace) -> int:
    """Analytical parameter count for this repository's Mamba parameterization."""

    d = args.d_model
    inner = args.expand * d
    vocab = padded_vocab_size(args.vocab_size, args.pad_vocab_size_multiple)
    embeddings_and_final_norm = vocab * d + d

    if args.ssm_layer == "Mamba1":
        dt_rank = math.ceil(d / 16)
        block = (
            d * (2 * inner)  # in_proj
            + inner * args.d_conv
            + inner  # conv bias
            + inner * (dt_rank + 2 * args.d_state)  # x_proj
            + dt_rank * inner
            + inner  # dt_proj bias
            + inner * args.d_state  # A_log
            + inner  # D
            + inner * d  # out_proj
            + d  # block RMSNorm
        )
    elif args.ssm_layer == "Mamba2":
        nheads = inner // args.headdim
        in_features = 2 * inner + 2 * args.ngroups * args.d_state + nheads
        conv_dim = inner + 2 * args.ngroups * args.d_state
        block = (
            d * in_features
            + conv_dim * args.d_conv
            + conv_dim  # conv bias
            + 3 * nheads  # dt_bias, A_log, D
            + inner  # gated RMSNorm
            + inner * d  # out_proj
            + d  # block RMSNorm
        )
    else:
        nheads = inner // args.headdim
        rank = args.mimo_rank if args.is_mimo else 1
        rotary_values = int(args.d_state * args.rope_fraction)
        rotary_values -= rotary_values % 2
        num_rope_angles = rotary_values // 2
        in_features = (
            2 * inner
            + 2 * args.d_state * args.ngroups * rank
            + 3 * nheads
            + num_rope_angles
        )
        block = (
            d * in_features  # in_proj
            + nheads  # dt_bias
            + 2 * nheads * rank * args.d_state  # B/C bias
            + 2 * args.d_state  # B/C RMSNorm
            + nheads  # D
            + (3 * inner * rank if args.is_mimo else 0)  # MIMO projections
            + (inner if args.is_outproj_norm else 0)  # optional gated RMSNorm
            + inner * d  # out_proj
            + d  # block RMSNorm
        )
    if args.d_intermediate:
        # GatedMLP rounds its hidden dimension to a multiple of 128 and has
        # two input projections, one output projection, and a second RMSNorm.
        mlp_hidden = math.ceil(args.d_intermediate / 128) * 128
        block += 3 * d * mlp_hidden + d
    return int(embeddings_and_final_norm + args.n_layer * block)


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    # Lazy imports let --dry-run work on login nodes without an active CUDA driver.
    if args.ssm_layer == "Mamba3":
        import triton
        from packaging.version import Version

        if device.type != "meta" and Version(triton.__version__) < Version("3.5.0"):
            raise RuntimeError(
                "Mamba3 requires triton>=3.5.0 in the training environment"
            )
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    ssm_cfg: dict[str, Any] = {
        "layer": args.ssm_layer,
        "d_state": args.d_state,
        "expand": args.expand,
    }
    if args.ssm_layer in {"Mamba1", "Mamba2"}:
        ssm_cfg["d_conv"] = args.d_conv
    if args.ssm_layer == "Mamba2":
        ssm_cfg.update(
            headdim=args.headdim,
            ngroups=args.ngroups,
            chunk_size=args.chunk_size,
        )
    elif args.ssm_layer == "Mamba3":
        ssm_cfg.update(
            headdim=args.headdim,
            ngroups=args.ngroups,
            chunk_size=args.chunk_size,
            rope_fraction=args.rope_fraction,
            is_outproj_norm=args.is_outproj_norm,
            is_mimo=args.is_mimo,
            mimo_rank=args.mimo_rank,
            fuse_pregate_headwise_norm=args.fuse_pregate_headwise_norm,
        )
    config = MambaConfig(
        d_model=args.d_model,
        d_intermediate=args.d_intermediate,
        n_layer=args.n_layer,
        vocab_size=args.vocab_size,
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=args.fused_add_norm,
        pad_vocab_size_multiple=args.pad_vocab_size_multiple,
        tie_embeddings=True,
    )
    # Training callers leave dtype=None to keep master parameters in fp32.
    # Inference adapters may request the same mixed layout as from_pretrained.
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    model.backbone.gradient_checkpointing = args.gradient_checkpointing
    return model


def unique_trainable_parameters(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return int(total)


def optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        skip_decay = (
            parameter.ndim < 2
            or name.endswith("bias")
            or "norm" in name.lower()
            or "embedding" in name.lower()
            or bool(getattr(parameter, "_no_weight_decay", False))
        )
        (no_decay if skip_decay else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def learning_rate_at_step(
    step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr_ratio: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = getattr(model, "module", getattr(model, "_orig_mod", model))
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    args: argparse.Namespace,
    runtime: dict[str, int],
) -> None:
    unwrapped = unwrap_model(model)
    model_uses_cuda = any(parameter.device.type == "cuda" for parameter in unwrapped.parameters())
    state = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "best_val_loss": float(best_val_loss),
        "args": vars(args),
        "runtime": runtime,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if model_uses_cuda else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: AdamW
) -> tuple[int, int, float, dict[str, int]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    model_uses_cuda = any(
        parameter.device.type == "cuda" for parameter in unwrap_model(model).parameters()
    )
    if cuda_rng_state_all is not None and model_uses_cuda:
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("tokens_seen", 0)),
        float(checkpoint.get("best_val_loss", math.inf)),
        dict(checkpoint.get("runtime", {})),
    )


def infinite_train_batches(
    loader: DataLoader[tuple[Tensor, Tensor]],
    sampler: ResumableDistributedSampler,
    start_micro_step: int,
    batches_per_epoch: int,
    batch_size: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    epoch = start_micro_step // batches_per_epoch
    batch_offset = start_micro_step % batches_per_epoch
    while True:
        sampler.set_epoch(epoch)
        sampler.start_index = batch_offset * batch_size
        yield from loader
        epoch += 1
        batch_offset = 0


def reduce_sum(values: Tensor) -> Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def causal_lm_loss(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    loss_chunk_size: int = 256,
    full_logits_loss: bool = False,
) -> Tensor:
    if full_logits_loss:
        logits = model(inputs).logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    # Keep the chunked loss inside forward so DDP sees the full differentiable graph.
    return model(inputs, targets=targets, loss_chunk_size=loss_chunk_size).loss


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    max_batches: int,
    loss_chunk_size: int = 256,
    full_logits_loss: bool = False,
) -> tuple[float, float]:
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = causal_lm_loss(
                model, inputs, targets, loss_chunk_size, full_logits_loss
            )
        token_count = targets.numel()
        totals[0] += loss.double() * token_count
        totals[1] += token_count
    totals = reduce_sum(totals)
    if totals[1].item() == 0:
        raise RuntimeError("Validation loader produced zero tokens")
    loss = float((totals[0] / totals[1]).item())
    model.train()
    return loss, math.exp(min(loss, 20.0))


def synchronize(context: DistributedContext) -> None:
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def resolve_tensorboard_log_dir(args: argparse.Namespace) -> Path:
    if args.tensorboard_log_dir:
        return Path(args.tensorboard_log_dir)
    output_dir = Path(args.output_dir)
    return output_dir.parent / "tensorboard" / output_dir.name


def print_plan(
    args: argparse.Namespace, data: DataPlan, world_size: int, estimated_params: int,
    pg19_probe: Any | None = None,
) -> None:
    tokens_per_step = (
        world_size
        * args.batch_size
        * args.grad_accum_steps
        * args.sequence_length
    )
    total_steps = math.ceil(args.target_tokens / tokens_per_step)
    schedule_tokens = args.lr_schedule_tokens or args.target_tokens
    schedule_steps = math.ceil(schedule_tokens / tokens_per_step)
    plan = {
        "ssm_layer": args.ssm_layer,
        "is_mimo": args.is_mimo,
        "rope_fraction": args.rope_fraction,
        "gradient_checkpointing": args.gradient_checkpointing,
        "loss_chunk_size": args.loss_chunk_size,
        "full_logits_loss": args.full_logits_loss,
        "eval_batch_size": args.eval_batch_size,
        "tensorboard_log_dir": str(resolve_tensorboard_log_dir(args)) if args.tensorboard else None,
        "fp32_adam_parameters_gradients_states_gib": estimated_params * 16 / 2**30,
        "estimated_parameters": estimated_params,
        "target_tokens": args.target_tokens,
        "tokens_per_parameter": args.target_tokens / estimated_params,
        "world_size": world_size,
        "per_gpu_batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum_steps,
        "sequence_length": args.sequence_length,
        "global_tokens_per_step": tokens_per_step,
        "optimizer_steps": total_steps,
        "lr_schedule_tokens": schedule_tokens,
        "lr_schedule_steps": schedule_steps,
        "processed_tokens_after_last_step": total_steps * tokens_per_step,
        "final_step_token_overshoot": total_steps * tokens_per_step - args.target_tokens,
        "stored_train_tokens": data.stored_train_tokens,
        "usable_train_tokens_per_epoch": data.usable_train_tokens,
        "approx_training_passes": args.target_tokens / data.usable_train_tokens,
        "stored_val_tokens": data.stored_val_tokens,
        "stored_test_tokens": data.stored_test_tokens,
        "train_shards": len(data.train_files),
        "val_shards": len(data.val_files),
        "test_shards": len(data.test_files),
        "pg19_manifest": args.pg19_manifest or None,
        "pg19_eval_every_tokens": args.pg19_eval_every_tokens or None,
        "pg19_context_lengths": list(pg19_probe.context_lengths) if pg19_probe is not None else [],
        "pg19_num_windows": pg19_probe.num_windows if pg19_probe is not None else 0,
        "pg19_prediction_length": pg19_probe.prediction_length if pg19_probe is not None else 0,
        "dataset_metadata": data.metadata,
    }
    print(json.dumps(plan, indent=2), flush=True)


def pg19_milestones(target_tokens: int, interval_tokens: int) -> tuple[int, ...]:
    """Return token milestones, always including the final training budget."""

    if interval_tokens <= 0:
        return ()
    milestones = list(range(interval_tokens, target_tokens + 1, interval_tokens))
    if not milestones or milestones[-1] != target_tokens:
        milestones.append(target_tokens)
    return tuple(milestones)


def pg19_summary_path(output_dir: Path, milestone_tokens: int) -> Path:
    return (
        output_dir
        / "pg19_fixed"
        / f"milestone_{milestone_tokens:012d}"
        / "summary.json"
    )


def pending_pg19_milestones(
    output_dir: Path,
    tokens_seen: int,
    target_tokens: int,
    interval_tokens: int,
    probe_manifest_sha256: str | None = None,
) -> tuple[int, ...]:
    reached = min(tokens_seen, target_tokens)

    def summary_is_complete(milestone: int) -> bool:
        path = pg19_summary_path(output_dir, milestone)
        if not path.is_file():
            return False
        try:
            value = _load_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        return (
            value.get("protocol") == "pg19_fixed"
            and value.get("milestone_tokens") == milestone
            and (
                probe_manifest_sha256 is None
                or value.get("probe_manifest_sha256") == probe_manifest_sha256
            )
            and isinstance(value.get("metrics"), dict)
        )

    return tuple(
        milestone
        for milestone in pg19_milestones(target_tokens, interval_tokens)
        if milestone <= reached and not summary_is_complete(milestone)
    )


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_pg19_milestones(
    model: nn.Module,
    probe: Any,
    milestones: Sequence[int],
    context: DistributedContext,
    output_dir: Path,
    metrics_path: Path,
    metric_logger: Any | None,
    completed_step: int,
    tokens_seen: int,
) -> float:
    """Evaluate reached fixed-probe milestones and persist each result atomically."""

    from pretrain.pg19 import evaluate_pg19_fixed_probe

    total_seconds = 0.0
    for milestone in milestones:
        if context.is_main:
            print(
                f"pg19_fixed start milestone={milestone:,} "
                f"step={completed_step} actual_tokens={tokens_seen:,}",
                flush=True,
            )
        metrics = evaluate_pg19_fixed_probe(
            model,
            probe,
            device=context.device,
            batch_size=1,
            autocast_dtype=torch.bfloat16,
            metric_prefix="pg19_fixed",
        )
        if context.device.type == "cuda":
            metrics["pg19_fixed/peak_allocated_gib"] = torch.cuda.max_memory_allocated(context.device) / 2**30
        total_seconds += float(metrics["pg19_fixed/eval_seconds"])
        if context.is_main:
            summary = {
                "protocol": "pg19_fixed",
                "note": "Periodic fixed-window probe; not canonical full-corpus PG19.",
                "synthetic": bool(probe.manifest.get("synthetic", False)),
                "milestone_tokens": int(milestone),
                "evaluated_at_step": int(completed_step),
                "evaluated_at_tokens_seen": int(tokens_seen),
                "checkpoint": str(output_dir / "last.pt"),
                "probe_manifest": str(probe.manifest_path),
                "probe_manifest_sha256": str(probe.manifest_sha256),
                "metrics": metrics,
            }
            local_record = {
                "kind": "pg19_fixed",
                "step": int(completed_step),
                "tokens_seen": int(tokens_seen),
                "milestone_tokens": int(milestone),
                **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(local_record, allow_nan=False) + "\n")
            ppls = " ".join(
                f"ppl_{context_length}="
                f"{float(metrics[f'pg19_fixed/perplexity_{context_length}']):.3f}"
                for context_length in probe.context_lengths
            )
            print(
                f"pg19_fixed done milestone={milestone:,} {ppls}",
                flush=True,
            )
            if metric_logger is not None:
                metric_logger.log(
                    {
                        **metrics,
                        "pg19_fixed/milestone_tokens": int(milestone),
                        "pg19_fixed/evaluated_at_tokens": int(tokens_seen),
                        "train/tokens_seen": int(tokens_seen),
                    },
                    step=completed_step,
                )
                metric_logger.flush()
            # This completion marker is deliberately last. If JSONL or TensorBoard
            # logging fails, resume repeats the point instead of silently losing it.
            atomic_write_json(pg19_summary_path(output_dir, milestone), summary)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    return total_seconds


def train(args: argparse.Namespace) -> None:
    data_plan = inspect_data(args)
    estimated_params = estimate_parameter_count(args)
    pg19_probe: Any | None = None
    if args.pg19_manifest:
        from pretrain.pg19 import load_fixed_pg19_probe

        pg19_probe = load_fixed_pg19_probe(args.pg19_manifest)
        if pg19_probe.tokenizer_length != args.vocab_size:
            raise ValueError(
                "PG19 probe tokenizer length does not match the model vocabulary: "
                f"{pg19_probe.tokenizer_length} != {args.vocab_size}"
            )
    dry_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.dry_run:
        print_plan(args, data_plan, dry_world_size, estimated_params, pg19_probe)
        return

    context = initialize_distributed()
    metric_logger: Any | None = None
    training_error: str | None = None
    try:
        seed_everything(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

        if context.is_main:
            print_plan(args, data_plan, context.world_size, estimated_params, pg19_probe)

        output_dir = Path(args.output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "args.json", "w", encoding="utf-8") as handle:
                json.dump(vars(args), handle, indent=2, sort_keys=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        metrics_path = output_dir / "metrics.jsonl"
        if pg19_probe is not None:
            if context.is_main:
                contexts = ",".join(str(value) for value in pg19_probe.context_lengths)
                print(
                    f"pg19_fixed manifest={pg19_probe.manifest_path} "
                    f"windows={pg19_probe.num_windows} contexts={contexts} "
                    f"prediction_tokens={pg19_probe.prediction_length}",
                    flush=True,
                )

        train_dataset = ShardedTokenDataset(
            data_plan.train_files, args.sequence_length, args.token_dtype
        )
        val_dataset = ShardedTokenDataset(
            data_plan.val_files, args.sequence_length, args.token_dtype
        )
        test_dataset = ShardedTokenDataset(
            data_plan.test_files, args.sequence_length, args.token_dtype
        )
        train_sampler = ResumableDistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
        rank_samples = len(train_sampler)
        batches_per_epoch = rank_samples // args.batch_size
        if batches_per_epoch <= 0:
            raise ValueError("Training data is smaller than one rank-local batch")

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
        }
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            drop_last=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            sampler=test_sampler,
            drop_last=False,
            **loader_kwargs,
        )

        model = build_model(args, context.device)
        actual_params = unique_trainable_parameters(model)
        if context.is_main:
            print(
                f"actual_parameters={actual_params:,} "
                f"tokens_per_parameter={args.target_tokens / actual_params:.3f}",
                flush=True,
            )
            if actual_params != estimated_params:
                print(
                    f"warning: analytical estimate was {estimated_params:,}; "
                    "using the actual count above",
                    flush=True,
                )

        if args.compile:
            model = torch.compile(model)
        if context.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        optimizer = AdamW(
            optimizer_groups(model, args.weight_decay),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=1e-8,
            fused=True,
        )

        tokens_per_step = (
            context.world_size
            * args.batch_size
            * args.grad_accum_steps
            * args.sequence_length
        )
        total_steps = math.ceil(args.target_tokens / tokens_per_step)
        schedule_tokens = args.lr_schedule_tokens or args.target_tokens
        schedule_steps = math.ceil(schedule_tokens / tokens_per_step)
        runtime = {
            "world_size": context.world_size,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "sequence_length": args.sequence_length,
            "tokens_per_step": tokens_per_step,
        }
        warmup_steps = int(round(schedule_steps * args.warmup_ratio))
        start_step = 0
        tokens_seen = 0
        best_val_loss = math.inf

        resume_path: Path | None = None
        if args.resume == "auto":
            candidate = output_dir / "last.pt"
            resume_path = candidate if candidate.exists() else None
        elif args.resume:
            resume_path = Path(args.resume)
        if resume_path is not None:
            start_step, tokens_seen, best_val_loss, saved_runtime = load_checkpoint(
                resume_path, model, optimizer
            )
            if saved_runtime and saved_runtime != runtime:
                raise ValueError(
                    "Cannot deterministically resume with a changed distributed/batch "
                    f"layout. Checkpoint runtime={saved_runtime}, current runtime={runtime}"
                )
            if context.is_main:
                print(
                    f"resumed={resume_path} step={start_step} tokens={tokens_seen:,}",
                    flush=True,
                )
        if context.is_main and args.tensorboard:
            metric_logger = TensorBoardLogger(
                resolve_tensorboard_log_dir(args),
                {**vars(args), "actual_parameters": actual_params, **runtime},
                resume_step=start_step if resume_path is not None else None,
                resume_tokens=tokens_seen if resume_path is not None else None,
            )
            print(f"tensorboard_log_dir={metric_logger.log_dir}", flush=True)
        resume_pg19_due = pending_pg19_milestones(
            output_dir,
            tokens_seen,
            args.target_tokens,
            args.pg19_eval_every_tokens,
            pg19_probe.manifest_sha256 if pg19_probe is not None else None,
        )
        if resume_pg19_due:
            assert pg19_probe is not None
            previous_step_tokens = max(0, tokens_seen - tokens_per_step)
            unavailable_history = tuple(
                milestone
                for milestone in resume_pg19_due
                if milestone <= previous_step_tokens
            )
            if unavailable_history:
                formatted = ", ".join(f"{value:,}" for value in unavailable_history)
                raise RuntimeError(
                    "Cannot reconstruct historical PG19 milestones from the current "
                    f"checkpoint (tokens_seen={tokens_seen:,}): {formatted}. Resume "
                    "with the original output directory/summaries or an earlier checkpoint."
                )
            model.train()
            run_pg19_milestones(
                model,
                pg19_probe,
                resume_pg19_due,
                context,
                output_dir,
                metrics_path,
                metric_logger,
                start_step,
                tokens_seen,
            )

        training_needed = start_step < total_steps
        if not training_needed:
            if context.is_main:
                print(
                    "Checkpoint already reached the requested token budget; "
                    "checking whether final evaluation is complete.",
                    flush=True,
                )

        train_iterator = infinite_train_batches(
            train_loader,
            train_sampler,
            start_micro_step=start_step * args.grad_accum_steps,
            batches_per_epoch=batches_per_epoch,
            batch_size=args.batch_size,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        interval_steps = 0
        synchronize(context)
        interval_start = time.monotonic()
        run_start = interval_start

        for step in range(start_step, total_steps):
            lr = learning_rate_at_step(
                step,
                schedule_steps,
                warmup_steps,
                args.learning_rate,
                args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            step_loss = 0.0
            for micro_step in range(args.grad_accum_steps):
                inputs, targets = next(train_iterator)
                inputs = inputs.to(context.device, non_blocking=True)
                targets = targets.to(context.device, non_blocking=True)
                should_sync = micro_step + 1 == args.grad_accum_steps
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel) and not should_sync
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = causal_lm_loss(
                            model,
                            inputs,
                            targets,
                            args.loss_chunk_size,
                            args.full_logits_loss,
                        )
                        scaled_loss = loss / args.grad_accum_steps
                    scaled_loss.backward()
                step_loss += float(loss.detach())

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            completed_step = step + 1
            tokens_seen = completed_step * tokens_per_step
            running_loss += step_loss / args.grad_accum_steps
            interval_steps += 1
            is_last = completed_step == total_steps
            pg19_due = pending_pg19_milestones(
                output_dir,
                tokens_seen,
                args.target_tokens,
                args.pg19_eval_every_tokens,
                pg19_probe.manifest_sha256 if pg19_probe is not None else None,
            )

            if completed_step % args.log_every == 0 or is_last or pg19_due:
                synchronize(context)
                elapsed = time.monotonic() - interval_start
                totals = torch.tensor(
                    [running_loss, interval_steps],
                    dtype=torch.float64,
                    device=context.device,
                )
                totals = reduce_sum(totals)
                mean_loss = float((totals[0] / totals[1]).item())
                interval_tokens = interval_steps * tokens_per_step
                tokens_per_second = interval_tokens / max(elapsed, 1e-9)
                remaining_tokens = max(0, args.target_tokens - tokens_seen)
                eta = remaining_tokens / max(tokens_per_second, 1e-9)
                if context.is_main:
                    print(
                        f"step={completed_step}/{total_steps} "
                        f"tokens={tokens_seen:,}/{args.target_tokens:,} "
                        f"loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20.0)):.2f} "
                        f"lr={lr:.3e} tok/s={tokens_per_second:,.0f} "
                        f"eta={format_duration(eta)}",
                        flush=True,
                    )
                    train_record = {
                        "train/loss": mean_loss,
                        "train/perplexity": math.exp(min(mean_loss, 20.0)),
                        "train/learning_rate": lr,
                        "train/tokens_per_second": tokens_per_second,
                        "train/tokens_seen": tokens_seen,
                    }
                    if context.device.type == "cuda":
                        peak_allocated_gib = torch.cuda.max_memory_allocated(context.device) / 2**30
                        peak_reserved_gib = torch.cuda.max_memory_reserved(context.device) / 2**30
                        train_record.update({
                            "memory/peak_allocated_gib": peak_allocated_gib,
                            "memory/peak_reserved_gib": peak_reserved_gib,
                        })
                        print(
                            f"memory step={completed_step} "
                            f"peak_allocated_gib={peak_allocated_gib:.3f} "
                            f"peak_reserved_gib={peak_reserved_gib:.3f}",
                            flush=True,
                        )
                    if metric_logger is not None:
                        metric_logger.log(train_record, step=completed_step)
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"kind": "train", "step": completed_step, **train_record}) + "\n")
                running_loss = 0.0
                interval_steps = 0
                interval_start = time.monotonic()

            # A zero batch limit explicitly disables the corresponding split.
            # This is important for long runs whose configs intentionally set
            # test_eval_batches=0; treating zero as "unlimited" would scan the
            # entire validation corpus at the end of training.
            should_eval = args.eval_batches > 0 and (
                completed_step % args.eval_every == 0 or is_last
            )
            should_save = args.save_checkpoints and (
                completed_step % args.save_every == 0 or is_last or bool(pg19_due)
            )
            improved = False
            val_loss = math.nan
            val_ppl = math.nan
            if should_eval:
                val_loss, val_ppl = evaluate(
                    model,
                    val_loader,
                    context.device,
                    args.eval_batches,
                    args.loss_chunk_size,
                    args.full_logits_loss,
                )
                improved = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)
                if context.is_main:
                    record = {
                        "step": completed_step,
                        "tokens_seen": tokens_seen,
                        "train_seconds": time.monotonic() - run_start,
                        "lr": lr,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "best_val_loss": best_val_loss,
                    }
                    with open(metrics_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(
                        f"validation step={completed_step} "
                        f"loss={val_loss:.4f} ppl={val_ppl:.2f}",
                        flush=True,
                    )
                    if metric_logger is not None:
                        eval_record = {
                            "eval/loss": val_loss,
                            "eval/perplexity": val_ppl,
                            "eval/best_loss": best_val_loss,
                            "train/learning_rate": lr,
                            "train/tokens_seen": tokens_seen,
                        }
                        metric_logger.log(eval_record, step=completed_step)

            if should_save or (args.save_checkpoints and improved):
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                if context.is_main:
                    save_checkpoint(
                        output_dir / "last.pt",
                        model,
                        optimizer,
                        completed_step,
                        tokens_seen,
                        best_val_loss,
                        args,
                        runtime,
                    )
                    if improved:
                        save_checkpoint(
                            output_dir / "best.pt",
                            model,
                            optimizer,
                            completed_step,
                            tokens_seen,
                            best_val_loss,
                            args,
                            runtime,
                        )
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            if pg19_due:
                assert pg19_probe is not None
                run_pg19_milestones(
                    model,
                    pg19_probe,
                    pg19_due,
                    context,
                    output_dir,
                    metrics_path,
                    metric_logger,
                    completed_step,
                    tokens_seen,
                )
                # PG19 is a deliberate pause. The forced train log above closed
                # the preceding throughput interval, so start a fresh timer.
                interval_start = time.monotonic()
        test_metrics_path = output_dir / "test_metrics.json"
        should_run_final_test = args.test_eval_batches > 0 and (
            training_needed or not test_metrics_path.is_file()
        )
        if should_run_final_test:
            test_loss, test_ppl = evaluate(
                model,
                test_loader,
                context.device,
                args.test_eval_batches,
                args.loss_chunk_size,
                args.full_logits_loss,
            )
            if context.is_main:
                record = {
                    "split": "test",
                    "step": total_steps,
                    "tokens_seen": tokens_seen,
                    "test_loss": test_loss,
                    "test_ppl": test_ppl,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"test step={total_steps} loss={test_loss:.4f} ppl={test_ppl:.2f}",
                    flush=True,
                )
                if metric_logger is not None:
                    metric_logger.log(
                        {
                            "test/loss": test_loss,
                            "test/perplexity": test_ppl,
                            "train/tokens_seen": tokens_seen,
                        },
                        step=total_steps,
                    )
                atomic_write_json(test_metrics_path, record)
        elif context.is_main and args.test_eval_batches > 0:
            print(f"final test already recorded in {test_metrics_path}", flush=True)
    except BaseException as error:
        training_error = str(error)
        raise
    finally:
        try:
            if metric_logger is not None:
                metric_logger.finish(
                    state="crashed" if training_error is not None else "success",
                    error=training_error,
                )
        finally:
            cleanup_distributed()


if __name__ == "__main__":
    train(parse_args())
