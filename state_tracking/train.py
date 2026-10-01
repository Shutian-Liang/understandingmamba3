from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from state_tracking.data import (
    TASK_SPECS,
    StateTrackingBatch,
    generate_batch,
    scaled_accuracy,
)


MAMBA3_ROTATION_CONTROLS = {
    "mamba3_fixed_rope",
    "mamba3_no_rope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the paper state-tracking models on three synthetic tasks."
        )
    )
    parser.add_argument(
        "--model",
        choices=(
            "mamba3_fixed_rope",
            "mamba3_no_rope",
            "mamba3_second_order",
            "mamba2",
        ),
        default="mamba3_no_rope",
    )
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--siso-backend", choices=("triton", "torch_reference"), default="triton")
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/eval/state_tracking")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=64,
        help="Per-forward GPU batch; gradients accumulate to --batch-size.",
    )
    parser.add_argument("--context-length", type=int, default=224)
    parser.add_argument("--curriculum-min-length", type=int, default=3)
    parser.add_argument("--curriculum-start-max", type=int, default=40)
    parser.add_argument("--curriculum-end-max", type=int, default=160)
    parser.add_argument(
        "--curriculum-ramp-start-step",
        type=int,
        default=0,
        help=(
            "Hold curriculum-start-max through this optimizer step, then "
            "begin the length ramp. Zero preserves the immediate ramp."
        ),
    )
    parser.add_argument(
        "--curriculum-ramp-steps",
        type=int,
        default=None,
        help=(
            "Steps used to increase curriculum-start-max to "
            "curriculum-end-max. The maximum length is held at the end value "
            "afterward. Defaults to all training steps."
        ),
    )
    parser.add_argument(
        "--long-sequence-mix-ratio",
        type=float,
        choices=(0.0,),
        default=0.0,
        help=(
            "Compatibility field in the paper recipes; must remain zero."
        ),
    )
    parser.add_argument("--d-model", type=int, choices=(32, 64), default=64)
    parser.add_argument(
        "--d-intermediate",
        type=int,
        default=0,
        help=(
            "SwiGLU intermediate width after each mixer block. Use 0 to "
            "disable the MLP; official Mamba-3 language-model configs use "
            "2 * d_model."
        ),
    )
    parser.add_argument(
        "--tie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Share the token embedding and LM-head weights. Official released "
            "Mamba-3 language-model configs enable this option."
        ),
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Defaults to 1 for parity/counting and 3 for arithmetic.",
    )
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument(
        "--rope-fraction",
        type=float,
        choices=(0.5, 1.0),
        default=0.5,
        help="Fraction of the Mamba-3 state dimensions receiving data-dependent RoPE.",
    )
    parser.add_argument(
        "--mimo-rank",
        type=int,
        choices=(1,),
        default=1,
        help=(
            "Compatibility field in the paper recipes; only rank 1 is supported."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "SSM kernel chunk size; defaults to 64."
        ),
    )
    parser.add_argument(
        "--transition-mode",
        choices=(
            "matched_underdamped",
            "critically_damped",
            "overdamped",
        ),
        default="matched_underdamped",
        help="Rotary-subspace dynamics for Mamba3SecondOrder.",
    )
    parser.add_argument(
        "--overdamped-rho-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier in rho=scale*a*tanh(nu) for the over-damped mode. "
            "Values below one keep the slow continuous-time root away from zero."
        ),
    )
    parser.add_argument(
        "--second-order-transition",
        choices=("matrix_exp", "closed_form"),
        default="closed_form",
        help="Sequential-oracle transition implementation for Mamba3SecondOrder.",
    )
    parser.add_argument(
        "--scan-backend",
        choices=(
            "sequential",
            "parallel",
            "triton",
            "triton_associative",
            "triton_legacy",
        ),
        default="triton",
        help="Training scan backend for Mamba3SecondOrder.",
    )
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=64,
        help="Local affine-scan chunk size for Mamba3SecondOrder.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument(
        "--lr-decay-horizon-steps",
        type=int,
        default=None,
        help=(
            "Virtual end step for the base cosine curve. Defaults to --steps. "
            "A value larger than --steps keeps a useful LR in a short run."
        ),
    )
    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=0,
        help=(
            "During the final N training steps, cosine-decay from the base "
            "curve's LR at the cooldown boundary to --min-lr."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "constant"),
        default="cosine",
        help="After warmup, either cosine-decay to --min-lr or keep --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Use 0 to disable clipping."
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--validation-samples", type=int, default=2_048)
    parser.add_argument("--test-samples", type=int, default=8_192)
    parser.add_argument(
        "--eval-lengths",
        default="224",
        help="Comma-separated exact sequence lengths. Paper score uses 224.",
    )
    parser.add_argument(
        "--precision", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow TF32 for FP32 CUDA matrix multiplications. Use "
            "--no-allow-tf32 for an IEEE-FP32 control run. The config launcher "
            "also sets TRITON_F32_DEFAULT=ieee so custom Triton dot products "
            "follow this choice."
        ),
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the module call to fuse operations around custom kernels.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or a checkpoint path.",
    )
    args = parser.parse_args()
    args.eval_lengths = tuple(
        int(item.strip()) for item in args.eval_lengths.split(",") if item.strip()
    )
    return args


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def safe_float(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolved_chunk_size(args: argparse.Namespace) -> int:
    configured = getattr(args, "chunk_size", None)
    return 64 if configured is None else int(configured)


def config_dict(args: argparse.Namespace, n_layers: int) -> dict[str, Any]:
    is_rotation_control = args.model in MAMBA3_ROTATION_CONTROLS
    is_second_order = args.model == "mamba3_second_order"
    is_mamba3 = is_rotation_control or is_second_order
    config = vars(args).copy()
    config["output_root"] = str(config["output_root"])
    config["eval_lengths"] = list(config["eval_lengths"])
    config["n_layers"] = n_layers
    config["primary_evaluation_distribution"] = "original_task"
    config["implementation"] = {
        "siso_backend": getattr(args, "siso_backend", "triton") if is_rotation_control else None,
        "model": {
            "mamba3_fixed_rope": "Mamba-3 SISO (standard fixed RoPE)",
            "mamba3_no_rope": "Mamba-3 SISO (without RoPE)",
            "mamba3_second_order": "Mamba-3 Second-Order",
            "mamba2": "Mamba-2",
        }[args.model],
        "state_size": args.d_state,
        "rope_fraction": args.rope_fraction if is_mamba3 else None,
        "rope_mode": {"mamba3_fixed_rope": "fixed", "mamba3_no_rope": "none"}.get(args.model),
        "mimo": False if is_rotation_control else None,
        "mimo_rank": 1 if is_rotation_control else None,
        "chunk_size": resolved_chunk_size(args),
        "transition_mode": args.transition_mode if is_second_order else None,
        "overdamped_rho_scale": args.overdamped_rho_scale if is_second_order else None,
        "second_order_transition": args.second_order_transition if is_second_order else None,
        "scan_backend": args.scan_backend if is_second_order else None,
        "scan_chunk_size": args.scan_chunk_size if is_second_order else None,
        "short_convolution": not is_mamba3,
        "mlp": args.d_intermediate > 0,
        "d_intermediate": args.d_intermediate,
        "tie_embeddings": args.tie_embeddings,
        "loss": "single cloze target / causal next-token",
    }
    return config


def run_directory(args: argparse.Namespace, n_layers: int) -> Path:
    lr_name = f"{args.lr:.8g}".replace(".", "p")
    variant = f"_di{args.d_intermediate}" if args.d_intermediate > 0 else ""
    if getattr(args, "siso_backend", "triton") != "triton":
        variant += "_torchref"
    if args.tie_embeddings:
        variant += "_tied"
    return (
        args.output_root / args.model / args.task
        / f"d{args.d_model}_l{n_layers}_lr{lr_name}_seed{args.seed}{variant}"
    )


def build_model(args: argparse.Namespace, n_layers: int) -> MambaLMHeadModel:
    if args.model not in MAMBA3_ROTATION_CONTROLS | {"mamba3_second_order", "mamba2"}:
        raise ValueError(f"Unsupported state-tracking model: {args.model}")
    spec = TASK_SPECS[args.task]
    mimo_rank = getattr(args, "mimo_rank", 1)
    if mimo_rank != 1:
        raise ValueError("The paper state-tracking models require mimo_rank=1")
    if getattr(args, "siso_backend", "triton") == "torch_reference":
        if args.model not in MAMBA3_ROTATION_CONTROLS or mimo_rank != 1:
            raise ValueError("torch_reference requires a standard Mamba-3 SISO model")
        if args.precision != "fp32" or getattr(args, "allow_tf32", False):
            raise ValueError("torch_reference diagnostic requires precision=fp32 and allow_tf32=False")
    d_inner = args.d_model * args.expand
    if d_inner % args.headdim != 0:
        raise ValueError(
            f"d_model * expand ({d_inner}) must be divisible by headdim "
            f"({args.headdim})"
        )
    if args.model in MAMBA3_ROTATION_CONTROLS:
        rope_mode = {
            "mamba3_fixed_rope": "fixed",
            "mamba3_no_rope": "none",
        }[args.model]
        ssm_cfg = {
            "layer": "Mamba3",
            "siso_backend": getattr(args, "siso_backend", "triton"),
            "d_state": args.d_state,
            "expand": args.expand,
            "headdim": args.headdim,
            "ngroups": 1,
            "rope_fraction": args.rope_fraction,
            "rope_mode": rope_mode,
            "chunk_size": resolved_chunk_size(args),
            "is_mimo": False,
            "mimo_rank": mimo_rank,
            "is_outproj_norm": False,
        }
    elif args.model == "mamba3_second_order":
        ssm_cfg = {
            "layer": "Mamba3SecondOrder",
            "d_state": args.d_state,
            "expand": args.expand,
            "headdim": args.headdim,
            "ngroups": 1,
            "rope_fraction": args.rope_fraction,
            "chunk_size": resolved_chunk_size(args),
            "is_mimo": False,
            "is_outproj_norm": False,
            "transition_mode": args.transition_mode,
            "overdamped_rho_scale": getattr(
                args, "overdamped_rho_scale", 1.0
            ),
            "second_order_transition": args.second_order_transition,
            "scan_backend": args.scan_backend,
            "scan_chunk_size": args.scan_chunk_size,
        }
    else:
        ssm_cfg = {
            "layer": "Mamba2",
            "d_state": args.d_state,
            "d_conv": 4,
            "expand": args.expand,
            "headdim": args.headdim,
            "ngroups": 1,
            "chunk_size": 64,
            "rmsnorm": True,
            # The optional causal_conv1d fused extension is not available in
            # every environment and its channel-last kernel rejects the small
            # synthetic model's projection stride. This path keeps the same
            # Mamba-2 operations (depthwise causal conv + SSD scan) with the
            # portable PyTorch convolution implementation.
            "use_mem_eff_path": False,
            "use_causal_conv1d": False,
        }
    config = MambaConfig(
        d_model=args.d_model,
        d_intermediate=args.d_intermediate,
        n_layer=n_layers,
        vocab_size=spec.vocab_size,
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=False,
        pad_vocab_size_multiple=1,
        tie_embeddings=args.tie_embeddings,
    )
    return MambaLMHeadModel(config)


def optimizer_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            getattr(parameter, "_no_weight_decay", False)
            or parameter.ndim < 2
            or name.endswith(".bias")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def learning_rate_at_step(args: argparse.Namespace, step: int) -> float:
    if args.warmup_steps > 0 and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    if args.scheduler == "constant":
        return args.lr

    decay_horizon = getattr(args, "lr_decay_horizon_steps", None) or args.steps
    decay_steps = max(1, decay_horizon - args.warmup_steps)
    progress = min(1.0, max(0.0, (step - args.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    base_lr = args.min_lr + (args.lr - args.min_lr) * cosine

    cooldown_steps = getattr(args, "cooldown_steps", 0)
    if cooldown_steps <= 0:
        return base_lr
    cooldown_start = args.steps - cooldown_steps
    if step <= cooldown_start:
        return base_lr

    boundary_progress = min(
        1.0,
        max(0.0, (cooldown_start - args.warmup_steps) / decay_steps),
    )
    boundary_cosine = 0.5 * (1.0 + math.cos(math.pi * boundary_progress))
    boundary_lr = args.min_lr + (args.lr - args.min_lr) * boundary_cosine
    cooldown_progress = min(
        1.0, max(0.0, (step - cooldown_start) / cooldown_steps)
    )
    cooldown_cosine = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
    return args.min_lr + (boundary_lr - args.min_lr) * cooldown_cosine


def curriculum_max_length(args: argparse.Namespace, step: int) -> int:
    ramp_start_step = int(getattr(args, "curriculum_ramp_start_step", 0))
    configured_ramp_steps = getattr(args, "curriculum_ramp_steps", None)
    if ramp_start_step <= 0:
        ramp_steps = (
            args.steps if configured_ramp_steps is None else configured_ramp_steps
        )
        # A shortened debug run should still reach the configured end length.
        ramp_steps = min(ramp_steps, args.steps)
        if ramp_steps <= 1:
            return args.curriculum_end_max
        progress = min(1.0, max(0.0, (step - 1) / (ramp_steps - 1)))
    else:
        if step <= ramp_start_step:
            return args.curriculum_start_max
        ramp_steps = (
            args.steps - ramp_start_step
            if configured_ramp_steps is None
            else configured_ramp_steps
        )
        ramp_steps = min(ramp_steps, args.steps - ramp_start_step)
        progress = min(
            1.0,
            max(0.0, (step - ramp_start_step) / max(1, ramp_steps)),
        )
    value = args.curriculum_start_max + progress * (
        args.curriculum_end_max - args.curriculum_start_max
    )
    return int(round(value))


def generate_training_batch(
    args: argparse.Namespace,
    *,
    batch_size: int,
    current_max_length: int,
    seed: int,
) -> StateTrackingBatch:
    """Sample fresh examples uniformly over the current curriculum lengths."""
    return generate_batch(
        args.task,
        batch_size=batch_size,
        min_sequence_length=args.curriculum_min_length,
        max_sequence_length=current_max_length,
        context_length=args.context_length,
        seed=seed,
    )


def precision_context(args: argparse.Namespace):
    if args.precision == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; use --precision fp16")
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def selected_logits_and_targets(model, batch, device, args):
    inputs = batch.input_ids.to(device=device, non_blocking=True)
    labels = batch.labels.to(device=device, non_blocking=True)
    active = labels.ne(-100)
    target_counts = active.sum(dim=1)
    if not torch.all(target_counts == 1):
        raise ValueError("every sequence must contain exactly one supervised answer")
    with precision_context(args):
        logits = model(inputs).logits[active]
        targets = labels[active]
        loss = F.cross_entropy(logits.float(), targets, reduction="none").mean()
    return logits, targets, loss, target_counts


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    device = torch.device(args.device)
    spec = TASK_SPECS[args.task]
    results: dict[str, dict[str, float | int]] = {}
    for length in args.eval_lengths:
        total_correct = 0
        total_target_tokens = 0
        target_histogram = torch.zeros(spec.vocab_size, dtype=torch.long)
        total_exact_sequences = 0
        total_loss = 0.0
        total_examples = 0
        batch_index = 0
        while total_examples < samples:
            current_batch_size = min(
                args.micro_batch_size, samples - total_examples
            )
            batch = generate_batch(
                args.task,
                batch_size=current_batch_size,
                min_sequence_length=length,
                max_sequence_length=length,
                context_length=args.context_length,
                seed=seed + length * 1_000_003 + batch_index,
            )
            logits, targets, loss, target_counts = selected_logits_and_targets(
                model, batch, device, args
            )
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets)
            total_correct += int(correct.sum().item())
            total_target_tokens += int(targets.numel())
            target_histogram += torch.bincount(
                targets.detach().cpu(), minlength=spec.vocab_size
            )
            offset = 0
            for count in target_counts.tolist():
                total_exact_sequences += int(correct[offset : offset + count].all().item())
                offset += count
            total_loss += float(loss.item()) * current_batch_size
            total_examples += current_batch_size
            batch_index += 1
        raw_accuracy = total_correct / total_target_tokens
        paper_scaled_accuracy = scaled_accuracy(
            raw_accuracy, spec.answer_classes
        )
        majority_baseline_accuracy = (
            int(target_histogram.max().item()) / total_target_tokens
        )
        results[str(length)] = {
            "loss": total_loss / total_examples,
            "raw_accuracy": raw_accuracy,
            "raw_accuracy_percent": 100.0 * raw_accuracy,
            "scaled_accuracy": paper_scaled_accuracy,
            "scaled_accuracy_percent": 100.0 * paper_scaled_accuracy,
            "majority_baseline_accuracy": majority_baseline_accuracy,
            "majority_baseline_accuracy_percent": 100.0
            * majority_baseline_accuracy,
            "raw_minus_majority_percent_points": 100.0
            * (raw_accuracy - majority_baseline_accuracy),
            "exact_sequence_accuracy": total_exact_sequences / total_examples,
            "samples": total_examples,
            "target_tokens": total_target_tokens,
        }
    return results


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_scaled_accuracy: float,
    best_step: int,
    best_validation: dict[str, Any] | None,
    last_validation: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "best_scaled_accuracy": best_scaled_accuracy,
        "best_step": best_step,
        "best_validation": best_validation,
        "last_validation": last_validation,
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def resolve_resume_checkpoint(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.resume == "none":
        return None
    if args.resume == "auto":
        path = run_dir / "checkpoints" / "latest.pt"
        return path if path.exists() else None
    path = Path(args.resume)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Mamba-3 Triton training requires a CUDA device")
    if args.d_intermediate < 0:
        raise ValueError("d_intermediate must be non-negative")
    if args.context_length < max(
        args.curriculum_end_max, max(args.eval_lengths, default=0)
    ):
        raise ValueError("context_length must cover curriculum and evaluation lengths")
    if args.micro_batch_size < 1 or args.micro_batch_size > args.batch_size:
        raise ValueError("micro_batch_size must be in [1, batch_size]")
    if args.curriculum_ramp_steps is not None and args.curriculum_ramp_steps < 1:
        raise ValueError("curriculum_ramp_steps must be positive")
    if not 0 <= args.curriculum_ramp_start_step < args.steps:
        raise ValueError("curriculum_ramp_start_step must be in [0, steps)")
    if (
        args.lr_decay_horizon_steps is not None
        and args.lr_decay_horizon_steps <= args.warmup_steps
    ):
        raise ValueError("lr_decay_horizon_steps must be greater than warmup_steps")
    if args.cooldown_steps < 0:
        raise ValueError("cooldown_steps must be non-negative")
    if args.cooldown_steps and args.scheduler != "cosine":
        raise ValueError("cooldown_steps requires the cosine scheduler")
    if args.cooldown_steps >= args.steps - args.warmup_steps:
        raise ValueError("cooldown_steps must start after warmup")
    if args.cooldown_steps:
        decay_horizon = args.lr_decay_horizon_steps or args.steps
        cooldown_start = args.steps - args.cooldown_steps
        if decay_horizon < cooldown_start:
            raise ValueError(
                "lr_decay_horizon_steps must reach the cooldown boundary"
            )
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    if args.scan_chunk_size < 1 or args.scan_chunk_size & (args.scan_chunk_size - 1):
        raise ValueError("--scan-chunk-size must be a positive power of two")
    chunk_size = resolved_chunk_size(args)
    if chunk_size < 1 or chunk_size & (chunk_size - 1):
        raise ValueError("--chunk-size must be a positive power of two")

    n_layers = args.n_layers or TASK_SPECS[args.task].default_layers
    run_dir = run_directory(args, n_layers)
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = config_dict(args, n_layers)
    atomic_json(run_dir / "config.json", config)
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    device = torch.device(args.device)
    model = build_model(args, n_layers).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args.weight_decay), lr=args.lr
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    resume_path = resolve_resume_checkpoint(args, run_dir)
    start_step = 0
    best_scaled_accuracy = -float("inf")
    best_step = 0
    best_validation: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_scaled_accuracy = float(checkpoint.get("best_scaled_accuracy", -float("inf")))
        best_step = int(checkpoint.get("best_step", 0))
        best_validation = checkpoint.get("best_validation")
        last_validation = checkpoint.get("last_validation")
        # A continuation launched into a new run directory from an explicitly
        # selected best checkpoint must retain that selection even if the
        # continuation never improves it.
        if resume_path.name == "best.pt" and not (checkpoint_dir / "best.pt").exists():
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                step=start_step,
                best_scaled_accuracy=best_scaled_accuracy,
                best_step=best_step,
                best_validation=best_validation,
                last_validation=last_validation,
                config=config,
            )
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    if start_step >= args.steps and (run_dir / "summary.json").exists():
        print(f"Run already complete: {run_dir}")
        return 0

    if args.compile_model:
        if not hasattr(model, "compile"):
            raise RuntimeError("--compile-model requires torch.nn.Module.compile()")
        model.compile(mode=args.compile_mode)

    print(
        f"model={args.model} task={args.task} device={device} "
        f"parameters={parameter_count:,} "
        f"steps={start_step}->{args.steps} compile={args.compile_model} "
        f"output={run_dir}",
        flush=True,
    )
    atomic_json(
        status_path,
        {
            "state": "running",
            "task": args.task,
            "step": start_step,
            "total_steps": args.steps,
            "pid": os.getpid(),
            "updated_at": time.time(),
            "run_dir": str(run_dir),
            "parameter_count": parameter_count,
        },
    )

    started_at = time.time()
    window_started = started_at
    window_loss = 0.0
    window_correct = 0
    window_targets = 0
    window_examples = 0
    try:
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            current_lr = learning_rate_at_step(args, step)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            current_max_length = curriculum_max_length(args, step)
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            step_correct = 0
            step_targets = 0
            step_examples = 0
            micro_index = 0
            while step_examples < args.batch_size:
                current_micro_batch = min(
                    args.micro_batch_size, args.batch_size - step_examples
                )
                batch = generate_training_batch(
                    args,
                    batch_size=current_micro_batch,
                    current_max_length=current_max_length,
                    seed=(
                        args.seed * 10_000_019
                        + step * 1_000_003
                        + micro_index
                    ),
                )
                logits, targets, loss, target_counts = selected_logits_and_targets(
                    model, batch, device, args
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at step {step}, micro-batch {micro_index}: "
                        f"{loss.item()}"
                    )
                loss_weight = current_micro_batch / args.batch_size
                (loss * loss_weight).backward()
                step_loss_sum += float(loss.item()) * current_micro_batch
                correct = logits.detach().argmax(dim=-1) == targets
                step_correct += int(correct.sum().item())
                step_targets += int(targets.numel())
                step_examples += current_micro_batch
                micro_index += 1
            current_grad_clip = args.grad_clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                current_grad_clip if current_grad_clip > 0 else float("inf"),
                error_if_nonfinite=True,
            )
            optimizer.step()

            window_loss += step_loss_sum
            window_correct += step_correct
            window_targets += step_targets
            window_examples += step_examples

            if step % args.log_every == 0 or step == 1:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = max(time.time() - window_started, 1e-9)
                train_raw_accuracy = window_correct / window_targets
                train_scaled_accuracy = scaled_accuracy(
                    train_raw_accuracy, TASK_SPECS[args.task].answer_classes
                )
                train_payload = {
                    "train/loss": window_loss / window_examples,
                    "train/scaled_accuracy": train_scaled_accuracy,
                    "train/scaled_accuracy_percent": 100.0
                    * train_scaled_accuracy,
                    "train/raw_accuracy": train_raw_accuracy,
                    "train/raw_accuracy_percent": 100.0 * train_raw_accuracy,
                    "train/lr": current_lr,
                    "train/curriculum_max_length": current_max_length,
                    "train/examples_per_second": window_examples / elapsed,
                    "train/grad_norm": safe_float(grad_norm) if grad_norm is not None else 0.0,
                    "train/grad_clip_threshold": current_grad_clip,
                }
                record = {"type": "train", "step": step, **train_payload}
                append_jsonl(metrics_path, record)
                print(
                    f"step {step:5d}/{args.steps} "
                    f"loss={train_payload['train/loss']:.4f} "
                    f"scaled={train_payload['train/scaled_accuracy_percent']:.2f}% "
                    f"raw={train_payload['train/raw_accuracy_percent']:.2f}% "
                    f"lr={current_lr:.3e} max_len={current_max_length} "
                    f"ex/s={train_payload['train/examples_per_second']:.1f}",
                    flush=True,
                )
                window_started = time.time()
                window_loss = 0.0
                window_correct = 0
                window_targets = 0
                window_examples = 0

            if step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(
                    model,
                    args,
                    samples=args.validation_samples,
                    seed=args.seed + 1_000_000_007,
                )
                primary = validation[str(args.eval_lengths[-1])]
                last_validation = validation
                append_jsonl(
                    metrics_path,
                    {
                        "type": "validation",
                        "step": step,
                        "results": validation,
                    },
                )
                print(
                    f"validation step={step} length={args.eval_lengths[-1]} "
                    f"scaled={primary['scaled_accuracy_percent']:.2f}% "
                    f"raw={primary['raw_accuracy_percent']:.2f}% "
                    f"majority={primary['majority_baseline_accuracy_percent']:.2f}% "
                    f"raw-majority={primary['raw_minus_majority_percent_points']:+.2f}pp",
                    flush=True,
                )
                if float(primary["scaled_accuracy"]) > best_scaled_accuracy:
                    best_scaled_accuracy = float(primary["scaled_accuracy"])
                    best_step = step
                    best_validation = validation
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        best_scaled_accuracy=best_scaled_accuracy,
                        best_step=best_step,
                        best_validation=best_validation,
                        last_validation=last_validation,
                        config=config,
                    )

            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_scaled_accuracy=best_scaled_accuracy,
                    best_step=best_step,
                    best_validation=best_validation,
                    last_validation=last_validation,
                    config=config,
                )

            if step % args.log_every == 0 or step == args.steps:
                atomic_json(
                    status_path,
                    {
                        "state": "running",
                        "task": args.task,
                        "step": step,
                        "total_steps": args.steps,
                        "pid": os.getpid(),
                        "updated_at": time.time(),
                        "run_dir": str(run_dir),
                        "parameter_count": parameter_count,
                    },
                )

        final_checkpoint = checkpoint_dir / "final.pt"
        save_checkpoint(
            final_checkpoint,
            model=model,
            optimizer=optimizer,
            step=args.steps,
            best_scaled_accuracy=best_scaled_accuracy,
            best_step=best_step,
            best_validation=best_validation,
            last_validation=last_validation,
            config=config,
        )
        # ``final.pt`` is an archival/resume checkpoint only.  The held-out
        # test set is evaluated exactly once, after restoring the checkpoint
        # selected by validation accuracy.
        best_checkpoint = checkpoint_dir / "best.pt"
        best_payload = torch.load(
            best_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(best_payload["model"])
        best_checkpoint_test = evaluate(
            model,
            args,
            samples=args.test_samples,
            seed=args.seed + 2_000_000_011,
        )
        primary_test = best_checkpoint_test[str(args.eval_lengths[-1])]
        summary = {
            "state": "complete",
            "task": args.task,
            "model": config["implementation"]["model"],
            "model_key": args.model,
            "seed": args.seed,
            "steps": args.steps,
            "parameter_count": parameter_count,
            "best_validation_scaled_accuracy": best_scaled_accuracy,
            "best_validation_step": best_step,
            "best_validation": best_validation,
            "final_validation": last_validation,
            "selection_rule": (
                "Maximum validation scaled accuracy at the primary length; "
                "the held-out test set is never used for hyperparameter selection."
            ),
            "test": best_checkpoint_test,
            "best_checkpoint_test": best_checkpoint_test,
            "primary_length": args.eval_lengths[-1],
            "primary_evaluation_distribution": config["primary_evaluation_distribution"],
            "primary_raw_accuracy": primary_test["raw_accuracy"],
            "primary_exact_sequence_accuracy": primary_test[
                "exact_sequence_accuracy"
            ],
            "primary_scaled_accuracy": primary_test["scaled_accuracy"],
            "primary_scaled_accuracy_percent": primary_test[
                "scaled_accuracy_percent"
            ],
            "primary_majority_baseline_accuracy": primary_test[
                "majority_baseline_accuracy"
            ],
            "primary_raw_minus_majority_percent_points": primary_test[
                "raw_minus_majority_percent_points"
            ],
            "elapsed_seconds": time.time() - started_at,
            "run_dir": str(run_dir),
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint": str(best_checkpoint),
        }
        atomic_json(run_dir / "summary.json", summary)
        append_jsonl(
            metrics_path,
            {
                "type": "test",
                "checkpoint": "best_validation",
                "step": best_step,
                "results": best_checkpoint_test,
            },
        )
        atomic_json(
            status_path,
            {
                "state": "complete",
                "task": args.task,
                "step": args.steps,
                "total_steps": args.steps,
                "pid": os.getpid(),
                "updated_at": time.time(),
                "run_dir": str(run_dir),
                "summary": str(run_dir / "summary.json"),
            },
        )
        print(
            f"complete model={args.model} task={args.task} "
            f"length={args.eval_lengths[-1]} "
            f"scaled={primary_test['scaled_accuracy_percent']:.2f}% "
            f"raw={primary_test['raw_accuracy_percent']:.2f}% "
            f"majority={primary_test['majority_baseline_accuracy_percent']:.2f}% "
            f"raw-majority={primary_test['raw_minus_majority_percent_points']:+.2f}pp "
            f"summary={run_dir / 'summary.json'}",
            flush=True,
        )
        return 0
    except BaseException as error:
        error_text = "".join(traceback.format_exception(error))
        atomic_json(
            status_path,
            {
                "state": "failed",
                "task": args.task,
                "step": locals().get("step", start_step),
                "total_steps": args.steps,
                "pid": os.getpid(),
                "updated_at": time.time(),
                "run_dir": str(run_dir),
                "error": str(error),
            },
        )
        (run_dir / "error.txt").write_text(error_text, encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
