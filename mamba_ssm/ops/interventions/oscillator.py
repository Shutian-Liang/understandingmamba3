"""Exact phase-increment permutations for Mamba-3 oscillator ablations."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
from typing import Any, Callable

import torch
import triton
import triton.language as tl

from mamba_ssm.ops.triton.mamba3.utils import tanh_approx


ANGLE_DT_MODULES = (
    "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
    "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",
)

SHARED_TOKEN_SHUFFLE = "shared_token_shuffle_direct"
INDEPENDENT_TOKEN_SHUFFLE = "independent_oscillator_token_shuffle"
WITHIN_TOKEN_OSCILLATOR_SHUFFLE = "within_token_oscillator_shuffle"
MODES = (
    SHARED_TOKEN_SHUFFLE,
    INDEPENDENT_TOKEN_SHUFFLE,
    WITHIN_TOKEN_OSCILLATOR_SHUFFLE,
)


@triton.jit
def _permuted_angle_dt_kernel(
    OUT,
    OUTPUT_STATE,
    ANGLE,
    DT,
    INIT_STATE,
    BASE_PERMUTATION,
    TOKEN_MULTIPLIER,
    TOKEN_OFFSET,
    stride_out_batch: tl.constexpr,
    stride_out_seq: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_dim: tl.constexpr,
    stride_output_state_batch: tl.constexpr,
    stride_output_state_head: tl.constexpr,
    stride_output_state_dim: tl.constexpr,
    stride_angle_batch: tl.constexpr,
    stride_angle_seq: tl.constexpr,
    stride_angle_head: tl.constexpr,
    stride_angle_dim: tl.constexpr,
    stride_dt_batch: tl.constexpr,
    stride_dt_head: tl.constexpr,
    stride_dt_seq: tl.constexpr,
    stride_init_batch: tl.constexpr,
    stride_init_head: tl.constexpr,
    stride_init_dim: tl.constexpr,
    stride_map_head: tl.constexpr,
    stride_map_dim: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    layer_key,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    RETURN_OUTPUT_STATE: tl.constexpr,
    WITHIN_TOKEN: tl.constexpr,
):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < dim

    angle_base = ANGLE + batch * stride_angle_batch + head * stride_angle_head
    dt_base = DT + batch * stride_dt_batch + head * stride_dt_head
    out_base = OUT + batch * stride_out_batch + head * stride_out_head

    if HAS_INIT_STATE:
        init_ptr = (
            INIT_STATE + batch * stride_init_batch + head * stride_init_head
            + dims * stride_init_dim
        )
        state = tl.load(init_ptr, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if not WITHIN_TOKEN:
        multiplier = tl.load(
            TOKEN_MULTIPLIER + head * stride_map_head + dims * stride_map_dim,
            mask=dim_mask, other=1,
        ).to(tl.int64)
        offset = tl.load(
            TOKEN_OFFSET + head * stride_map_head + dims * stride_map_dim,
            mask=dim_mask, other=0,
        ).to(tl.int64)

    nchunks = tl.cdiv(seqlen, CHUNK_SIZE)
    pi = 3.141592653589793
    two_pi = 2.0 * pi
    for chunk_index in range(nchunks):
        tokens = chunk_index * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
        token_mask = tokens < seqlen

        if WITHIN_TOKEN:
            # One independently keyed affine permutation of oscillator indices
            # for every (layer, head, token).  dim is 32 in the 1.5B models;
            # an odd multiplier is therefore bijective modulo dim.
            key = tokens.to(tl.uint32)
            key ^= (head * 1103515245 + layer_key).to(tl.uint32)
            key ^= key >> 16
            key *= 1664525
            key ^= key >> 15
            key *= 1013904223
            key ^= key >> 16
            channel_multiplier = 2 * (key % (dim // 2)) + 1
            channel_offset = ((key >> 8) ^ key) % dim
            source_dim = (
                channel_multiplier[:, None] * dims[None, :]
                + channel_offset[:, None]
            ) % dim
            source_token = tokens[:, None]
            dt_values = tl.load(
                dt_base + tokens * stride_dt_seq,
                mask=token_mask, other=0.0,
            ).to(tl.float32)[:, None]
        else:
            affine_index = (
                tokens[:, None].to(tl.int64) * multiplier[None, :]
                + offset[None, :]
            ) % seqlen
            source_token = tl.load(
                BASE_PERMUTATION + affine_index,
                mask=token_mask[:, None] & dim_mask[None, :], other=0,
            ).to(tl.int64)
            source_dim = dims[None, :]
            dt_values = tl.load(
                dt_base + source_token * stride_dt_seq,
                mask=token_mask[:, None] & dim_mask[None, :], other=0.0,
            ).to(tl.float32)

        angle_values = tl.load(
            angle_base
            + source_token * stride_angle_seq
            + source_dim * stride_angle_dim,
            mask=token_mask[:, None] & dim_mask[None, :], other=0.0,
        ).to(tl.float32)
        increments = tanh_approx(angle_values) * pi * dt_values
        chunk_cumsum = tl.cumsum(increments, axis=0)
        out_values = chunk_cumsum + state[None, :]
        out_values -= two_pi * tl.floor(out_values / two_pi)
        tl.store(
            out_base
            + tokens[:, None] * stride_out_seq
            + dims[None, :] * stride_out_dim,
            out_values,
            mask=token_mask[:, None] & dim_mask[None, :],
        )
        state += tl.sum(increments, axis=0)
        state -= two_pi * tl.floor(state / two_pi)

    if RETURN_OUTPUT_STATE:
        output_ptr = (
            OUTPUT_STATE + batch * stride_output_state_batch
            + head * stride_output_state_head + dims * stride_output_state_dim
        )
        tl.store(output_ptr, state, mask=dim_mask)


def _base_permutation(
    seed: int, length: int, layer: int, device: torch.device,
) -> torch.Tensor:
    # Intentionally identical to the established PhaseOrderMap seed scheme.
    digest = hashlib.sha256(
        f"mamba3-phase-shuffle:{seed}:{length}:{layer}".encode()
    ).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big"))
    return torch.randperm(length, generator=generator, dtype=torch.int64).to(device)


def _token_affine_maps(
    seed: int, length: int, layer: int, heads: int, dim: int,
    device: torch.device, shared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shared:
        return (
            torch.ones((heads, dim), dtype=torch.int64, device=device),
            torch.zeros((heads, dim), dtype=torch.int64, device=device),
        )
    if length % 10:
        raise ValueError(
            "independent token shuffles use multipliers coprime to the PG-19 lengths; "
            f"expected a multiple of ten, got {length}"
        )
    digest = hashlib.sha256(
        f"mamba3-independent-oscillator-shuffle:{seed}:{length}:{layer}".encode()
    ).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big"))
    count = heads * dim
    quotient = torch.randint(0, length // 10, (count,), generator=generator)
    residues = torch.tensor((1, 3, 7, 9), dtype=torch.int64)
    choices = torch.randint(0, 4, (count,), generator=generator)
    multiplier = 10 * quotient + residues[choices]
    offset = torch.randint(0, length, (count,), generator=generator)
    # The four residues are coprime to every configured length (2K/16K/64K).
    if any(math.gcd(int(value), length) != 1 for value in multiplier.tolist()):
        raise AssertionError("generated token map is not bijective")
    return (
        multiplier.reshape(heads, dim).to(device),
        offset.reshape(heads, dim).to(device),
    )


def permuted_angle_dt_fwd(
    angle: torch.Tensor,
    dt: torch.Tensor,
    *,
    mode: str,
    seed: int,
    layer: int,
    init_state: torch.Tensor | None = None,
    chunk_size: int = 64,
    return_output_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
):
    """Drop-in inference-only replacement for ``angle_dt_fwd``."""

    if mode not in MODES:
        raise ValueError(mode)
    if cu_seqlens is not None:
        raise NotImplementedError("oscillator shuffles do not support varlen batches")
    if angle.ndim != 4 or dt.ndim != 3:
        raise ValueError(f"unexpected phase tensors: angle={angle.shape}, dt={dt.shape}")
    batch, length, heads, dim = angle.shape
    if dt.shape != (batch, heads, length):
        raise ValueError(f"DT mismatch: {dt.shape}")
    if batch != 1:
        raise ValueError("oscillator shuffle evaluation requires batch size one")
    if dim & (dim - 1) and mode == WITHIN_TOKEN_OSCILLATOR_SHUFFLE:
        raise ValueError("within-token oscillator shuffle requires power-of-two angle dimension")
    if init_state is not None and init_state.shape != (batch, heads, dim):
        raise ValueError(f"initial angle state mismatch: {init_state.shape}")

    within = mode == WITHIN_TOKEN_OSCILLATOR_SHUFFLE
    base = (
        torch.empty(1, dtype=torch.int64, device=angle.device)
        if within else _base_permutation(seed, length, layer, angle.device)
    )
    multiplier, offset = _token_affine_maps(
        seed, length, layer, heads, dim, angle.device,
        shared=mode == SHARED_TOKEN_SHUFFLE,
    ) if not within else (
        torch.empty(1, dtype=torch.int64, device=angle.device),
        torch.empty(1, dtype=torch.int64, device=angle.device),
    )
    output = torch.empty_like(angle)
    final = torch.empty((batch, heads, dim), dtype=angle.dtype, device=angle.device)
    dummy_init = angle if init_state is None else init_state
    init_stride = (0, 0, 0) if init_state is None else init_state.stride()
    block_dim = triton.next_power_of_2(dim)
    layer_digest = hashlib.sha256(
        f"mamba3-within-token-oscillator-shuffle:{seed}:{length}:{layer}".encode()
    ).digest()
    layer_key = int.from_bytes(layer_digest[:4], "little") & 0x7FFFFFFF
    _permuted_angle_dt_kernel[(heads, batch)](
        output, final, angle, dt, dummy_init, base, multiplier, offset,
        *output.stride(), *final.stride(), *angle.stride(), *dt.stride(),
        *init_stride,
        multiplier.stride(0) if not within else 0,
        multiplier.stride(1) if not within else 0,
        length, dim, layer_key,
        CHUNK_SIZE=chunk_size,
        BLOCK_D=block_dim,
        HAS_INIT_STATE=init_state is not None,
        RETURN_OUTPUT_STATE=return_output_state,
        WITHIN_TOKEN=within,
        num_warps=4,
    )
    return (output, final) if return_output_state else output


class OscillatorPhaseMap(contextlib.AbstractContextManager):
    """Patch both Mamba-3 fused paths with one exact delta-phi permutation."""

    MODULES = ANGLE_DT_MODULES

    def __init__(
        self, expected_layers: int, mode: str, seed: int, sequence_length: int,
    ) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        self.expected_layers = int(expected_layers)
        self.mode = mode
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.modules: list[Any] = []
        self.originals: list[Callable[..., Any]] = []
        self.calls = 0

    def __enter__(self):
        for name in self.MODULES:
            module = importlib.import_module(name)
            original = module.angle_dt_fwd
            self.modules.append(module)
            self.originals.append(original)

            def wrapper(angle, dt, *args, **kwargs):
                layer = self.calls
                self.calls += 1
                if layer >= self.expected_layers:
                    raise RuntimeError("angle_dt_fwd called more than once per layer")
                if angle.shape[1] != self.sequence_length:
                    raise ValueError(
                        f"phase length {angle.shape[1]} != {self.sequence_length}"
                    )
                if args:
                    names = ("init_state", "chunk_size", "return_output_state", "cu_seqlens")
                    if len(args) > len(names):
                        raise TypeError("unexpected positional arguments to angle_dt_fwd")
                    kwargs = {**dict(zip(names, args)), **kwargs}
                return permuted_angle_dt_fwd(
                    angle, dt, mode=self.mode, seed=self.seed, layer=layer, **kwargs,
                )

            module.angle_dt_fwd = wrapper
        return self

    def __exit__(self, exc_type, exc, traceback):
        for module, original in zip(self.modules, self.originals):
            module.angle_dt_fwd = original
        if exc_type is None and self.calls != self.expected_layers:
            raise RuntimeError(
                f"expected {self.expected_layers} phase calls, saw {self.calls}"
            )
        return False
