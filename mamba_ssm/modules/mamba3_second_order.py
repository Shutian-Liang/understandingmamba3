# Copyright (c) 2026, Dao AI Lab, Goombalab.
"""Readable PyTorch reference transitions for Mamba-3 experiments.

This module deliberately does not use the fused Mamba-3 kernels.  It inherits
the parameterization of :class:`Mamba3` and can use either the original rotary
transition or a spectrum-matched canonical second-order transition.
"""

import math

from einops import rearrange
import torch
import torch.nn.functional as F

from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_second_order_scan import (
    triton_affine_scan_2x2,
    triton_affine_scan_scalar,
    triton_second_order_scan_mixed,
    triton_second_order_scan_mixed_associative,
)


def second_order_generator(a: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Construct [[0, 1], [-(a**2 + nu**2), -2*a]]."""
    a, nu = torch.broadcast_tensors(a, nu)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    return torch.stack(
        (
            torch.stack((zero, one), dim=-1),
            torch.stack((-(a.square() + nu.square()), -2.0 * a), dim=-1),
        ),
        dim=-2,
    )


def second_order_transition_matrix_exp(
    a: torch.Tensor, nu: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """Exact discrete second-order transition computed by ``matrix_exp``."""
    a, nu, dt = torch.broadcast_tensors(a, nu, dt)
    return torch.matrix_exp(second_order_generator(a, nu) * dt[..., None, None])


def second_order_transition_closed_form(
    a: torch.Tensor, nu: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """Exact transition in closed form, stable in the ``nu -> 0`` limit."""
    a, nu, dt = torch.broadcast_tensors(a, nu, dt)
    theta = nu * dt
    # torch.sinc(x) is sin(pi*x)/(pi*x), including its continuous value at 0.
    sin_over_nu = dt * torch.sinc(theta / math.pi)
    decay = torch.exp(-a * dt)
    cosine = torch.cos(theta)
    a_sine = a * sin_over_nu

    return decay[..., None, None] * torch.stack(
        (
            torch.stack((cosine + a_sine, sin_over_nu), dim=-1),
            torch.stack(
                (-(a.square() + nu.square()) * sin_over_nu, cosine - a_sine),
                dim=-1,
            ),
        ),
        dim=-2,
    )


def critically_damped_transition(
    a: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """Exact transition for the repeated-eigenvalue critical-damping limit."""
    a, dt = torch.broadcast_tensors(a, dt)
    decay = torch.exp(-a * dt)
    a_dt = a * dt
    return decay[..., None, None] * torch.stack(
        (
            torch.stack((1.0 + a_dt, dt), dim=-1),
            torch.stack((-a.square() * dt, 1.0 - a_dt), dim=-1),
        ),
        dim=-2,
    )


def overdamped_rate(
    a: torch.Tensor, nu: torch.Tensor, rho_scale: float = 1.0
) -> torch.Tensor:
    """Map the learned frequency channel to a stable real eigenvalue split.

    ``rho = rho_scale * a * tanh(nu)`` with ``0 < rho_scale <= 1`` guarantees
    ``abs(rho) <= rho_scale * a``. A scale below one therefore keeps both
    continuous-time eigenvalues ``-a +/- rho`` uniformly away from zero.
    """
    if not 0.0 < rho_scale <= 1.0:
        raise ValueError(f"rho_scale must be in (0, 1], got {rho_scale}")
    a, nu = torch.broadcast_tensors(a, nu)
    return rho_scale * a * torch.tanh(nu)


def overdamped_generator(
    a: torch.Tensor, nu: torch.Tensor, rho_scale: float = 1.0
) -> torch.Tensor:
    """Construct a stable over-damped generator with real negative spectrum."""
    a, nu = torch.broadcast_tensors(a, nu)
    rho = overdamped_rate(a, nu, rho_scale)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    return torch.stack(
        (
            torch.stack((zero, one), dim=-1),
            torch.stack((-(a.square() - rho.square()), -2.0 * a), dim=-1),
        ),
        dim=-2,
    )


def overdamped_transition_matrix_exp(
    a: torch.Tensor,
    nu: torch.Tensor,
    dt: torch.Tensor,
    rho_scale: float = 1.0,
) -> torch.Tensor:
    """Exact stable over-damped transition computed by ``matrix_exp``."""
    a, nu, dt = torch.broadcast_tensors(a, nu, dt)
    return torch.matrix_exp(
        overdamped_generator(a, nu, rho_scale) * dt[..., None, None]
    )


def overdamped_transition_closed_form(
    a: torch.Tensor,
    nu: torch.Tensor,
    dt: torch.Tensor,
    rho_scale: float = 1.0,
) -> torch.Tensor:
    """Exact over-damped transition, continuous at the critical limit.

    The direct ``exp(-a*dt) * cosh(rho*dt)`` expression can form ``inf * 0``.
    Computing the two negative real eigenmodes first avoids that overflow.
    """
    a, nu, dt = torch.broadcast_tensors(a, nu, dt)
    rho = overdamped_rate(a, nu, rho_scale)
    theta = rho * dt
    positive_mode = torch.exp((-a + rho) * dt)
    negative_mode = torch.exp((-a - rho) * dt)
    decay_cosh = 0.5 * (positive_mode + negative_mode)
    theta2 = theta.square()
    sinhc_series = 1.0 + theta2 / 6.0 + theta2.square() / 120.0
    sinh_series = theta * sinhc_series
    decay = torch.exp(-a * dt)
    decay_sinh = torch.where(
        theta.abs() < 0.1,
        decay * sinh_series,
        0.5 * (positive_mode - negative_mode),
    )
    rho_safe = torch.where(rho.abs() < 1.0e-12, torch.ones_like(rho), rho)
    decay_sinh_over_rho = torch.where(
        theta.abs() < 0.1,
        decay * dt * sinhc_series,
        decay_sinh / rho_safe,
    )
    stiffness = a.square() - rho.square()
    a_sinh = a * decay_sinh_over_rho
    return torch.stack(
        (
            torch.stack(
                (decay_cosh + a_sinh, decay_sinh_over_rho), dim=-1
            ),
            torch.stack(
                (
                    -stiffness * decay_sinh_over_rho,
                    decay_cosh - a_sinh,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )


def rotation_reference_transition(
    a: torch.Tensor, nu: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """Original Mamba-3 rotary transition in direct recurrent coordinates."""
    a, nu, dt = torch.broadcast_tensors(a, nu, dt)
    theta = nu * dt
    decay = torch.exp(-a * dt)
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    return decay[..., None, None] * torch.stack(
        (
            torch.stack((cosine, sine), dim=-1),
            torch.stack((-sine, cosine), dim=-1),
        ),
        dim=-2,
    )


def apply_2x2(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply ``matrix[..., K, 2, 2]`` to ``vector[..., P, K, 2]``."""
    m00 = matrix[..., 0, 0].unsqueeze(-2)
    m01 = matrix[..., 0, 1].unsqueeze(-2)
    m10 = matrix[..., 1, 0].unsqueeze(-2)
    m11 = matrix[..., 1, 1].unsqueeze(-2)
    v0, v1 = vector[..., 0], vector[..., 1]
    return torch.stack((m00 * v0 + m01 * v1, m10 * v0 + m11 * v1), dim=-1)


def multiply_2x2(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Explicitly compute ``left @ right`` for batches of 2x2 matrices."""
    l00, l01 = left[..., 0, 0], left[..., 0, 1]
    l10, l11 = left[..., 1, 0], left[..., 1, 1]
    r00, r01 = right[..., 0, 0], right[..., 0, 1]
    r10, r11 = right[..., 1, 0], right[..., 1, 1]
    return torch.stack(
        (
            torch.stack((l00 * r00 + l01 * r10, l00 * r01 + l01 * r11), dim=-1),
            torch.stack((l10 * r00 + l11 * r10, l10 * r01 + l11 * r11), dim=-1),
        ),
        dim=-2,
    )


def compose_affine_2x2(
    left_transition: torch.Tensor,
    left_bias: torch.Tensor,
    right_transition: torch.Tensor,
    right_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose an earlier ``left`` interval followed by a later ``right`` one.

    The result represents ``right(left(z))`` and is therefore
    ``(F_right @ F_left, F_right @ b_left + b_right)``.
    """
    transition = multiply_2x2(right_transition, left_transition)
    bias = apply_2x2(right_transition, left_bias) + right_bias
    return transition, bias


def compose_affine_scalar(
    left_transition: torch.Tensor,
    left_bias: torch.Tensor,
    right_transition: torch.Tensor,
    right_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar analogue of :func:`compose_affine_2x2`."""
    transition = right_transition * left_transition
    bias = right_transition[..., None, None] * left_bias + right_bias
    return transition, bias


def _inclusive_scan_2x2(
    transition: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele scan along dimension 1."""
    stride = 1
    length = transition.shape[1]
    while stride < length:
        combined_transition, combined_bias = compose_affine_2x2(
            transition[:, :-stride],
            bias[:, :-stride],
            transition[:, stride:],
            bias[:, stride:],
        )
        transition = torch.cat((transition[:, :stride], combined_transition), dim=1)
        bias = torch.cat((bias[:, :stride], combined_bias), dim=1)
        stride *= 2
    return transition, bias


def _inclusive_scan_scalar(
    transition: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele scalar-affine scan along dimension 1."""
    stride = 1
    length = transition.shape[1]
    while stride < length:
        combined_transition, combined_bias = compose_affine_scalar(
            transition[:, :-stride],
            bias[:, :-stride],
            transition[:, stride:],
            bias[:, stride:],
        )
        transition = torch.cat((transition[:, :stride], combined_transition), dim=1)
        bias = torch.cat((bias[:, :stride], combined_bias), dim=1)
        stride *= 2
    return transition, bias


def chunked_affine_scan_2x2(
    transition: torch.Tensor, bias: torch.Tensor, chunk_size: int = 64
) -> torch.Tensor:
    """Return every zero-initialized state of a chunked 2x2 affine scan.

    Args:
        transition: ``[B, L, H, K, 2, 2]``.
        bias: ``[B, L, H, P, K, 2]``.
        chunk_size: Maximum local scan length.  Local scan memory is bounded by
            ``O(L log(chunk_size))`` rather than ``O(L log(L))``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    batch, seqlen, nheads, pairs = transition.shape[:4]
    head_dim = bias.shape[3]
    padded_length = ((seqlen + chunk_size - 1) // chunk_size) * chunk_size
    pad = padded_length - seqlen
    if pad:
        identity = torch.eye(2, device=transition.device, dtype=transition.dtype)
        identity = identity.view(1, 1, 1, 1, 2, 2).expand(
            batch, pad, nheads, pairs, 2, 2
        )
        transition = torch.cat((transition, identity), dim=1)
        bias = torch.cat(
            (
                bias,
                bias.new_zeros(batch, pad, nheads, head_dim, pairs, 2),
            ),
            dim=1,
        )

    num_chunks = padded_length // chunk_size
    local_transition = transition.reshape(
        batch * num_chunks, chunk_size, nheads, pairs, 2, 2
    )
    local_bias = bias.reshape(
        batch * num_chunks, chunk_size, nheads, head_dim, pairs, 2
    )
    local_transition, local_bias = _inclusive_scan_2x2(local_transition, local_bias)
    local_transition = local_transition.reshape(
        batch, num_chunks, chunk_size, nheads, pairs, 2, 2
    )
    local_bias = local_bias.reshape(
        batch, num_chunks, chunk_size, nheads, head_dim, pairs, 2
    )

    chunk_transition = local_transition[:, :, -1]
    chunk_bias = local_bias[:, :, -1]
    _, chunk_prefix_bias = _inclusive_scan_2x2(chunk_transition, chunk_bias)
    chunk_initial = torch.cat(
        (torch.zeros_like(chunk_prefix_bias[:, :1]), chunk_prefix_bias[:, :-1]), dim=1
    )
    states = apply_2x2(local_transition, chunk_initial[:, :, None]) + local_bias
    return states.reshape(
        batch, padded_length, nheads, head_dim, pairs, 2
    )[:, :seqlen]


def chunked_affine_scan_scalar(
    transition: torch.Tensor, bias: torch.Tensor, chunk_size: int = 64
) -> torch.Tensor:
    """Return every zero-initialized state of a chunked scalar affine scan.

    Args:
        transition: ``[B, L, H]``.
        bias: ``[B, L, H, P, N]``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    batch, seqlen, nheads = transition.shape
    head_dim, d_state = bias.shape[-2:]
    padded_length = ((seqlen + chunk_size - 1) // chunk_size) * chunk_size
    pad = padded_length - seqlen
    if pad:
        transition = torch.cat(
            (transition, transition.new_ones(batch, pad, nheads)), dim=1
        )
        bias = torch.cat(
            (bias, bias.new_zeros(batch, pad, nheads, head_dim, d_state)), dim=1
        )

    num_chunks = padded_length // chunk_size
    local_transition = transition.reshape(batch * num_chunks, chunk_size, nheads)
    local_bias = bias.reshape(
        batch * num_chunks, chunk_size, nheads, head_dim, d_state
    )
    local_transition, local_bias = _inclusive_scan_scalar(local_transition, local_bias)
    local_transition = local_transition.reshape(
        batch, num_chunks, chunk_size, nheads
    )
    local_bias = local_bias.reshape(
        batch, num_chunks, chunk_size, nheads, head_dim, d_state
    )

    chunk_transition = local_transition[:, :, -1]
    chunk_bias = local_bias[:, :, -1]
    _, chunk_prefix_bias = _inclusive_scan_scalar(chunk_transition, chunk_bias)
    chunk_initial = torch.cat(
        (torch.zeros_like(chunk_prefix_bias[:, :1]), chunk_prefix_bias[:, :-1]), dim=1
    )
    states = (
        local_transition[..., None, None] * chunk_initial[:, :, None] + local_bias
    )
    return states.reshape(
        batch, padded_length, nheads, head_dim, d_state
    )[:, :seqlen]


def _rms_norm_torch(module, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    """Pure PyTorch equivalent of this repository's gated RMSNorm module."""
    input_dtype = x.dtype
    x_float = x.float()
    z_float = None if z is None else z.float()
    if z_float is not None and not module.norm_before_gate:
        x_float = x_float * F.silu(z_float)

    group_size = module.group_size or x_float.shape[-1]
    if x_float.shape[-1] % group_size != 0:
        raise ValueError(
            f"RMSNorm dimension {x_float.shape[-1]} is not divisible by group_size {group_size}"
        )
    grouped = x_float.reshape(*x_float.shape[:-1], -1, group_size)
    grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + module.eps)
    normalized = grouped.reshape_as(x_float) * module.weight.float()

    if z_float is not None and module.norm_before_gate:
        normalized = normalized * F.silu(z_float)
    return normalized.to(input_dtype)


class Mamba3SecondOrder(Mamba3):
    """Experimental SISO Mamba-3 with selectable reference dynamics.

    All learnable parameters and their state-dict keys come from ``Mamba3``.
    ``matched_underdamped`` changes only the rotary-subspace transition,
    ``critically_damped`` uses its repeated-real-eigenvalue limit,
    ``overdamped`` uses two stable real eigenvalues, and
    ``rotation_reference`` reconstructs the ordinary Mamba-3 rotation in the
    same direct-coordinate recurrence.
    """

    def __init__(
        self,
        *args,
        transition_mode: str = "matched_underdamped",
        second_order_transition: str = "matrix_exp",
        scan_backend: str = "sequential",
        scan_chunk_size: int | None = None,
        overdamped_rho_scale: float = 1.0,
        is_mimo: bool = False,
        **kwargs,
    ):
        if is_mimo:
            raise NotImplementedError(
                "Mamba3SecondOrder currently implements only the SISO reference path"
            )
        if transition_mode not in {
            "matched_underdamped",
            "critically_damped",
            "overdamped",
            "rotation_reference",
        }:
            raise ValueError(
                "transition_mode must be 'matched_underdamped', "
                "'critically_damped', 'overdamped', or 'rotation_reference', "
                f"got {transition_mode!r}"
            )
        if second_order_transition not in {"matrix_exp", "closed_form"}:
            raise ValueError(
                "second_order_transition must be 'matrix_exp' or 'closed_form', "
                f"got {second_order_transition!r}"
            )
        if not 0.0 < overdamped_rho_scale <= 1.0:
            raise ValueError(
                "overdamped_rho_scale must be in (0, 1], "
                f"got {overdamped_rho_scale}"
            )
        if scan_backend not in {
            "sequential",
            "parallel",
            "triton",
            "triton_associative",
            "triton_legacy",
        }:
            raise ValueError(
                "scan_backend must be 'sequential', 'parallel', 'triton', "
                "'triton_associative', or 'triton_legacy', "
                f"got {scan_backend!r}"
            )
        super().__init__(*args, is_mimo=False, **kwargs)
        # These are implementation selectors, not parameters or persistent buffers.
        self.transition_mode = transition_mode
        self.second_order_transition = second_order_transition
        self.scan_backend = scan_backend
        self.scan_chunk_size = self.chunk_size if scan_chunk_size is None else scan_chunk_size
        self.overdamped_rho_scale = overdamped_rho_scale
        if self.scan_chunk_size <= 0:
            raise ValueError(
                f"scan_chunk_size must be positive, got {self.scan_chunk_size}"
            )

        if self.nheads % self.num_bc_heads != 0:
            raise ValueError(
                f"nheads ({self.nheads}) must be divisible by ngroups ({self.num_bc_heads})"
            )

    def _rotary_transition(
        self, a: torch.Tensor, nu: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        if self.transition_mode == "rotation_reference":
            return rotation_reference_transition(a, nu, dt)
        if self.transition_mode == "critically_damped":
            expanded_a, _, expanded_dt = torch.broadcast_tensors(a, nu, dt)
            if self.second_order_transition == "matrix_exp":
                return second_order_transition_matrix_exp(
                    expanded_a, torch.zeros_like(nu), expanded_dt
                )
            return critically_damped_transition(expanded_a, expanded_dt)
        if self.transition_mode == "overdamped":
            if self.second_order_transition == "matrix_exp":
                return overdamped_transition_matrix_exp(
                    a, nu, dt, self.overdamped_rho_scale
                )
            return overdamped_transition_closed_form(
                a, nu, dt, self.overdamped_rho_scale
            )
        if self.second_order_transition == "matrix_exp":
            return second_order_transition_matrix_exp(a, nu, dt)
        return second_order_transition_closed_form(a, nu, dt)

    def _prepare_inputs(self, u: torch.Tensor):
        """Compute every token-local projection and input write in parallel."""
        projected = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads,
                self.d_state * self.num_bc_heads,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)

        # Match the ordinary Mamba-3 parameterizations exactly.
        neg_a = -heavy_tail_activation(dd_A.float())
        neg_a = torch.clamp(neg_a, max=-self.A_floor)
        a = -neg_a
        dt = F.softplus(dd_dt + self.dt_bias)
        lam = torch.sigmoid(trap.float())
        nu = math.pi * torch.tanh(angles.float())
        nu = nu.unsqueeze(-2).expand(-1, -1, self.nheads, -1)

        # B/C normalization is over d_state before grouped heads are expanded.
        B = _rms_norm_torch(self.B_norm, B).squeeze(2)
        C = _rms_norm_torch(self.C_norm, C).squeeze(2)
        heads_per_group = self.nheads // self.num_bc_heads
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)
        B = B.float() + self.B_bias[:, 0, :].float()[None, None, :, :]
        C = C.float() + self.C_bias[:, 0, :].float()[None, None, :, :]
        input_writes = x.float().unsqueeze(-1) * B.unsqueeze(-2)
        return z, x, C, neg_a, a, dt, lam, nu, input_writes

    def _forward_sequential(
        self,
        x: torch.Tensor,
        neg_a: torch.Tensor,
        a: torch.Tensor,
        dt: torch.Tensor,
        lam: torch.Tensor,
        nu: torch.Tensor,
        input_writes: torch.Tensor,
    ) -> torch.Tensor:
        """Original Python-loop recurrence retained as the correctness oracle."""
        batch, seqlen = x.shape[:2]

        # Direct-coordinate state: (batch, heads, head_dim, d_state).
        state = torch.zeros(
            batch,
            self.nheads,
            self.headdim,
            self.d_state,
            device=x.device,
            dtype=torch.float32,
        )
        previous_input = torch.zeros_like(state)
        rotary_size = self.split_tensor_size
        pair_count = rotary_size // 2
        states = []

        for token_idx in range(seqlen):
            current_input = input_writes[:, token_idx]
            dt_t = dt[:, token_idx].float()
            a_t = a[:, token_idx].float()
            lam_t = lam[:, token_idx].float()

            state_rot = state[..., :rotary_size].reshape(
                batch, self.nheads, self.headdim, pair_count, 2
            )
            previous_rot = previous_input[..., :rotary_size].reshape_as(state_rot)
            current_rot = current_input[..., :rotary_size].reshape_as(state_rot)

            transition = self._rotary_transition(
                a_t.unsqueeze(-1), nu[:, token_idx], dt_t.unsqueeze(-1)
            )
            previous_weight = ((1.0 - lam_t) * dt_t)[..., None, None, None]
            current_weight = (lam_t * dt_t)[..., None, None, None]
            next_rot = torch.einsum(
                "bhkij,bhpkj->bhpki",
                transition,
                state_rot + previous_weight * previous_rot,
            )
            next_rot = next_rot + current_weight * current_rot
            next_rot = next_rot.reshape(batch, self.nheads, self.headdim, rotary_size)

            if rotary_size < self.d_state:
                alpha = torch.exp(neg_a[:, token_idx].float() * dt_t)
                alpha = alpha[..., None, None]
                previous_scalar_weight = ((1.0 - lam_t) * dt_t)[..., None, None]
                current_scalar_weight = (lam_t * dt_t)[..., None, None]
                next_scalar = alpha * (
                    state[..., rotary_size:]
                    + previous_scalar_weight * previous_input[..., rotary_size:]
                )
                next_scalar = (
                    next_scalar + current_scalar_weight * current_input[..., rotary_size:]
                )
                state = torch.cat((next_rot, next_scalar), dim=-1)
            else:
                state = next_rot

            states.append(state)
            previous_input = current_input

        return torch.stack(states, dim=1)

    def _parallel_rotary_transition(
        self, a: torch.Tensor, nu: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized transition used by the accelerated scan path."""
        if self.transition_mode == "rotation_reference":
            return rotation_reference_transition(a, nu, dt)
        if self.transition_mode == "critically_damped":
            expanded_a, _, expanded_dt = torch.broadcast_tensors(a, nu, dt)
            return critically_damped_transition(expanded_a, expanded_dt)
        if self.transition_mode == "overdamped":
            return overdamped_transition_closed_form(
                a, nu, dt, self.overdamped_rho_scale
            )
        # The accelerated matched-underdamped path always uses the validated
        # stable closed form. matrix_exp remains available to the oracle.
        return second_order_transition_closed_form(a, nu, dt)

    def _forward_parallel(
        self,
        neg_a: torch.Tensor,
        a: torch.Tensor,
        dt: torch.Tensor,
        lam: torch.Tensor,
        nu: torch.Tensor,
        input_writes: torch.Tensor,
        use_triton: bool = False,
        use_triton_associative: bool = False,
        use_legacy_triton: bool = False,
    ) -> torch.Tensor:
        """Vectorized preprocessing plus chunked associative affine scans."""
        batch, seqlen = input_writes.shape[:2]
        rotary_size = self.split_tensor_size
        pair_count = rotary_size // 2

        if use_triton or use_triton_associative:
            paired_writes = input_writes.reshape(
                batch,
                seqlen,
                self.nheads,
                self.headdim,
                self.d_state // 2,
                2,
            )
            scan_fn = (
                triton_second_order_scan_mixed_associative
                if use_triton_associative
                else triton_second_order_scan_mixed
            )
            return scan_fn(
                a.float(),
                nu.float(),
                dt.float(),
                lam.float(),
                paired_writes,
                self.transition_mode,
                self.overdamped_rho_scale,
            ).reshape(batch, seqlen, self.nheads, self.headdim, self.d_state)

        previous_writes = torch.cat(
            (torch.zeros_like(input_writes[:, :1]), input_writes[:, :-1]), dim=1
        )
        previous_weight = ((1.0 - lam) * dt)[..., None, None, None]
        current_weight = (lam * dt)[..., None, None, None]

        transition = self._parallel_rotary_transition(
            a.unsqueeze(-1), nu, dt.unsqueeze(-1)
        )
        current_rotary = input_writes[..., :rotary_size].reshape(
            batch, seqlen, self.nheads, self.headdim, pair_count, 2
        )
        previous_rotary = previous_writes[..., :rotary_size].reshape_as(
            current_rotary
        )
        affine_bias = apply_2x2(
            transition, previous_weight * previous_rotary
        ) + current_weight * current_rotary
        rotary_states = (
            triton_affine_scan_2x2(transition, affine_bias)
            if use_legacy_triton
            else chunked_affine_scan_2x2(
                transition, affine_bias, chunk_size=self.scan_chunk_size
            )
        ).reshape(
            batch, seqlen, self.nheads, self.headdim, rotary_size
        )

        if rotary_size == self.d_state:
            return rotary_states

        alpha = torch.exp(neg_a.float() * dt.float())
        previous_scalar = previous_writes[..., rotary_size:]
        current_scalar = input_writes[..., rotary_size:]
        scalar_bias = (
            alpha[..., None, None]
            * ((1.0 - lam) * dt)[..., None, None]
            * previous_scalar
            + (lam * dt)[..., None, None] * current_scalar
        )
        scalar_states = (
            triton_affine_scan_scalar(alpha, scalar_bias)
            if use_legacy_triton
            else chunked_affine_scan_scalar(
                alpha, scalar_bias, chunk_size=self.scan_chunk_size
            )
        )
        return torch.cat((rotary_states, scalar_states), dim=-1)

    def _readout(
        self,
        states: torch.Tensor,
        C: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Shared vectorized C readout, D skip, gate/norm, and projection."""
        y = torch.einsum("blhpn,blhn->blhp", states, C)
        y = y + self.D.float()[None, None, :, None] * x.float()
        if not self.is_outproj_norm:
            y = y * F.silu(z.float())
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            z_flat = rearrange(z, "b l h p -> b l (h p)")
            y = _rms_norm_torch(self.norm, y, z_flat)
        return self.out_proj(y.to(x.dtype))

    def forward(
        self,
        u,
        seq_idx=None,
        cu_seqlens=None,
        inference_params=None,
        scan_backend: str | None = None,
        return_states: bool = False,
    ):
        """Run either the sequential oracle or parallel associative scan."""
        if inference_params is not None:
            raise NotImplementedError(
                "Mamba3SecondOrder does not yet implement cached inference"
            )
        if seq_idx is not None or cu_seqlens is not None:
            raise NotImplementedError(
                "Mamba3SecondOrder does not yet implement packed variable-length sequences"
            )
        if u.ndim != 3:
            raise ValueError(f"Expected input shape (batch, seqlen, d_model), got {tuple(u.shape)}")
        if u.shape[1] == 0:
            raise ValueError("Mamba3SecondOrder requires a non-empty sequence")
        backend = self.scan_backend if scan_backend is None else scan_backend
        if backend not in {
            "sequential",
            "parallel",
            "triton",
            "triton_associative",
            "triton_legacy",
        }:
            raise ValueError(
                "scan_backend must be 'sequential', 'parallel', 'triton', "
                "'triton_associative', or 'triton_legacy', "
                f"got {backend!r}"
            )
        if backend in {"triton", "triton_associative", "triton_legacy"} and not u.is_cuda:
            raise ValueError(f"scan_backend={backend!r} requires CUDA input")

        z, x, C, neg_a, a, dt, lam, nu, input_writes = self._prepare_inputs(u)
        if backend == "sequential":
            states = self._forward_sequential(
                x, neg_a, a, dt, lam, nu, input_writes
            )
        else:
            states = self._forward_parallel(
                neg_a,
                a,
                dt,
                lam,
                nu,
                input_writes,
                use_triton=backend == "triton",
                use_triton_associative=backend == "triton_associative",
                use_legacy_triton=backend == "triton_legacy",
            )
        output = self._readout(states, C, x, z)
        return (output, states) if return_states else output

    def step(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba3SecondOrder cached decoding is intentionally deferred until after reference validation"
        )

    def allocate_inference_cache(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba3SecondOrder inference-cache support is intentionally deferred"
        )
