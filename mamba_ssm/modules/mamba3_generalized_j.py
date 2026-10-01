"""Experimental fixed-complex-structure variant of Mamba-3 SISO.

This module keeps ordinary :class:`~mamba_ssm.modules.mamba3.Mamba3`
unchanged.  It replaces the Euclidean RoPE transform with a fixed,
non-orthogonal complex structure ``K_kappa`` satisfying ``K_kappa**2 = -I``.
The data-dependent scalar phase is still accumulated exactly as in Mamba-3.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
    mamba3_siso_combined,
)


def generalized_j_generator(
    kappa: float | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``[[0, -kappa], [1 / kappa, 0]]``."""

    kappa_tensor = torch.as_tensor(kappa, device=device, dtype=dtype)
    if torch.any(kappa_tensor <= 0):
        raise ValueError("kappa must be positive")
    zero = torch.zeros_like(kappa_tensor)
    return torch.stack(
        (
            torch.stack((zero, -kappa_tensor), dim=-1),
            torch.stack((kappa_tensor.reciprocal(), zero), dim=-1),
        ),
        dim=-2,
    )


def generalized_j_exponential(
    phase: torch.Tensor, kappa: float | torch.Tensor
) -> torch.Tensor:
    """Evaluate ``exp(phase * K_kappa)`` in its closed form."""

    kappa_tensor = torch.as_tensor(kappa, device=phase.device, dtype=phase.dtype)
    if torch.any(kappa_tensor <= 0):
        raise ValueError("kappa must be positive")
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    return torch.stack(
        (
            torch.stack((cosine, -kappa_tensor * sine), dim=-1),
            torch.stack((kappa_tensor.reciprocal() * sine, cosine), dim=-1),
        ),
        dim=-2,
    )


def generalized_j_primal_pairs(
    pairs: torch.Tensor, phase: torch.Tensor, kappa: float
) -> torch.Tensor:
    """Transform B/K pairs by ``E_kappa(phase)``."""

    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    first, second = pairs.unbind(dim=-1)
    return torch.stack(
        (
            cosine * first - kappa * sine * second,
            (sine / kappa) * first + cosine * second,
        ),
        dim=-1,
    )


def generalized_j_dual_pairs(
    pairs: torch.Tensor, phase: torch.Tensor, kappa: float
) -> torch.Tensor:
    """Transform C/Q pairs by ``E_kappa(phase)^{-T}``."""

    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    first, second = pairs.unbind(dim=-1)
    return torch.stack(
        (
            cosine * first - (sine / kappa) * second,
            kappa * sine * first + cosine * second,
        ),
        dim=-1,
    )


def _apply_pair_transform(
    values: torch.Tensor,
    phase: torch.Tensor,
    *,
    rotary_size: int,
    kappa: float,
    dual: bool,
) -> torch.Tensor:
    """Apply a generalized transform to the leading rotary dimensions."""

    rotary = values[..., :rotary_size].reshape(*values.shape[:-1], -1, 2)
    transform = generalized_j_dual_pairs if dual else generalized_j_primal_pairs
    rotary = transform(rotary, phase, kappa).flatten(start_dim=-2)
    if rotary_size == values.shape[-1]:
        return rotary
    return torch.cat((rotary, values[..., rotary_size:]), dim=-1)


class Mamba3GeneralizedJ(Mamba3):
    """Mamba-3 SISO with fixed non-orthogonal ``K_kappa`` phase geometry.

    ``generalized_j_kappa`` is the direct user-facing hyperparameter.  Internally
    it is represented by ``eta=log(kappa)`` and recovered with ``exp(eta)``.
    Neither value is learnable or adds a state-dict entry.  The implementation
    computes the cumulative phase and
    generalized B/C transforms in differentiable PyTorch, then reuses the
    official fused SISO scalar-decay/trapezoid recurrence with zero internal
    angles and zero internal B/C biases.
    """

    def __init__(
        self,
        *args,
        generalized_j_kappa: float | None = None,
        generalized_j_eta: float | None = None,
        is_mimo: bool = False,
        **kwargs,
    ):
        if is_mimo:
            raise NotImplementedError("Mamba3GeneralizedJ currently supports only SISO")
        if generalized_j_kappa is not None and generalized_j_eta is not None:
            raise ValueError(
                "pass generalized_j_kappa or generalized_j_eta, not both"
            )
        if generalized_j_kappa is not None:
            kappa_input = float(generalized_j_kappa)
            if not math.isfinite(kappa_input) or kappa_input <= 0.0:
                raise ValueError("generalized_j_kappa must be finite and positive")
            eta = math.log(kappa_input)
        else:
            eta = 0.0 if generalized_j_eta is None else float(generalized_j_eta)
            if not math.isfinite(eta):
                raise ValueError("generalized_j_eta must be finite")
        # Keep the requested exp parameterization in one canonical place.
        kappa = math.exp(eta)
        if not math.isfinite(kappa):
            raise ValueError("exp(generalized_j_eta) must be finite")
        super().__init__(*args, is_mimo=False, **kwargs)
        # Plain attributes: fixed experiment selectors, not parameters/buffers.
        self.generalized_j_eta = eta
        self.generalized_j_kappa = kappa

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError(
                "Mamba3GeneralizedJ inference-state caching is not implemented yet"
            )
        if cu_seqlens is not None:
            raise NotImplementedError(
                "Mamba3GeneralizedJ packed variable-length input is not implemented yet"
            )

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

        neg_a = -heavy_tail_activation(dd_A.float())
        neg_a = torch.clamp(neg_a, max=-self.A_floor)
        dt = F.softplus(dd_dt + self.dt_bias)
        adt = neg_a * dt

        # Preserve the exact data-dependent scalar phase parameterization.
        nu = math.pi * torch.tanh(angles.float())
        phase_increment = nu.unsqueeze(-2) * dt.unsqueeze(-1)
        phase = torch.cumsum(phase_increment, dim=1)
        phase = torch.remainder(phase, 2.0 * math.pi)

        # Normalize in grouped-head form, then expand because each full head
        # has its own dt-dependent phase and its own B/C bias.
        B = self.B_norm(B).squeeze(2)
        C = self.C_norm(C).squeeze(2)
        heads_per_group = self.nheads // self.num_bc_heads
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)
        B = B.float() + self.B_bias[:, 0, :].float()[None, None, :, :]
        C = C.float() + self.C_bias[:, 0, :].float()[None, None, :, :]

        B = _apply_pair_transform(
            B,
            phase,
            rotary_size=self.split_tensor_size,
            kappa=self.generalized_j_kappa,
            dual=False,
        )
        C = _apply_pair_transform(
            C,
            phase,
            rotary_size=self.split_tensor_size,
            kappa=self.generalized_j_kappa,
            dual=True,
        )

        # The true bias and phase have already been applied externally.
        zero_bias = torch.zeros_like(self.B_bias[:, 0, :])
        zero_angles = torch.zeros_like(phase)
        y = mamba3_siso_combined(
            Q=C,
            K=B,
            V=x,
            ADT=rearrange(adt, "b l h -> b h l"),
            DT=rearrange(dt, "b l h -> b h l"),
            Trap=rearrange(trap, "b l h -> b h l"),
            Q_bias=zero_bias,
            K_bias=zero_bias,
            Angles=zero_angles,
            D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            z = rearrange(z, "b l h p -> b l (h p)")
            y = self.norm(y, z)
        return self.out_proj(y.to(x.dtype))
