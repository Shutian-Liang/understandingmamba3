"""Differentiable FP32 SISO oracle for short, fixed-batch training diagnostics.

Quadratic attention form of the same exponential-trapezoidal recurrence as
Mamba-3. This deliberately avoids the fused BF16 scan, not a new architecture.
Disable TF32 at the caller when using this as a numerical reference.
"""

import math

import torch
import torch.nn.functional as F


def mamba3_siso_reference(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
    D=None, Z=None, chunk_size=64, Input_States=None,
    return_final_states=False, cu_seqlens=None,
):
    if Input_States is not None or return_final_states or cu_seqlens is not None:
        raise NotImplementedError("FP32 diagnostic oracle supports full sequences without caches only")
    with torch.autocast(device_type=Q.device.type, enabled=False):
        q, k, v = Q.float(), K.float(), V.float()
        heads = v.shape[2]
        q = q.repeat_interleave(heads // q.shape[2], dim=2) + Q_bias.float()[None, None]
        k = k.repeat_interleave(heads // k.shape[2], dim=2) + K_bias.float()[None, None]
        dt, adt = DT.float(), ADT.float()
        lam = Trap.float().sigmoid()
        theta = (Angles.float().tanh() * math.pi * dt.transpose(1, 2)[..., None]).cumsum(1)
        theta = theta.remainder(2 * math.pi)
        cos, sin = theta.cos(), theta.sin()
        pad = q.shape[-1] // 2 - cos.shape[-1]
        cos, sin = F.pad(cos, (0, pad), value=1), F.pad(sin, (0, pad))

        def rotate(x):
            pairs = x.reshape(*x.shape[:-1], -1, 2)
            a, b = pairs.unbind(-1)
            return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)

        # Combine the current write with its contribution at the next token;
        # subtract that next-token contribution from the current diagonal.
        shifted = F.pad((dt * (1 - lam))[..., 1:], (0, 1))
        scale = dt * lam + shifted
        diagonal = (q * k).sum(-1) * shifted.transpose(1, 2)
        q, k = rotate(q), rotate(k) * scale.transpose(1, 2)[..., None]
        length = q.shape[1]
        lower = torch.ones(length, length, device=q.device, dtype=torch.bool).tril(-1)
        # Stable segment sums (avoid subtracting two large cumulative sums).
        segments = adt[..., :, None].expand(*adt.shape, length).masked_fill(~lower, 0).cumsum(-2)
        causal = torch.ones_like(lower).tril()
        decay = segments.masked_fill(~causal, -torch.inf).exp()
        weights = torch.einsum("bthn,bshn->bhts", q, k) * decay
        out = torch.einsum("bhts,bshp->bthp", weights, v) - diagonal[..., None] * v
        if D is not None:
            out = out + D.float()[None, None, :, None] * v
        if Z is not None:
            out = out * F.silu(Z.float())
        return out
