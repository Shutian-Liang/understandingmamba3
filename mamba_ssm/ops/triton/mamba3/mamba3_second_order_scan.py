"""Triton recurrent scans for the experimental second-order Mamba-3 path.

These kernels are isolated from the ordinary Mamba-3 fused implementation.
The original entry points operate on already constructed affine maps.  The
fused entry points construct the token-local transition and trapezoidal input
write inside the recurrent kernel so training does not materialize the large
affine-bias tensor.  Both paths preserve FP32 recurrent states.
"""

import torch
import triton
import triton.language as tl


_TRANSITION_MODES = {
    "matched_underdamped": 0,
    "critically_damped": 1,
    "rotation_reference": 2,
    "overdamped": 3,
}


def _scan_frequency(
    a: torch.Tensor,
    nu: torch.Tensor,
    transition_mode: str,
    overdamped_rho_scale: float,
) -> torch.Tensor:
    if transition_mode != "overdamped":
        return nu
    if not 0.0 < overdamped_rho_scale <= 1.0:
        raise ValueError(
            "overdamped_rho_scale must be in (0, 1], "
            f"got {overdamped_rho_scale}"
        )
    return overdamped_rho_scale * a.unsqueeze(-1) * torch.tanh(nu)


@triton.jit
def _transition_values(a, nu, dt, MODE: tl.constexpr):
    decay = tl.exp(-a * dt)
    if MODE == 1:
        a_dt = a * dt
        f00 = decay * (1.0 + a_dt)
        f01 = decay * dt
        f10 = decay * (-a * a * dt)
        f11 = decay * (1.0 - a_dt)
    elif MODE == 3:
        # For this mode ``nu`` carries rho=rho_scale*a*tanh(original_nu),
        # computed by the PyTorch wrapper so Triton and the reference share
        # the exact same parameterization. Autograd applies the
        # rho -> (a, original_nu) chain rule outside this custom scan.
        rho = nu
        theta = rho * dt
        theta2 = theta * theta
        positive_mode = tl.exp((-a + rho) * dt)
        negative_mode = tl.exp((-a - rho) * dt)
        decay_cosh = 0.5 * (positive_mode + negative_mode)
        sinhc_series = 1.0 + theta2 / 6.0 + theta2 * theta2 / 120.0
        sinh_series = theta * sinhc_series
        decay_sinh = tl.where(
            tl.abs(theta) < 0.1,
            decay * sinh_series,
            0.5 * (positive_mode - negative_mode),
        )
        safe_rho = tl.where(tl.abs(rho) < 1.0e-12, 1.0, rho)
        decay_sinh_over_rho = tl.where(
            tl.abs(theta) < 0.1,
            decay * dt * sinhc_series,
            decay_sinh / safe_rho,
        )
        stiffness = a * a - rho * rho
        a_sinh = a * decay_sinh_over_rho
        f00 = decay_cosh + a_sinh
        f01 = decay_sinh_over_rho
        f10 = -stiffness * decay_sinh_over_rho
        f11 = decay_cosh - a_sinh
    else:
        theta = nu * dt
        cosine = tl.cos(theta)
        sine = tl.sin(theta)
        if MODE == 2:
            f00 = decay * cosine
            f01 = decay * sine
            f10 = -f01
            f11 = f00
        else:
            theta2 = theta * theta
            sinc_series = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
            sinc = tl.where(tl.abs(theta) < 1.0e-3, sinc_series, sine / theta)
            sin_over_nu = dt * sinc
            a_sine = a * sin_over_nu
            f00 = decay * (cosine + a_sine)
            f01 = decay * sin_over_nu
            f10 = decay * (-(a * a + nu * nu) * sin_over_nu)
            f11 = decay * (cosine - a_sine)
    return f00, f01, f10, f11


@triton.jit
def _transition_vjp(a, nu, dt, g00, g01, g10, g11, MODE: tl.constexpr):
    """Analytic VJP of the selected exact 2x2 transition."""
    decay = tl.exp(-a * dt)
    if MODE == 1:
        a_dt = a * dt
        m00 = 1.0 + a_dt
        m01 = dt
        m10 = -a * a * dt
        m11 = 1.0 - a_dt
        da00 = decay * (dt - dt * m00)
        da01 = decay * (-dt * m01)
        da10 = decay * (-2.0 * a * dt - dt * m10)
        da11 = decay * (-dt - dt * m11)
        ddt00 = decay * (a - a * m00)
        ddt01 = decay * (1.0 - a * m01)
        ddt10 = decay * (-a * a - a * m10)
        ddt11 = decay * (-a - a * m11)
        grad_a = g00 * da00 + g01 * da01 + g10 * da10 + g11 * da11
        grad_nu = 0.0
        grad_dt = g00 * ddt00 + g01 * ddt01 + g10 * ddt10 + g11 * ddt11
    elif MODE == 3:
        # ``nu`` is the independently differentiated rho input here. Its
        # dependence on a and the original frequency channel lives outside
        # the custom autograd Function.
        rho = nu
        theta = rho * dt
        theta2 = theta * theta
        positive_mode = tl.exp((-a + rho) * dt)
        negative_mode = tl.exp((-a - rho) * dt)
        decay_cosh = 0.5 * (positive_mode + negative_mode)
        sinhc_series = 1.0 + theta2 / 6.0 + theta2 * theta2 / 120.0
        sinh_series = theta * sinhc_series
        decay_sinh = tl.where(
            tl.abs(theta) < 0.1,
            decay * sinh_series,
            0.5 * (positive_mode - negative_mode),
        )
        safe_rho = tl.where(tl.abs(rho) < 1.0e-12, 1.0, rho)
        scaled_sinh_over_rho = tl.where(
            tl.abs(theta) < 0.1,
            decay * dt * sinhc_series,
            decay_sinh / safe_rho,
        )
        stiffness = a * a - rho * rho
        m00 = decay_cosh + a * scaled_sinh_over_rho
        m01 = scaled_sinh_over_rho
        m10 = -stiffness * scaled_sinh_over_rho
        m11 = decay_cosh - a * scaled_sinh_over_rho

        # d(exp(-a*t) * sinh(rho*t) / rho) / d(rho), with a
        # series around rho=0 to retain the critical-damping limit.
        rho2 = rho * rho
        safe_rho2 = tl.where(tl.abs(rho) < 1.0e-12, 1.0, rho2)
        t2 = dt * dt
        t3 = t2 * dt
        t5 = t3 * t2
        scaled_q_rho = tl.where(
            tl.abs(theta) < 0.1,
            decay * (
                rho * t3 / 3.0
                + rho * rho2 * t5 / 30.0
                + rho * rho2 * rho2 * t5 * t2 / 840.0
            ),
            (rho * dt * decay_cosh - decay_sinh) / safe_rho2,
        )

        c_a = -dt * decay_cosh
        q_a = -dt * scaled_sinh_over_rho
        stiffness_a = 2.0 * a
        da00 = c_a + scaled_sinh_over_rho + a * q_a
        da01 = q_a
        da10 = -(stiffness_a * scaled_sinh_over_rho + stiffness * q_a)
        da11 = c_a - scaled_sinh_over_rho - a * q_a

        c_nu = dt * decay_sinh
        q_nu = scaled_q_rho
        stiffness_nu = -2.0 * rho
        dnu00 = c_nu + a * q_nu
        dnu01 = q_nu
        dnu10 = -(stiffness_nu * scaled_sinh_over_rho + stiffness * q_nu)
        dnu11 = c_nu - a * q_nu

        c_dt = -a * decay_cosh + rho * decay_sinh
        q_dt = decay_cosh - a * scaled_sinh_over_rho
        ddt00 = c_dt + a * q_dt
        ddt01 = q_dt
        ddt10 = -stiffness * q_dt
        ddt11 = c_dt - a * q_dt

        grad_a = g00 * da00 + g01 * da01 + g10 * da10 + g11 * da11
        grad_nu = (
            g00 * dnu00 + g01 * dnu01 + g10 * dnu10 + g11 * dnu11
        )
        grad_dt = g00 * ddt00 + g01 * ddt01 + g10 * ddt10 + g11 * ddt11
    else:
        theta = nu * dt
        sine = tl.sin(theta)
        cosine = tl.cos(theta)
        if MODE == 2:
            m00 = cosine
            m01 = sine
            m10 = -sine
            m11 = cosine
            grad_a = -dt * decay * (
                g00 * m00 + g01 * m01 + g10 * m10 + g11 * m11
            )
            grad_nu = decay * dt * (
                -g00 * sine + g01 * cosine - g10 * cosine - g11 * sine
            )
            grad_dt = decay * (
                g00 * (-nu * sine - a * m00)
                + g01 * (nu * cosine - a * m01)
                + g10 * (-nu * cosine - a * m10)
                + g11 * (-nu * sine - a * m11)
            )
        else:
            theta2 = theta * theta
            sinc_series = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
            sinc = tl.where(tl.abs(theta) < 1.0e-3, sinc_series, sine / theta)
            q = dt * sinc
            theta3 = theta2 * theta
            theta5 = theta3 * theta2
            dq_dnu_series = dt * dt * (
                -theta / 3.0 + theta3 / 30.0 - theta5 / 840.0
            )
            dq_dnu = tl.where(
                tl.abs(theta) < 1.0e-3,
                dq_dnu_series,
                (nu * dt * cosine - sine) / (nu * nu),
            )
            radius = a * a + nu * nu
            m00 = cosine + a * q
            m01 = q
            m10 = -radius * q
            m11 = cosine - a * q
            da00 = decay * (q - dt * m00)
            da01 = decay * (-dt * m01)
            da10 = decay * (-2.0 * a * q - dt * m10)
            da11 = decay * (-q - dt * m11)
            dc_dnu = -dt * sine
            dnu00 = decay * (dc_dnu + a * dq_dnu)
            dnu01 = decay * dq_dnu
            dnu10 = decay * (-2.0 * nu * q - radius * dq_dnu)
            dnu11 = decay * (dc_dnu - a * dq_dnu)
            dc_ddt = -nu * sine
            ddt00 = decay * (dc_ddt + a * cosine - a * m00)
            ddt01 = decay * (cosine - a * m01)
            ddt10 = decay * (-radius * cosine - a * m10)
            ddt11 = decay * (dc_ddt - a * cosine - a * m11)
            grad_a = g00 * da00 + g01 * da01 + g10 * da10 + g11 * da11
            grad_nu = (
                g00 * dnu00 + g01 * dnu01 + g10 * dnu10 + g11 * dnu11
            )
            grad_dt = g00 * ddt00 + g01 * ddt01 + g10 * ddt10 + g11 * ddt11
    return grad_a, grad_nu, grad_dt


@triton.jit
def _affine_2x2_combine(
    left_f00,
    left_f01,
    left_f10,
    left_f11,
    left_b0,
    left_b1,
    right_f00,
    right_f01,
    right_f10,
    right_f11,
    right_b0,
    right_b1,
):
    """Compose an earlier affine map with a later affine map.

    Triton's scan passes the earlier prefix as ``left`` and the later item as
    ``right``.  The returned map is therefore ``right(left(x))``.  Matrix
    multiplication order is intentional because token-dependent 2x2
    transitions do not generally commute.
    """
    out_f00 = right_f00 * left_f00 + right_f01 * left_f10
    out_f01 = right_f00 * left_f01 + right_f01 * left_f11
    out_f10 = right_f10 * left_f00 + right_f11 * left_f10
    out_f11 = right_f10 * left_f01 + right_f11 * left_f11
    out_b0 = right_f00 * left_b0 + right_f01 * left_b1 + right_b0
    out_b1 = right_f10 * left_b0 + right_f11 * left_b1 + right_b1
    return out_f00, out_f01, out_f10, out_f11, out_b0, out_b1


@triton.jit
def _second_order_associative_fwd_kernel(
    a,
    nu,
    dt,
    trap,
    writes,
    states,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_P: tl.constexpr,
    MODE: tl.constexpr,
):
    """Inclusive affine prefix scan, parallelized over the time dimension."""
    pid_pair = tl.program_id(0)
    pid_p = tl.program_id(1)
    pair = pid_pair % PAIRS
    pid_pair = pid_pair // PAIRS
    head = pid_pair % NHEADS
    batch = pid_pair // NHEADS

    offsets_l_vec = tl.arange(0, BLOCK_L)
    offsets_p_vec = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offsets_l = offsets_l_vec[:, None]
    offsets_p = offsets_p_vec[None, :]
    mask_l_vec = offsets_l_vec < SEQLEN
    mask = mask_l_vec[:, None] & (offsets_p < HEAD_DIM)
    token_head_vec = (batch * SEQLEN + offsets_l_vec) * NHEADS + head

    a_vec = tl.load(a + token_head_vec, mask=mask_l_vec, other=0.0)
    dt_vec = tl.load(dt + token_head_vec, mask=mask_l_vec, other=0.0)
    trap_vec = tl.load(trap + token_head_vec, mask=mask_l_vec, other=0.0)
    is_rotary = pair < ROTARY_PAIRS
    nu_vec = tl.load(
        nu + token_head_vec * ROTARY_PAIRS + pair,
        mask=mask_l_vec & is_rotary,
        other=0.0,
    )
    rot_f00, rot_f01, rot_f10, rot_f11 = _transition_values(
        a_vec, nu_vec, dt_vec, MODE
    )
    alpha = tl.exp(-a_vec * dt_vec)
    f00_vec = tl.where(is_rotary, rot_f00, alpha)
    f01_vec = tl.where(is_rotary, rot_f01, 0.0)
    f10_vec = tl.where(is_rotary, rot_f10, 0.0)
    f11_vec = tl.where(is_rotary, rot_f11, alpha)
    # Padded tokens are identity maps so they cannot alter valid prefixes.
    f00_vec = tl.where(mask_l_vec, f00_vec, 1.0)
    f01_vec = tl.where(mask_l_vec, f01_vec, 0.0)
    f10_vec = tl.where(mask_l_vec, f10_vec, 0.0)
    f11_vec = tl.where(mask_l_vec, f11_vec, 1.0)

    token_head = token_head_vec[:, None]
    write_base = ((token_head * HEAD_DIM + offsets_p) * PAIRS + pair) * 2
    current0 = tl.load(writes + write_base, mask=mask, other=0.0)
    current1 = tl.load(writes + write_base + 1, mask=mask, other=0.0)
    previous_head = (batch * SEQLEN + offsets_l_vec - 1) * NHEADS + head
    previous_base = (
        (previous_head[:, None] * HEAD_DIM + offsets_p) * PAIRS + pair
    ) * 2
    previous_mask = mask & (offsets_l > 0)
    previous0 = tl.load(writes + previous_base, mask=previous_mask, other=0.0)
    previous1 = tl.load(writes + previous_base + 1, mask=previous_mask, other=0.0)

    f00 = f00_vec[:, None] + tl.zeros((BLOCK_L, BLOCK_P), dtype=tl.float32)
    f01 = f01_vec[:, None] + tl.zeros((BLOCK_L, BLOCK_P), dtype=tl.float32)
    f10 = f10_vec[:, None] + tl.zeros((BLOCK_L, BLOCK_P), dtype=tl.float32)
    f11 = f11_vec[:, None] + tl.zeros((BLOCK_L, BLOCK_P), dtype=tl.float32)
    previous_weight = ((1.0 - trap_vec) * dt_vec)[:, None]
    current_weight = (trap_vec * dt_vec)[:, None]
    pre0 = previous_weight * previous0
    pre1 = previous_weight * previous1
    bias0 = f00 * pre0 + f01 * pre1 + current_weight * current0
    bias1 = f10 * pre0 + f11 * pre1 + current_weight * current1
    bias0 = tl.where(mask, bias0, 0.0)
    bias1 = tl.where(mask, bias1, 0.0)

    _, _, _, _, state0, state1 = tl.associative_scan(
        (f00, f01, f10, f11, bias0, bias1),
        axis=0,
        combine_fn=_affine_2x2_combine,
    )
    tl.store(states + write_base, state0, mask=mask)
    tl.store(states + write_base + 1, state1, mask=mask)


@triton.jit
def _second_order_associative_bwd_kernel(
    a,
    nu,
    dt,
    trap,
    writes,
    states,
    grad_states,
    grad_a,
    grad_nu,
    grad_dt,
    grad_trap,
    grad_writes,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_P: tl.constexpr,
    MODE: tl.constexpr,
):
    """Reverse associative scan plus pointwise recurrence VJP."""
    pid_pair = tl.program_id(0)
    pid_p = tl.program_id(1)
    pair = pid_pair % PAIRS
    pid_pair = pid_pair // PAIRS
    head = pid_pair % NHEADS
    batch = pid_pair // NHEADS

    offsets_l_vec = tl.arange(0, BLOCK_L)
    offsets_p_vec = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offsets_l = offsets_l_vec[:, None]
    offsets_p = offsets_p_vec[None, :]
    mask_l_vec = offsets_l_vec < SEQLEN
    mask = mask_l_vec[:, None] & (offsets_p < HEAD_DIM)
    token_head_vec = (batch * SEQLEN + offsets_l_vec) * NHEADS + head
    token_head = token_head_vec[:, None]

    a_vec = tl.load(a + token_head_vec, mask=mask_l_vec, other=0.0)
    dt_vec = tl.load(dt + token_head_vec, mask=mask_l_vec, other=0.0)
    trap_vec = tl.load(trap + token_head_vec, mask=mask_l_vec, other=0.0)
    is_rotary = pair < ROTARY_PAIRS
    nu_vec = tl.load(
        nu + token_head_vec * ROTARY_PAIRS + pair,
        mask=mask_l_vec & is_rotary,
        other=0.0,
    )
    rot_f00, rot_f01, rot_f10, rot_f11 = _transition_values(
        a_vec, nu_vec, dt_vec, MODE
    )
    alpha = tl.exp(-a_vec * dt_vec)
    f00_vec = tl.where(is_rotary, rot_f00, alpha)
    f01_vec = tl.where(is_rotary, rot_f01, 0.0)
    f10_vec = tl.where(is_rotary, rot_f10, 0.0)
    f11_vec = tl.where(is_rotary, rot_f11, alpha)

    write_base = ((token_head * HEAD_DIM + offsets_p) * PAIRS + pair) * 2
    direct0 = tl.load(grad_states + write_base, mask=mask, other=0.0)
    direct1 = tl.load(grad_states + write_base + 1, mask=mask, other=0.0)

    # G_t = direct_t + F_{t+1}^T G_{t+1}.  A reverse affine scan over
    # (F_{t+1}^T, direct_t) returns every total state gradient G_t.
    next_index = offsets_l_vec + 1
    next_valid = next_index < SEQLEN
    next_head_vec = (batch * SEQLEN + next_index) * NHEADS + head
    next_a = tl.load(a + next_head_vec, mask=next_valid, other=0.0)
    next_dt = tl.load(dt + next_head_vec, mask=next_valid, other=0.0)
    next_trap = tl.load(trap + next_head_vec, mask=next_valid, other=0.0)
    next_nu = tl.load(
        nu + next_head_vec * ROTARY_PAIRS + pair,
        mask=next_valid & is_rotary,
        other=0.0,
    )
    next_r00, next_r01, next_r10, next_r11 = _transition_values(
        next_a, next_nu, next_dt, MODE
    )
    next_alpha = tl.exp(-next_a * next_dt)
    next_f00 = tl.where(is_rotary, next_r00, next_alpha)
    next_f01 = tl.where(is_rotary, next_r01, 0.0)
    next_f10 = tl.where(is_rotary, next_r10, 0.0)
    next_f11 = tl.where(is_rotary, next_r11, next_alpha)
    # The final token has no future state; identity is harmless because the
    # inclusive affine scan starts from a zero state and its bias is direct_t.
    next_f00 = tl.where(next_valid, next_f00, 1.0)
    next_f01 = tl.where(next_valid, next_f01, 0.0)
    next_f10 = tl.where(next_valid, next_f10, 0.0)
    next_f11 = tl.where(next_valid, next_f11, 1.0)
    # Padded entries appear before valid tokens in reverse order.  Identity
    # maps with zero bias leave every valid reverse prefix unchanged.
    scan_f00_vec = tl.where(mask_l_vec, next_f00, 1.0)
    scan_f01_vec = tl.where(mask_l_vec, next_f10, 0.0)
    scan_f10_vec = tl.where(mask_l_vec, next_f01, 0.0)
    scan_f11_vec = tl.where(mask_l_vec, next_f11, 1.0)
    scan_f00 = scan_f00_vec[:, None] + tl.zeros(
        (BLOCK_L, BLOCK_P), dtype=tl.float32
    )
    scan_f01 = scan_f01_vec[:, None] + tl.zeros(
        (BLOCK_L, BLOCK_P), dtype=tl.float32
    )
    scan_f10 = scan_f10_vec[:, None] + tl.zeros(
        (BLOCK_L, BLOCK_P), dtype=tl.float32
    )
    scan_f11 = scan_f11_vec[:, None] + tl.zeros(
        (BLOCK_L, BLOCK_P), dtype=tl.float32
    )
    _, _, _, _, total0, total1 = tl.associative_scan(
        (scan_f00, scan_f01, scan_f10, scan_f11, direct0, direct1),
        axis=0,
        combine_fn=_affine_2x2_combine,
        reverse=True,
    )

    current0 = tl.load(writes + write_base, mask=mask, other=0.0)
    current1 = tl.load(writes + write_base + 1, mask=mask, other=0.0)
    previous_head = (batch * SEQLEN + offsets_l_vec - 1) * NHEADS + head
    previous_base = (
        (previous_head[:, None] * HEAD_DIM + offsets_p) * PAIRS + pair
    ) * 2
    previous_mask = mask & (offsets_l > 0)
    previous0 = tl.load(writes + previous_base, mask=previous_mask, other=0.0)
    previous1 = tl.load(writes + previous_base + 1, mask=previous_mask, other=0.0)
    previous_state0 = tl.load(
        states + previous_base, mask=previous_mask, other=0.0
    )
    previous_state1 = tl.load(
        states + previous_base + 1, mask=previous_mask, other=0.0
    )

    f00 = f00_vec[:, None]
    f01 = f01_vec[:, None]
    f10 = f10_vec[:, None]
    f11 = f11_vec[:, None]
    previous_weight = ((1.0 - trap_vec) * dt_vec)[:, None]
    current_weight = (trap_vec * dt_vec)[:, None]
    pre0 = previous_state0 + previous_weight * previous0
    pre1 = previous_state1 + previous_weight * previous1
    grad_pre0 = f00 * total0 + f10 * total1
    grad_pre1 = f01 * total0 + f11 * total1

    next_previous_weight = ((1.0 - next_trap) * next_dt)[:, None]
    next_previous_weight = tl.where(next_valid[:, None], next_previous_weight, 0.0)
    grad_write0 = current_weight * total0 + next_previous_weight * (total0 - direct0)
    grad_write1 = current_weight * total1 + next_previous_weight * (total1 - direct1)
    tl.store(grad_writes + write_base, grad_write0, mask=mask)
    tl.store(grad_writes + write_base + 1, grad_write1, mask=mask)

    grad_f00 = tl.sum(tl.where(mask, total0 * pre0, 0.0), axis=1)
    grad_f01 = tl.sum(tl.where(mask, total0 * pre1, 0.0), axis=1)
    grad_f10 = tl.sum(tl.where(mask, total1 * pre0, 0.0), axis=1)
    grad_f11 = tl.sum(tl.where(mask, total1 * pre1, 0.0), axis=1)
    grad_prev_weight = tl.sum(
        tl.where(mask, grad_pre0 * previous0 + grad_pre1 * previous1, 0.0),
        axis=1,
    )
    grad_current_weight = tl.sum(
        tl.where(mask, total0 * current0 + total1 * current1, 0.0),
        axis=1,
    )
    grad_a_rot, grad_nu_rot, grad_dt_rot = _transition_vjp(
        a_vec,
        nu_vec,
        dt_vec,
        grad_f00,
        grad_f01,
        grad_f10,
        grad_f11,
        MODE,
    )
    grad_diagonal = grad_f00 + grad_f11
    grad_a_scalar = -dt_vec * alpha * grad_diagonal
    grad_dt_scalar = -a_vec * alpha * grad_diagonal
    grad_a_value = tl.where(is_rotary, grad_a_rot, grad_a_scalar)
    grad_nu_value = tl.where(is_rotary, grad_nu_rot, 0.0)
    grad_dt_value = tl.where(is_rotary, grad_dt_rot, grad_dt_scalar)
    grad_dt_value += (1.0 - trap_vec) * grad_prev_weight
    grad_dt_value += trap_vec * grad_current_weight
    grad_trap_value = dt_vec * (grad_current_weight - grad_prev_weight)

    pair_offset = token_head_vec * PAIRS + pair
    tl.atomic_add(grad_a + pair_offset, grad_a_value, mask=mask_l_vec)
    tl.atomic_add(grad_dt + pair_offset, grad_dt_value, mask=mask_l_vec)
    tl.atomic_add(grad_trap + pair_offset, grad_trap_value, mask=mask_l_vec)
    tl.atomic_add(
        grad_nu + token_head_vec * ROTARY_PAIRS + pair,
        grad_nu_value,
        mask=mask_l_vec & is_rotary,
    )


@triton.jit
def _second_order_scan_2x2_fwd_kernel(
    a,
    nu,
    dt,
    trap,
    writes,
    states,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % PAIRS
    pid = pid // PAIRS
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    state0 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    state1 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    previous_input0 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    previous_input1 = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = 0
    while token < SEQLEN:
        token_head = (batch * SEQLEN + token) * NHEADS + head
        a_t = tl.load(a + token_head)
        dt_t = tl.load(dt + token_head)
        trap_t = tl.load(trap + token_head)
        is_rotary = pair < ROTARY_PAIRS
        nu_t = tl.load(
            nu + token_head * ROTARY_PAIRS + pair,
            mask=is_rotary,
            other=0.0,
        )
        rot_f00, rot_f01, rot_f10, rot_f11 = _transition_values(
            a_t, nu_t, dt_t, MODE
        )
        alpha = tl.exp(-a_t * dt_t)
        f00 = tl.where(is_rotary, rot_f00, alpha)
        f01 = tl.where(is_rotary, rot_f01, 0.0)
        f10 = tl.where(is_rotary, rot_f10, 0.0)
        f11 = tl.where(is_rotary, rot_f11, alpha)
        write_base = (
            ((token_head * HEAD_DIM + offsets_p) * PAIRS + pair) * 2
        )
        current_input0 = tl.load(writes + write_base, mask=mask_p, other=0.0)
        current_input1 = tl.load(writes + write_base + 1, mask=mask_p, other=0.0)
        previous_weight = (1.0 - trap_t) * dt_t
        current_weight = trap_t * dt_t
        pre0 = state0 + previous_weight * previous_input0
        pre1 = state1 + previous_weight * previous_input1
        next0 = f00 * pre0 + f01 * pre1 + current_weight * current_input0
        next1 = f10 * pre0 + f11 * pre1 + current_weight * current_input1
        tl.store(states + write_base, next0, mask=mask_p)
        tl.store(states + write_base + 1, next1, mask=mask_p)
        state0, state1 = next0, next1
        previous_input0, previous_input1 = current_input0, current_input1
        token += 1


@triton.jit
def _second_order_scan_2x2_bwd_kernel(
    a,
    nu,
    dt,
    trap,
    writes,
    states,
    grad_states,
    grad_a,
    grad_nu,
    grad_dt,
    grad_trap,
    grad_writes,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % PAIRS
    pid = pid // PAIRS
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    state_carry0 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    state_carry1 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    input_carry0 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    input_carry1 = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = SEQLEN - 1
    while token >= 0:
        token_head = (batch * SEQLEN + token) * NHEADS + head
        a_t = tl.load(a + token_head)
        dt_t = tl.load(dt + token_head)
        trap_t = tl.load(trap + token_head)
        is_rotary = pair < ROTARY_PAIRS
        nu_t = tl.load(
            nu + token_head * ROTARY_PAIRS + pair,
            mask=is_rotary,
            other=0.0,
        )
        rot_f00, rot_f01, rot_f10, rot_f11 = _transition_values(
            a_t, nu_t, dt_t, MODE
        )
        alpha = tl.exp(-a_t * dt_t)
        f00 = tl.where(is_rotary, rot_f00, alpha)
        f01 = tl.where(is_rotary, rot_f01, 0.0)
        f10 = tl.where(is_rotary, rot_f10, 0.0)
        f11 = tl.where(is_rotary, rot_f11, alpha)
        write_base = (
            ((token_head * HEAD_DIM + offsets_p) * PAIRS + pair) * 2
        )
        current_input0 = tl.load(writes + write_base, mask=mask_p, other=0.0)
        current_input1 = tl.load(writes + write_base + 1, mask=mask_p, other=0.0)
        previous_head = (batch * SEQLEN + token - 1) * NHEADS + head
        previous_base = (
            ((previous_head * HEAD_DIM + offsets_p) * PAIRS + pair) * 2
        )
        previous_input0 = tl.load(
            writes + previous_base, mask=mask_p & (token > 0), other=0.0
        )
        previous_input1 = tl.load(
            writes + previous_base + 1, mask=mask_p & (token > 0), other=0.0
        )
        previous_state0 = tl.load(
            states + previous_base, mask=mask_p & (token > 0), other=0.0
        )
        previous_state1 = tl.load(
            states + previous_base + 1, mask=mask_p & (token > 0), other=0.0
        )

        grad0 = tl.load(grad_states + write_base, mask=mask_p, other=0.0) + state_carry0
        grad1 = tl.load(grad_states + write_base + 1, mask=mask_p, other=0.0) + state_carry1
        previous_weight = (1.0 - trap_t) * dt_t
        current_weight = trap_t * dt_t
        pre0 = previous_state0 + previous_weight * previous_input0
        pre1 = previous_state1 + previous_weight * previous_input1

        grad_write0 = current_weight * grad0 + input_carry0
        grad_write1 = current_weight * grad1 + input_carry1
        tl.store(grad_writes + write_base, grad_write0, mask=mask_p)
        tl.store(grad_writes + write_base + 1, grad_write1, mask=mask_p)

        grad_pre0 = f00 * grad0 + f10 * grad1
        grad_pre1 = f01 * grad0 + f11 * grad1
        grad_previous_weight = tl.sum(
            grad_pre0 * previous_input0 + grad_pre1 * previous_input1
        )
        grad_current_weight = tl.sum(
            grad0 * current_input0 + grad1 * current_input1
        )
        pair_offset = token_head * PAIRS + pair
        grad_f00 = tl.sum(grad0 * pre0)
        grad_f01 = tl.sum(grad0 * pre1)
        grad_f10 = tl.sum(grad1 * pre0)
        grad_f11 = tl.sum(grad1 * pre1)
        grad_a_rot, grad_nu_rot, grad_dt_rot = _transition_vjp(
            a_t,
            nu_t,
            dt_t,
            grad_f00,
            grad_f01,
            grad_f10,
            grad_f11,
            MODE,
        )
        grad_diagonal = grad_f00 + grad_f11
        grad_a_scalar = -dt_t * alpha * grad_diagonal
        grad_dt_scalar = -a_t * alpha * grad_diagonal
        grad_a_t = tl.where(is_rotary, grad_a_rot, grad_a_scalar)
        grad_nu_t = tl.where(is_rotary, grad_nu_rot, 0.0)
        grad_dt_transition = tl.where(is_rotary, grad_dt_rot, grad_dt_scalar)
        tl.store(grad_a + pair_offset, grad_a_t)
        tl.store(
            grad_nu + token_head * ROTARY_PAIRS + pair,
            grad_nu_t,
            mask=is_rotary,
        )
        tl.store(
            grad_dt + pair_offset,
            grad_dt_transition
            + (1.0 - trap_t) * grad_previous_weight
            + trap_t * grad_current_weight,
        )
        tl.store(
            grad_trap + pair_offset,
            dt_t * (grad_current_weight - grad_previous_weight),
        )

        state_carry0, state_carry1 = grad_pre0, grad_pre1
        input_carry0 = previous_weight * grad_pre0
        input_carry1 = previous_weight * grad_pre1
        token -= 1


@triton.jit
def _second_order_scan_scalar_fwd_kernel(
    neg_a,
    dt,
    trap,
    writes,
    states,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    state_index = pid % DSTATE
    pid = pid // DSTATE
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    state = tl.zeros((BLOCK_P,), dtype=tl.float32)
    previous_input = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = 0
    while token < SEQLEN:
        token_head = (batch * SEQLEN + token) * NHEADS + head
        neg_a_t = tl.load(neg_a + token_head)
        dt_t = tl.load(dt + token_head)
        trap_t = tl.load(trap + token_head)
        alpha = tl.exp(neg_a_t * dt_t)
        write_base = (
            (token_head * HEAD_DIM + offsets_p) * DSTATE + state_index
        )
        current_input = tl.load(writes + write_base, mask=mask_p, other=0.0)
        previous_weight = (1.0 - trap_t) * dt_t
        current_weight = trap_t * dt_t
        state = alpha * (state + previous_weight * previous_input) + current_weight * current_input
        tl.store(states + write_base, state, mask=mask_p)
        previous_input = current_input
        token += 1


@triton.jit
def _second_order_scan_scalar_bwd_kernel(
    neg_a,
    dt,
    trap,
    writes,
    states,
    grad_states,
    grad_alpha,
    grad_dt_weight,
    grad_trap,
    grad_writes,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    state_index = pid % DSTATE
    pid = pid // DSTATE
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    state_carry = tl.zeros((BLOCK_P,), dtype=tl.float32)
    input_carry = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = SEQLEN - 1
    while token >= 0:
        token_head = (batch * SEQLEN + token) * NHEADS + head
        neg_a_t = tl.load(neg_a + token_head)
        dt_t = tl.load(dt + token_head)
        trap_t = tl.load(trap + token_head)
        alpha = tl.exp(neg_a_t * dt_t)
        write_base = (
            (token_head * HEAD_DIM + offsets_p) * DSTATE + state_index
        )
        current_input = tl.load(writes + write_base, mask=mask_p, other=0.0)
        previous_head = (batch * SEQLEN + token - 1) * NHEADS + head
        previous_base = (
            (previous_head * HEAD_DIM + offsets_p) * DSTATE + state_index
        )
        previous_input = tl.load(
            writes + previous_base, mask=mask_p & (token > 0), other=0.0
        )
        previous_state = tl.load(
            states + previous_base, mask=mask_p & (token > 0), other=0.0
        )
        grad = tl.load(grad_states + write_base, mask=mask_p, other=0.0) + state_carry
        previous_weight = (1.0 - trap_t) * dt_t
        current_weight = trap_t * dt_t
        pre = previous_state + previous_weight * previous_input

        tl.store(
            grad_writes + write_base,
            current_weight * grad + input_carry,
            mask=mask_p,
        )
        grad_pre = alpha * grad
        grad_previous_weight = tl.sum(grad_pre * previous_input)
        grad_current_weight = tl.sum(grad * current_input)
        scalar_offset = token_head * DSTATE + state_index
        tl.store(grad_alpha + scalar_offset, tl.sum(grad * pre))
        tl.store(
            grad_dt_weight + scalar_offset,
            (1.0 - trap_t) * grad_previous_weight
            + trap_t * grad_current_weight,
        )
        tl.store(
            grad_trap + scalar_offset,
            dt_t * (grad_current_weight - grad_previous_weight),
        )
        state_carry = grad_pre
        input_carry = alpha * previous_weight * grad
        token -= 1


@triton.jit
def _affine_scan_2x2_fwd_kernel(
    transition,
    bias,
    states,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % PAIRS
    pid = pid // PAIRS
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    state0 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    state1 = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = 0
    while token < SEQLEN:
        transition_base = (
            (((batch * SEQLEN + token) * NHEADS + head) * PAIRS + pair) * 4
        )
        f00 = tl.load(transition + transition_base)
        f01 = tl.load(transition + transition_base + 1)
        f10 = tl.load(transition + transition_base + 2)
        f11 = tl.load(transition + transition_base + 3)
        bias_base = (
            ((((batch * SEQLEN + token) * NHEADS + head) * HEAD_DIM + offsets_p)
             * PAIRS + pair)
            * 2
        )
        bias0 = tl.load(bias + bias_base, mask=mask_p, other=0.0)
        bias1 = tl.load(bias + bias_base + 1, mask=mask_p, other=0.0)
        next0 = f00 * state0 + f01 * state1 + bias0
        next1 = f10 * state0 + f11 * state1 + bias1
        tl.store(states + bias_base, next0, mask=mask_p)
        tl.store(states + bias_base + 1, next1, mask=mask_p)
        state0, state1 = next0, next1
        token += 1


@triton.jit
def _affine_scan_2x2_bwd_kernel(
    transition,
    states,
    grad_states,
    grad_transition,
    grad_bias,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % PAIRS
    pid = pid // PAIRS
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    carry0 = tl.zeros((BLOCK_P,), dtype=tl.float32)
    carry1 = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = SEQLEN - 1
    while token >= 0:
        state_base = (
            ((((batch * SEQLEN + token) * NHEADS + head) * HEAD_DIM + offsets_p)
             * PAIRS + pair)
            * 2
        )
        grad0 = tl.load(grad_states + state_base, mask=mask_p, other=0.0) + carry0
        grad1 = tl.load(grad_states + state_base + 1, mask=mask_p, other=0.0) + carry1
        tl.store(grad_bias + state_base, grad0, mask=mask_p)
        tl.store(grad_bias + state_base + 1, grad1, mask=mask_p)

        previous_base = (
            ((((batch * SEQLEN + token - 1) * NHEADS + head) * HEAD_DIM + offsets_p)
             * PAIRS + pair)
            * 2
        )
        previous0 = tl.load(
            states + previous_base, mask=mask_p & (token > 0), other=0.0
        )
        previous1 = tl.load(
            states + previous_base + 1, mask=mask_p & (token > 0), other=0.0
        )
        transition_base = (
            (((batch * SEQLEN + token) * NHEADS + head) * PAIRS + pair) * 4
        )
        tl.store(grad_transition + transition_base, tl.sum(grad0 * previous0))
        tl.store(grad_transition + transition_base + 1, tl.sum(grad0 * previous1))
        tl.store(grad_transition + transition_base + 2, tl.sum(grad1 * previous0))
        tl.store(grad_transition + transition_base + 3, tl.sum(grad1 * previous1))

        f00 = tl.load(transition + transition_base)
        f01 = tl.load(transition + transition_base + 1)
        f10 = tl.load(transition + transition_base + 2)
        f11 = tl.load(transition + transition_base + 3)
        carry0 = f00 * grad0 + f10 * grad1
        carry1 = f01 * grad0 + f11 * grad1
        token -= 1


@triton.jit
def _affine_scan_scalar_fwd_kernel(
    transition,
    bias,
    states,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    state_index = pid % DSTATE
    pid = pid // DSTATE
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    state = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = 0
    while token < SEQLEN:
        transition_offset = (batch * SEQLEN + token) * NHEADS + head
        alpha = tl.load(transition + transition_offset)
        bias_offset = (
            (((batch * SEQLEN + token) * NHEADS + head) * HEAD_DIM + offsets_p)
            * DSTATE
            + state_index
        )
        local_bias = tl.load(bias + bias_offset, mask=mask_p, other=0.0)
        state = alpha * state + local_bias
        tl.store(states + bias_offset, state, mask=mask_p)
        token += 1


@triton.jit
def _affine_scan_scalar_bwd_kernel(
    transition,
    states,
    grad_states,
    grad_transition,
    grad_bias,
    SEQLEN,
    NHEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    state_index = pid % DSTATE
    pid = pid // DSTATE
    head = pid % NHEADS
    batch = pid // NHEADS
    offsets_p = tl.arange(0, BLOCK_P)
    mask_p = offsets_p < HEAD_DIM
    carry = tl.zeros((BLOCK_P,), dtype=tl.float32)

    token = SEQLEN - 1
    while token >= 0:
        state_offset = (
            (((batch * SEQLEN + token) * NHEADS + head) * HEAD_DIM + offsets_p)
            * DSTATE
            + state_index
        )
        grad = tl.load(grad_states + state_offset, mask=mask_p, other=0.0) + carry
        tl.store(grad_bias + state_offset, grad, mask=mask_p)
        previous_offset = (
            (((batch * SEQLEN + token - 1) * NHEADS + head) * HEAD_DIM + offsets_p)
            * DSTATE
            + state_index
        )
        previous = tl.load(
            states + previous_offset, mask=mask_p & (token > 0), other=0.0
        )
        transition_offset = (batch * SEQLEN + token) * NHEADS + head
        tl.atomic_add(grad_transition + transition_offset, tl.sum(grad * previous))
        alpha = tl.load(transition + transition_offset)
        carry = alpha * grad
        token -= 1


class _AffineScan2x2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transition, bias):
        transition = transition.contiguous()
        bias = bias.contiguous()
        if transition.dtype != torch.float32 or bias.dtype != torch.float32:
            raise TypeError("The Triton 2x2 affine scan currently requires FP32 inputs")
        batch, seqlen, nheads, pairs = transition.shape[:4]
        head_dim = bias.shape[3]
        if head_dim > 128:
            raise ValueError("The Triton affine scan currently supports headdim <= 128")
        states = torch.empty_like(bias)
        block_p = triton.next_power_of_2(head_dim)
        _affine_scan_2x2_fwd_kernel[(batch * nheads * pairs,)](
            transition,
            bias,
            states,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            PAIRS=pairs,
            BLOCK_P=block_p,
            num_warps=4,
        )
        ctx.save_for_backward(transition, states)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        transition, states = ctx.saved_tensors
        grad_states = grad_states.contiguous()
        batch, seqlen, nheads, pairs = transition.shape[:4]
        head_dim = states.shape[3]
        grad_transition = torch.empty_like(transition)
        grad_bias = torch.empty_like(states)
        block_p = triton.next_power_of_2(head_dim)
        _affine_scan_2x2_bwd_kernel[(batch * nheads * pairs,)](
            transition,
            states,
            grad_states,
            grad_transition,
            grad_bias,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            PAIRS=pairs,
            BLOCK_P=block_p,
            num_warps=4,
        )
        return grad_transition, grad_bias


class _AffineScanScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transition, bias):
        transition = transition.contiguous()
        bias = bias.contiguous()
        if transition.dtype != torch.float32 or bias.dtype != torch.float32:
            raise TypeError("The Triton scalar affine scan currently requires FP32 inputs")
        batch, seqlen, nheads = transition.shape
        head_dim, d_state = bias.shape[-2:]
        if head_dim > 128:
            raise ValueError("The Triton affine scan currently supports headdim <= 128")
        states = torch.empty_like(bias)
        block_p = triton.next_power_of_2(head_dim)
        _affine_scan_scalar_fwd_kernel[(batch * nheads * d_state,)](
            transition,
            bias,
            states,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            DSTATE=d_state,
            BLOCK_P=block_p,
            num_warps=4,
        )
        ctx.save_for_backward(transition, states)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        transition, states = ctx.saved_tensors
        grad_states = grad_states.contiguous()
        batch, seqlen, nheads = transition.shape
        head_dim, d_state = states.shape[-2:]
        grad_transition = torch.zeros_like(transition)
        grad_bias = torch.empty_like(states)
        block_p = triton.next_power_of_2(head_dim)
        _affine_scan_scalar_bwd_kernel[(batch * nheads * d_state,)](
            transition,
            states,
            grad_states,
            grad_transition,
            grad_bias,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            DSTATE=d_state,
            BLOCK_P=block_p,
            num_warps=4,
        )
        return grad_transition, grad_bias


def triton_affine_scan_2x2(transition: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Differentiable zero-initialized FP32 affine scan on CUDA."""
    if not transition.is_cuda or not bias.is_cuda:
        raise ValueError("The Triton affine scan requires CUDA tensors")
    return _AffineScan2x2.apply(transition, bias)


def triton_affine_scan_scalar(transition: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Differentiable zero-initialized scalar-affine scan on CUDA."""
    if not transition.is_cuda or not bias.is_cuda:
        raise ValueError("The Triton affine scan requires CUDA tensors")
    return _AffineScanScalar.apply(transition, bias)


class _SecondOrderScan2x2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, nu, dt, trap, writes, rotary_pairs, mode):
        a = a.float().contiguous()
        nu = nu.float().contiguous()
        dt = dt.float().contiguous()
        trap = trap.float().contiguous()
        writes = writes.float().contiguous()
        batch, seqlen, nheads, head_dim, pairs = writes.shape[:5]
        if head_dim > 128:
            raise ValueError("The fused second-order scan supports headdim <= 128")
        if a.shape != (batch, seqlen, nheads):
            raise ValueError(f"Unexpected decay shape {tuple(a.shape)}")
        if not 0 < rotary_pairs <= pairs:
            raise ValueError(
                f"rotary_pairs must be in [1, {pairs}], got {rotary_pairs}"
            )
        if nu.shape != (batch, seqlen, nheads, rotary_pairs):
            raise ValueError(f"Unexpected frequency shape {tuple(nu.shape)}")
        if dt.shape != a.shape or trap.shape != a.shape:
            raise ValueError("dt and trap must have the same shape as decay")
        states = torch.empty_like(writes)
        block_p = triton.next_power_of_2(head_dim)
        _second_order_scan_2x2_fwd_kernel[(batch * nheads * pairs,)](
            a,
            nu,
            dt,
            trap,
            writes,
            states,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            PAIRS=pairs,
            ROTARY_PAIRS=rotary_pairs,
            BLOCK_P=block_p,
            MODE=mode,
            num_warps=4,
        )
        ctx.mode = mode
        ctx.rotary_pairs = rotary_pairs
        ctx.save_for_backward(a, nu, dt, trap, writes, states)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        a, nu, dt, trap, writes, states = ctx.saved_tensors
        grad_states = grad_states.float().contiguous()
        batch, seqlen, nheads, head_dim, pairs = writes.shape[:5]
        partial_shape = (batch, seqlen, nheads, pairs)
        grad_a_pair = torch.empty(
            partial_shape,
            device=writes.device,
            dtype=torch.float32,
        )
        grad_nu = torch.empty_like(nu)
        grad_dt_pair = torch.empty_like(grad_a_pair)
        grad_trap_pair = torch.empty_like(grad_a_pair)
        grad_writes = torch.empty_like(writes)
        block_p = triton.next_power_of_2(head_dim)
        _second_order_scan_2x2_bwd_kernel[(batch * nheads * pairs,)](
            a,
            nu,
            dt,
            trap,
            writes,
            states,
            grad_states,
            grad_a_pair,
            grad_nu,
            grad_dt_pair,
            grad_trap_pair,
            grad_writes,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            PAIRS=pairs,
            ROTARY_PAIRS=ctx.rotary_pairs,
            BLOCK_P=block_p,
            MODE=ctx.mode,
            num_warps=4,
        )
        grad_a = grad_a_pair.sum(dim=-1)
        grad_dt = grad_dt_pair.sum(dim=-1)
        grad_trap = grad_trap_pair.sum(dim=-1)
        return grad_a, grad_nu, grad_dt, grad_trap, grad_writes, None, None


class _SecondOrderAssociativeScan2x2(torch.autograd.Function):
    """Full-sequence 2x2 affine scan using Triton's parallel scan primitive."""

    @staticmethod
    def forward(ctx, a, nu, dt, trap, writes, rotary_pairs, mode):
        a = a.float().contiguous()
        nu = nu.float().contiguous()
        dt = dt.float().contiguous()
        trap = trap.float().contiguous()
        writes = writes.float().contiguous()
        batch, seqlen, nheads, head_dim, pairs = writes.shape[:5]
        if head_dim > 128:
            raise ValueError("The associative second-order scan supports headdim <= 128")
        if seqlen > 1024:
            raise ValueError(
                "The full-sequence associative scan currently supports seqlen <= 1024; "
                "use scan_backend='triton' for longer sequences"
            )
        if a.shape != (batch, seqlen, nheads):
            raise ValueError(f"Unexpected decay shape {tuple(a.shape)}")
        if not 0 < rotary_pairs <= pairs:
            raise ValueError(
                f"rotary_pairs must be in [1, {pairs}], got {rotary_pairs}"
            )
        if nu.shape != (batch, seqlen, nheads, rotary_pairs):
            raise ValueError(f"Unexpected frequency shape {tuple(nu.shape)}")
        if dt.shape != a.shape or trap.shape != a.shape:
            raise ValueError("dt and trap must have the same shape as decay")

        states = torch.empty_like(writes)
        block_l = triton.next_power_of_2(seqlen)
        # Eight independent state rows per program amortize transition loads
        # while keeping the scan tile small enough for the current workloads.
        block_p = min(8, triton.next_power_of_2(head_dim))
        grid = (batch * nheads * pairs, triton.cdiv(head_dim, block_p))
        _second_order_associative_fwd_kernel[grid](
            a,
            nu,
            dt,
            trap,
            writes,
            states,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            PAIRS=pairs,
            ROTARY_PAIRS=rotary_pairs,
            BLOCK_L=block_l,
            BLOCK_P=block_p,
            MODE=mode,
            num_warps=8 if block_l * block_p >= 512 else 4,
        )
        ctx.mode = mode
        ctx.rotary_pairs = rotary_pairs
        ctx.save_for_backward(a, nu, dt, trap, writes, states)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        a, nu, dt, trap, writes, states = ctx.saved_tensors
        grad_states = grad_states.float().contiguous()
        batch, seqlen, nheads, head_dim, pairs = writes.shape[:5]
        partial_shape = (batch, seqlen, nheads, pairs)
        grad_a_pair = torch.empty(
            partial_shape, device=writes.device, dtype=torch.float32
        )
        grad_nu = torch.empty_like(nu)
        grad_dt_pair = torch.empty_like(grad_a_pair)
        grad_trap_pair = torch.empty_like(grad_a_pair)
        grad_writes = torch.empty_like(writes)
        # A time-parallel reverse scan performs more matrix compositions than
        # the exact recurrent VJP.  For the state-rich Mamba workload there is
        # already ample parallelism across batch/head/pair, so the fused
        # reverse recurrence is faster while differentiating the identical
        # affine scan.  The forward remains the LinOSS-style prefix scan.
        block_p = triton.next_power_of_2(head_dim)
        _second_order_scan_2x2_bwd_kernel[(batch * nheads * pairs,)](
            a,
            nu,
            dt,
            trap,
            writes,
            states,
            grad_states,
            grad_a_pair,
            grad_nu,
            grad_dt_pair,
            grad_trap_pair,
            grad_writes,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            PAIRS=pairs,
            ROTARY_PAIRS=ctx.rotary_pairs,
            BLOCK_P=block_p,
            MODE=ctx.mode,
            num_warps=4,
        )
        grad_a = grad_a_pair.sum(dim=-1)
        grad_dt = grad_dt_pair.sum(dim=-1)
        grad_trap = grad_trap_pair.sum(dim=-1)
        return grad_a, grad_nu, grad_dt, grad_trap, grad_writes, None, None


class _SecondOrderScanScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, neg_a, dt, trap, writes):
        neg_a = neg_a.float().contiguous()
        dt = dt.float().contiguous()
        trap = trap.float().contiguous()
        writes = writes.float().contiguous()
        batch, seqlen, nheads, head_dim, dstate = writes.shape
        if head_dim > 128:
            raise ValueError("The fused scalar scan supports headdim <= 128")
        if neg_a.shape != (batch, seqlen, nheads):
            raise ValueError(f"Unexpected decay shape {tuple(neg_a.shape)}")
        if dt.shape != neg_a.shape or trap.shape != neg_a.shape:
            raise ValueError("dt and trap must have the same shape as decay")
        states = torch.empty_like(writes)
        block_p = triton.next_power_of_2(head_dim)
        _second_order_scan_scalar_fwd_kernel[(batch * nheads * dstate,)](
            neg_a,
            dt,
            trap,
            writes,
            states,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            DSTATE=dstate,
            BLOCK_P=block_p,
            num_warps=4,
        )
        ctx.save_for_backward(neg_a, dt, trap, writes, states)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        neg_a, dt, trap, writes, states = ctx.saved_tensors
        grad_states = grad_states.float().contiguous()
        batch, seqlen, nheads, head_dim, dstate = writes.shape
        partial_shape = (batch, seqlen, nheads, dstate)
        grad_alpha = torch.empty(partial_shape, device=writes.device, dtype=torch.float32)
        grad_dt_weight = torch.empty_like(grad_alpha)
        grad_trap_state = torch.empty_like(grad_alpha)
        grad_writes = torch.empty_like(writes)
        block_p = triton.next_power_of_2(head_dim)
        _second_order_scan_scalar_bwd_kernel[(batch * nheads * dstate,)](
            neg_a,
            dt,
            trap,
            writes,
            states,
            grad_states,
            grad_alpha,
            grad_dt_weight,
            grad_trap_state,
            grad_writes,
            SEQLEN=seqlen,
            NHEADS=nheads,
            HEAD_DIM=head_dim,
            DSTATE=dstate,
            BLOCK_P=block_p,
            num_warps=4,
        )
        grad_alpha = grad_alpha.sum(dim=-1)
        alpha = torch.exp(neg_a * dt)
        grad_neg_a = grad_alpha * alpha * dt
        grad_dt = grad_alpha * alpha * neg_a + grad_dt_weight.sum(dim=-1)
        grad_trap = grad_trap_state.sum(dim=-1)
        return grad_neg_a, grad_dt, grad_trap, grad_writes


def triton_second_order_scan_2x2(
    a: torch.Tensor,
    nu: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    writes: torch.Tensor,
    transition_mode: str,
    overdamped_rho_scale: float = 1.0,
) -> torch.Tensor:
    """Fused transition, trapezoid-write, and 2x2 recurrent scan."""
    if transition_mode not in _TRANSITION_MODES:
        raise ValueError(f"Unsupported transition mode {transition_mode!r}")
    if not all(tensor.is_cuda for tensor in (a, nu, dt, trap, writes)):
        raise ValueError("The fused second-order scan requires CUDA tensors")
    scan_nu = _scan_frequency(a, nu, transition_mode, overdamped_rho_scale)
    return _SecondOrderScan2x2.apply(
        a,
        scan_nu,
        dt,
        trap,
        writes,
        writes.shape[-2],
        _TRANSITION_MODES[transition_mode],
    )


def triton_second_order_scan_mixed(
    a: torch.Tensor,
    nu: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    writes: torch.Tensor,
    transition_mode: str,
    overdamped_rho_scale: float = 1.0,
) -> torch.Tensor:
    """Fused paired scan with second-order and scalar-diagonal subspaces."""
    if transition_mode not in _TRANSITION_MODES:
        raise ValueError(f"Unsupported transition mode {transition_mode!r}")
    if not all(tensor.is_cuda for tensor in (a, nu, dt, trap, writes)):
        raise ValueError("The fused mixed second-order scan requires CUDA tensors")
    scan_nu = _scan_frequency(a, nu, transition_mode, overdamped_rho_scale)
    return _SecondOrderScan2x2.apply(
        a,
        scan_nu,
        dt,
        trap,
        writes,
        nu.shape[-1],
        _TRANSITION_MODES[transition_mode],
    )


def triton_second_order_scan_mixed_associative(
    a: torch.Tensor,
    nu: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    writes: torch.Tensor,
    transition_mode: str,
    overdamped_rho_scale: float = 1.0,
) -> torch.Tensor:
    """LinOSS-style parallel prefix scan over token-local 2x2 affine maps.

    This preserves the same transition, trapezoidal write, and FP32 state
    equations as :func:`triton_second_order_scan_mixed`; only the scan
    schedule differs.
    """
    if transition_mode not in _TRANSITION_MODES:
        raise ValueError(f"Unsupported transition mode {transition_mode!r}")
    if not all(tensor.is_cuda for tensor in (a, nu, dt, trap, writes)):
        raise ValueError("The associative second-order scan requires CUDA tensors")
    scan_nu = _scan_frequency(a, nu, transition_mode, overdamped_rho_scale)
    return _SecondOrderAssociativeScan2x2.apply(
        a,
        scan_nu,
        dt,
        trap,
        writes,
        nu.shape[-1],
        _TRANSITION_MODES[transition_mode],
    )


def triton_second_order_scan_scalar(
    neg_a: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    writes: torch.Tensor,
) -> torch.Tensor:
    """Fused decay, trapezoid-write, and scalar recurrent scan."""
    if not all(tensor.is_cuda for tensor in (neg_a, dt, trap, writes)):
        raise ValueError("The fused scalar scan requires CUDA tensors")
    return _SecondOrderScanScalar.apply(neg_a, dt, trap, writes)
