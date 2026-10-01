from __future__ import annotations

import contextlib
import hashlib
from typing import Any

import torch


def state_indices(is_mimo: bool, d_state: int) -> tuple[torch.Tensor, torch.Tensor]:
    if is_mimo:
        rope = list(range(0, d_state // 4)) + list(range(d_state // 2, 3 * d_state // 4))
    else:
        rope = list(range(0, d_state // 2))
    non_rope = [index for index in range(d_state) if index not in set(rope)]
    return torch.tensor(rope), torch.tensor(non_rope)


def donor_shifts(layers: int, nheads: int, seed: int) -> tuple[int, ...]:
    if nheads < 2:
        raise ValueError("head swapping requires at least two heads")
    shifts = []
    for layer in range(layers):
        digest = hashlib.sha256(f"rope-a-donor:{seed}:{layer}".encode()).digest()
        shifts.append(1 + int.from_bytes(digest[:8], "big") % (nheads - 1))
    return tuple(shifts)


class RoPECoordinateASwap(contextlib.AbstractContextManager):
    """Run RoPE coordinates with donor-head A and all other coordinates natively."""

    def __init__(
        self,
        model,
        mix: float,
        target_heads: tuple[int, ...],
        shifts: tuple[int, ...],
    ) -> None:
        import mamba_ssm.modules.mamba3 as implementation

        self.implementation = implementation
        self.layers = [block.mixer for block in model.backbone.layers]
        self.is_mimo = bool(self.layers[0].is_mimo)
        self.mix = float(mix)
        self.target_heads = target_heads
        self.shifts = shifts
        self.nheads = int(self.layers[0].nheads)
        self.d_state = int(self.layers[0].d_state)
        self.rope, _ = state_indices(self.is_mimo, self.d_state)
        self.seen = [0] * len(self.layers)
        self.original_siso = implementation.mamba3_siso_combined
        self.old_mimo: list[tuple[bool, Any]] = []
        self.max_component_nonrope = 0.0

        if not 0 <= self.mix <= 1:
            raise ValueError("mix must be in [0, 1]")
        if len(shifts) != len(self.layers):
            raise ValueError("one donor shift is required per layer")
        if any(int(layer.nheads) != self.nheads for layer in self.layers):
            raise ValueError("all layers must have the same head count")
        if any(bool(layer.is_outproj_norm) for layer in self.layers):
            raise ValueError("coordinate decomposition requires pre-norm additive readout")

    def _donor_adt(self, layer: int, kwargs: dict[str, Any]) -> torch.Tensor:
        native_adt = kwargs["ADT"]
        dt = kwargs["DT"]
        if native_adt.shape != dt.shape or native_adt.ndim != 3:
            raise ValueError(f"unexpected ADT/DT shapes: {native_adt.shape}/{dt.shape}")
        rates = native_adt / dt
        donor = torch.roll(rates, shifts=-self.shifts[layer], dims=1)
        mixed = rates.clone()
        target = torch.tensor(self.target_heads, device=rates.device)
        mixed[:, target] = (
            (1 - self.mix) * rates.index_select(1, target)
            + self.mix * donor.index_select(1, target)
        )
        return mixed * dt

    def _rope_component(
        self, kwargs: dict[str, Any], adt: torch.Tensor,
    ) -> dict[str, Any]:
        rope = self.rope.to(kwargs["Q"].device)
        mask = torch.zeros(self.d_state, device=kwargs["Q"].device, dtype=kwargs["Q"].dtype)
        mask.index_fill_(0, rope, 1)
        component = dict(kwargs)
        component["Q"] = kwargs["Q"] * mask
        component["K"] = kwargs["K"] * mask
        component["Q_bias"] = kwargs["Q_bias"] * mask.to(kwargs["Q_bias"].dtype)
        component["K_bias"] = kwargs["K_bias"] * mask.to(kwargs["K_bias"].dtype)
        component["ADT"] = adt
        # D skip has no d_state decomposition and remains only in the full pass.
        component["D"] = None
        return component

    def _replace(self, layer: int, original, kwargs: dict[str, Any]):
        donor_adt = self._donor_adt(layer, kwargs)
        full = original(**kwargs)
        native_rope = original(**self._rope_component(kwargs, kwargs["ADT"]))
        donor_rope = original(**self._rope_component(kwargs, donor_adt))

        has_state = isinstance(full, tuple)
        full_y = full[0] if has_state else full
        native_y = native_rope[0] if has_state else native_rope
        donor_y = donor_rope[0] if has_state else donor_rope
        y = (full_y.float() + donor_y.float() - native_y.float()).to(full_y.dtype)
        if not has_state:
            return y

        full_state = full[2]
        native_state = native_rope[2]
        donor_state = donor_rope[2]
        state = (
            full_state.float() + donor_state.float() - native_state.float()
        ).to(full_state.dtype)
        _, nonrope = state_indices(self.is_mimo, self.d_state)
        nonrope = nonrope.to(state.device)
        leakage = max(
            float(native_state.index_select(-1, nonrope).abs().max().cpu()),
            float(donor_state.index_select(-1, nonrope).abs().max().cpu()),
        )
        self.max_component_nonrope = max(self.max_component_nonrope, leakage)
        # Phase, current K/B cache, and V cache are independent of radial A.
        return (y, full[1], state, full[3], full[4], *full[5:])

    def __enter__(self):
        if self.is_mimo:
            for layer_index, mixer in enumerate(self.layers):
                had = hasattr(mixer, "mimo_train_fn")
                original = getattr(
                    mixer, "mimo_train_fn", self.implementation.mamba3_mimo_combined,
                )
                self.old_mimo.append((had, original))

                def wrapper(*, _layer=layer_index, _original=original, **kwargs):
                    self.seen[_layer] += 1
                    return self._replace(_layer, _original, kwargs)

                mixer.mimo_train_fn = wrapper
            return self

        call_index = 0

        def wrapper(**kwargs):
            nonlocal call_index
            layer = call_index
            call_index += 1
            if layer >= len(self.layers):
                raise RuntimeError("SISO kernel called too many times")
            self.seen[layer] += 1
            return self._replace(layer, self.original_siso, kwargs)

        self.implementation.mamba3_siso_combined = wrapper
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.is_mimo:
            for mixer, (had, original) in zip(self.layers, self.old_mimo):
                if had:
                    mixer.mimo_train_fn = original
                else:
                    delattr(mixer, "mimo_train_fn")
        else:
            self.implementation.mamba3_siso_combined = self.original_siso
        if exc_type is None:
            if self.seen != [1] * len(self.layers):
                raise RuntimeError(f"expected one kernel call/layer, saw {self.seen}")
            if self.max_component_nonrope != 0.0:
                raise AssertionError(
                    f"RoPE-only component leaked into non-RoPE state: {self.max_component_nonrope}"
                )
        return False
