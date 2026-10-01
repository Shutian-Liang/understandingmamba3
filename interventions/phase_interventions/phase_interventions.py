from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
from typing import Any, Callable

import torch

CONDITIONS = ("phase_shuffle", "zero_angles")


class KernelMap(contextlib.AbstractContextManager):
    """Map kernel kwargs once per Mamba-3 layer for one full-sequence forward."""

    def __init__(self, model, transform: Callable[[int, dict[str, Any]], dict[str, Any]]) -> None:
        import mamba_ssm.modules.mamba3 as implementation

        self.implementation = implementation
        self.layers = [block.mixer for block in model.backbone.layers]
        self.is_mimo = bool(self.layers[0].is_mimo)
        self.transform = transform
        self.seen = [0] * len(self.layers)
        self.call_index = 0
        self.original_siso = implementation.mamba3_siso_combined
        self.old_mimo: list[tuple[bool, Any]] = []

    def __enter__(self):
        if self.is_mimo:
            for layer_index, mixer in enumerate(self.layers):
                had = hasattr(mixer, "mimo_train_fn")
                original = getattr(mixer, "mimo_train_fn", self.implementation.mamba3_mimo_combined)
                self.old_mimo.append((had, original))

                def wrapper(*, _layer=layer_index, _original=original, **kwargs):
                    self.seen[_layer] += 1
                    return _original(**self.transform(_layer, kwargs))

                mixer.mimo_train_fn = wrapper
            return self

        def wrapper(**kwargs):
            layer = self.call_index
            self.call_index += 1
            if layer >= len(self.layers):
                raise RuntimeError("SISO kernel called more than once per layer")
            self.seen[layer] += 1
            return self.original_siso(**self.transform(layer, kwargs))

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
        if exc_type is None and self.seen != [1] * len(self.layers):
            raise RuntimeError(f"expected one kernel call/layer, saw {self.seen}")
        return False


class PhaseScaleMap(contextlib.AbstractContextManager):
    """Scale only the actual phase increment seen by ``angle_dt_fwd``.

    Mamba-3 computes ``delta_phi = pi * tanh(raw_angle) * DT``.  Passing a
    scaled *view* of DT to the angle accumulator makes delta_phi exactly
    ``scale`` times the native increment while the recurrence kernel continues
    to receive the original DT/ADT tensors.  This avoids the nonlinear and
    approximate error from multiplying the pre-tanh raw Angle projection.
    """

    MODULES = (
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
        "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",
    )

    def __init__(self, expected_layers: int, scale: float) -> None:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(f"phase scale must be finite and non-negative, got {scale}")
        self.expected_layers = int(expected_layers)
        self.scale = float(scale)
        self.modules: list[Any] = []
        self.originals: list[Callable[..., Any]] = []
        self.calls = 0

    def __enter__(self) -> "PhaseScaleMap":
        for name in self.MODULES:
            module = importlib.import_module(name)
            original = module.angle_dt_fwd
            self.modules.append(module)
            self.originals.append(original)

            def wrapper(angle, dt, *args, _original=original, **kwargs):
                self.calls += 1
                # Do not mutate DT: the recurrent scan still consumes the
                # original tensor through its own kernel argument.
                phase_dt = dt * self.scale
                return _original(angle, phase_dt, *args, **kwargs)

            module.angle_dt_fwd = wrapper
        return self

    def __exit__(self, exc_type, exc, traceback):
        for module, original in zip(self.modules, self.originals):
            module.angle_dt_fwd = original
        if exc_type is None and self.calls != self.expected_layers:
            raise RuntimeError(
                f"expected one angle_dt_fwd call/layer, saw {self.calls} "
                f"for {self.expected_layers} layers"
            )
        return False


class PhaseOrderMap(contextlib.AbstractContextManager):
    """Change only the temporal order/sign of actual phase increments.

    ``angle_dt_fwd`` receives the complete token sequence.  Shuffling raw Angle
    and its matching phase-DT with the same token permutation makes

        pi * tanh(Angle'[t]) * phase_DT'[t]

    an exact permutation of the native increments.  The recurrent scan still
    receives the original DT/ADT tensors through its separate arguments.
    Reversal passes ``-DT`` only to the angle accumulator, hence every native
    increment changes sign without changing its magnitude.
    """

    MODULES = PhaseScaleMap.MODULES

    def __init__(
        self, expected_layers: int, mode: str, seed: int, sequence_length: int,
        sanity_registry: set[tuple[str, int, int]] | None = None,
    ) -> None:
        if mode not in {"shuffle", "reverse"}:
            raise ValueError(mode)
        self.expected_layers = int(expected_layers)
        self.mode = mode
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.sanity_registry = sanity_registry if sanity_registry is not None else set()
        self.modules: list[Any] = []
        self.originals: list[Callable[..., Any]] = []
        self.permutations: dict[tuple[int, str], torch.Tensor] = {}
        self.calls = 0

    def permutation(self, layer: int, device: torch.device) -> torch.Tensor:
        key = (layer, str(device))
        if key not in self.permutations:
            digest = hashlib.sha256(
                f"mamba3-phase-shuffle:{self.seed}:{self.sequence_length}:{layer}".encode()
            ).digest()
            generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big"))
            self.permutations[key] = torch.randperm(
                self.sequence_length, generator=generator, device="cpu",
            ).to(device)
        return self.permutations[key]

    def _check_shuffle(
        self, layer: int, angle: torch.Tensor, dt: torch.Tensor,
        shuffled_angle: torch.Tensor, shuffled_dt: torch.Tensor,
        permutation: torch.Tensor,
    ) -> None:
        key = (self.mode, self.sequence_length, layer)
        if key in self.sanity_registry:
            return
        identity = torch.arange(self.sequence_length, device=permutation.device)
        if torch.equal(permutation, identity):
            raise AssertionError("phase-shuffle permutation is identity")
        inverse = torch.empty_like(permutation)
        inverse[permutation] = identity
        if not torch.equal(shuffled_angle.index_select(1, inverse), angle):
            raise AssertionError("phase shuffle does not preserve the exact raw-Angle multiset")
        if not torch.equal(shuffled_dt.index_select(2, inverse), dt):
            raise AssertionError("phase shuffle does not preserve matching phase-DT values")
        self.sanity_registry.add(key)

    def __enter__(self) -> "PhaseOrderMap":
        for name in self.MODULES:
            module = importlib.import_module(name)
            original = module.angle_dt_fwd
            self.modules.append(module)
            self.originals.append(original)

            def wrapper(angle, dt, *args, _original=original, **kwargs):
                layer = self.calls
                self.calls += 1
                if layer >= self.expected_layers:
                    raise RuntimeError("angle_dt_fwd called more than once per layer")
                if angle.shape[1] != self.sequence_length or dt.shape[2] != self.sequence_length:
                    raise ValueError(
                        f"phase-order length mismatch: angle={angle.shape}, dt={dt.shape}, "
                        f"expected={self.sequence_length}"
                    )
                if self.mode == "reverse":
                    return _original(angle, -dt, *args, **kwargs)
                permutation = self.permutation(layer, angle.device)
                shuffled_angle = angle.index_select(1, permutation)
                shuffled_dt = dt.index_select(2, permutation)
                self._check_shuffle(
                    layer, angle, dt, shuffled_angle, shuffled_dt, permutation,
                )
                return _original(shuffled_angle, shuffled_dt, *args, **kwargs)

            module.angle_dt_fwd = wrapper
        return self

    def __exit__(self, exc_type, exc, traceback):
        for module, original in zip(self.modules, self.originals):
            module.angle_dt_fwd = original
        if exc_type is None and self.calls != self.expected_layers:
            raise RuntimeError(
                f"expected one angle_dt_fwd call/layer, saw {self.calls} "
                f"for {self.expected_layers} layers"
            )
        return False


def phase_context(model, condition, sequence_length, seed=1234, sanity_registry=None):
    """Apply one phase intervention during a full-sequence forward pass."""
    if condition == "phase_shuffle":
        return PhaseOrderMap(len(model.backbone.layers), "shuffle", seed,
                             sequence_length, sanity_registry)
    if condition == "zero_angles":
        return KernelMap(model, lambda layer, kwargs: {
            **kwargs, "Angles": torch.zeros_like(kwargs["Angles"]),
        })
    raise ValueError(condition)
