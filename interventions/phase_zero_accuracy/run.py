from contextlib import AbstractContextManager
import importlib

import torch


class ZeroAngles(AbstractContextManager):
    """Zero raw phase projections in selected Mamba-3 layers for one forward pass."""

    def __init__(self, model, layers):
        self.num_layers = len(model.backbone.layers)
        self.layers = frozenset(layers)
        if not self.layers <= set(range(self.num_layers)):
            raise ValueError(f"layers must be within [0, {self.num_layers})")
        mixer = model.backbone.layers[0].mixer
        module = (
            "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo"
            if mixer.is_mimo
            else "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined"
        )
        self.kernel = importlib.import_module(module)
        self.original = self.kernel.angle_dt_fwd
        self.calls = 0

    def _forward(self, angle, dt, *args, **kwargs):
        layer = self.calls
        self.calls += 1
        if layer >= self.num_layers:
            raise RuntimeError("angle_dt_fwd was called too many times")
        if layer in self.layers:
            angle = torch.zeros_like(angle)
        return self.original(angle, dt, *args, **kwargs)

    def __enter__(self):
        self.kernel.angle_dt_fwd = self._forward
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.kernel.angle_dt_fwd = self.original
        if exc_type is None and self.calls != self.num_layers:
            raise RuntimeError(
                f"expected {self.num_layers} phase calls, observed {self.calls}"
            )
        return False


__all__ = ["ZeroAngles"]
