import contextlib
import functools
import importlib

import triton


FLAGS = {
    "baseline": (True, True),
    "b_only": (True, False),
    "c_only": (False, True),
    "both": (False, False),
}


def predictor_bounds(context, targets=100):
    if not context > targets > 0:
        raise ValueError("context must be larger than the positive target count")
    return context - targets - 1, context - 1


def _configure_siso(kernel):
    kernel.configs = [triton.Config({}, num_warps=8, num_ctas=1, num_stages=3)]


class BCPhase(contextlib.AbstractContextManager):
    """Remove the B/write phase, C/read phase, or both for one forward pass."""

    def __init__(self, model_name, condition, expected_layers=24):
        if model_name not in ("siso", "mimo") or condition not in FLAGS:
            raise ValueError((model_name, condition))
        self.model_name = model_name
        self.condition = condition
        self.expected_layers = expected_layers
        self.calls = 0

    def __enter__(self):
        rotate_q, rotate_k = FLAGS[self.condition]
        if self.model_name == "siso":
            from mamba_ssm.ops.interventions import siso_fwd as kernel

            _configure_siso(kernel.mamba3_siso_fwd_kernel)
            self.module = importlib.import_module(
                "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined"
            )
            self.attribute = "mamba3_siso_fwd"
            function = kernel.mamba3_siso_fwd
        else:
            from mamba_ssm.ops.interventions import mimo_fwd as kernel

            self.module = importlib.import_module("mamba_ssm.ops.tilelang.mamba3.mamba3_mimo")
            self.attribute = "mamba_mimo_forward"
            function = kernel.mamba_mimo_forward
        self.original = getattr(self.module, self.attribute)

        @functools.wraps(function)
        def forward(*args, **kwargs):
            self.calls += 1
            return function(*args, **kwargs, rotate_q=rotate_q, rotate_k=rotate_k)

        setattr(self.module, self.attribute, forward)
        return self

    def __exit__(self, exc_type, exc, traceback):
        setattr(self.module, self.attribute, self.original)
        if exc_type is None and self.calls != self.expected_layers:
            raise RuntimeError(f"expected {self.expected_layers} layer calls, saw {self.calls}")
        return False


class TailBCPhase(BCPhase):
    """Apply B/C phase removal only inside a predictor interval."""

    def __init__(self, model_name, condition, phase_start, phase_end, expected_layers=24):
        super().__init__(model_name, condition, expected_layers)
        if not 0 <= phase_start <= phase_end:
            raise ValueError((phase_start, phase_end))
        self.phase_start = phase_start
        self.phase_end = phase_end

    def __enter__(self):
        rotate_q, rotate_k = FLAGS[self.condition]
        if self.model_name == "siso":
            from mamba_ssm.ops.interventions import tail_siso_fwd as kernel

            _configure_siso(kernel.mamba3_siso_fwd_kernel)
            self.module = importlib.import_module(
                "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined"
            )
            self.attribute = "mamba3_siso_fwd"
            function = kernel.mamba3_siso_fwd
        else:
            from mamba_ssm.ops.interventions import tail_mimo_fwd as kernel

            self.module = importlib.import_module("mamba_ssm.ops.tilelang.mamba3.mamba3_mimo")
            self.attribute = "mamba_mimo_forward"
            function = kernel.mamba_mimo_forward
        self.original = getattr(self.module, self.attribute)

        @functools.wraps(function)
        def forward(*args, **kwargs):
            self.calls += 1
            if self.model_name == "siso" and self.condition == "both":
                values = list(args)
                angles = values[8].clone()
                angles[:, self.phase_start:self.phase_end] = 0
                values[8] = angles
                return self.original(*values, **kwargs)
            return function(
                *args,
                **kwargs,
                rotate_q=rotate_q,
                rotate_k=rotate_k,
                phase_start=self.phase_start,
                phase_end=self.phase_end,
            )

        setattr(self.module, self.attribute, forward)
        return self


__all__ = ["BCPhase", "TailBCPhase", "predictor_bounds"]
