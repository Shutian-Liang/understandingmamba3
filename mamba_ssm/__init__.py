__version__ = "2.3.2.post1"

from importlib import import_module

# Mamba3 uses Triton/TileLang and does not need the Mamba1 selective_scan_cuda
# extension. Load each public entry point only when it is requested.
_EXPORTS = {
    "selective_scan_fn": "mamba_ssm.ops.selective_scan_interface",
    "mamba_inner_fn": "mamba_ssm.ops.selective_scan_interface",
    "Mamba": "mamba_ssm.modules.mamba_simple",
    "Mamba2": "mamba_ssm.modules.mamba2",
    "Mamba3": "mamba_ssm.modules.mamba3",
    "MambaLMHeadModel": "mamba_ssm.models.mixer_seq_simple",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
