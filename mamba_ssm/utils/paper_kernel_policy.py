"""Reproduce the Triton launch choices used by the reported paper results."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from importlib import resources
from pathlib import Path

import triton


LAUNCH_KEYS = frozenset(("num_warps", "num_stages", "num_ctas", "maxnreg"))


def _config(values):
    kernel_arguments = {key: value for key, value in values.items() if key not in LAUNCH_KEYS}
    launch_arguments = {key: value for key, value in values.items() if key in LAUNCH_KEYS}
    return triton.Config(kernel_arguments, **launch_arguments)


def _read_policy(path=None):
    if path is None:
        resource = resources.files("mamba_ssm").joinpath("paper_kernel_policy.json")
        raw = resource.read_bytes()
        source = str(resource)
    else:
        path = Path(path).resolve()
        raw = path.read_bytes()
        source = str(path)
    return json.loads(raw), source, hashlib.sha256(raw).hexdigest()


def apply_paper_kernel_policy(model, path=None):
    """Pin paper-time Triton configs for ``siso`` or ``mimo`` evaluation.

    Call this once after importing the local ``mamba_ssm`` package and before
    the first model forward. It changes launch choices only; model parameters,
    tensors, dtypes, and intervention definitions are untouched.
    """
    policy, source, sha256 = _read_policy(path)
    if policy.get("schema_version") != 1:
        raise ValueError(f"unsupported kernel-policy schema: {policy.get('schema_version')}")
    if model not in policy["models"]:
        raise ValueError(f"unknown model {model!r}; choose from {tuple(policy['models'])}")

    selected = policy["models"][model]
    applied = {}
    for qualified_name, cache in selected["kernels"].items():
        module_name, attribute = qualified_name.rsplit(".", 1)
        kernel = getattr(importlib.import_module(module_name), attribute)
        parsed_cache = {ast.literal_eval(key): _config(value) for key, value in cache.items()}
        kernel.cache = parsed_cache
        unique = {json.dumps(value, sort_keys=True): value for value in cache.values()}
        kernel.configs = [_config(value) for value in unique.values()]
        observed = {repr(key): config.all_kwargs() for key, config in kernel.cache.items()}
        if observed != cache:
            raise RuntimeError(f"failed to pin Triton config for {qualified_name}")
        applied[qualified_name] = observed

    return {
        "model": model,
        "source": source,
        "sha256": sha256,
        "kernels": applied,
        "reference_nll": selected.get("reference_nll", {}),
    }


__all__ = ["apply_paper_kernel_policy"]
