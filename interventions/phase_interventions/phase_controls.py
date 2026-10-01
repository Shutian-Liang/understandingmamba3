from __future__ import annotations

import numpy as np
import torch

from .phase_interventions import KernelMap, PhaseOrderMap

CONDITIONS = ("phase_reverse", "zero_angles", "phase_oscillator_shuffle")


def oscillator_permutations(pair_counts, seed):
    generator = np.random.default_rng(seed)
    return [generator.permutation(count).tolist() for count in pair_counts]


def phase_context(model, condition, sequence_length, seed=1234, sanity_registry=None):
    layers = model.backbone.layers
    if condition == "phase_reverse":
        return PhaseOrderMap(len(layers), "reverse", seed, sequence_length, sanity_registry)
    if condition == "zero_angles":
        return KernelMap(model, lambda layer, kwargs: {
            **kwargs, "Angles": torch.zeros_like(kwargs["Angles"]),
        })
    if condition != "phase_oscillator_shuffle":
        raise ValueError(condition)
    permutations = oscillator_permutations([layer.mixer.num_rope_angles for layer in layers], seed)
    registry = sanity_registry if sanity_registry is not None else set()

    def transform(layer, kwargs):
        angle = kwargs["Angles"]
        permutation = torch.tensor(permutations[layer], device=angle.device, dtype=torch.long)
        if angle.stride(2) == 0:
            shuffled = angle[:, :, :1, :].index_select(-1, permutation).expand_as(angle)
        else:
            shuffled = angle.index_select(-1, permutation)
        check = (condition, sequence_length, layer)
        if check not in registry:
            identity = torch.arange(angle.shape[-1], device=angle.device)
            assert torch.equal(permutation.sort().values, identity), "not a bijection"
            assert not torch.equal(permutation, identity), "identity oscillator shuffle"
            inverse = permutation.argsort()
            original_check = angle[:, :, :1] if angle.stride(2) == 0 else angle
            shuffled_check = shuffled[:, :, :1] if angle.stride(2) == 0 else shuffled
            assert torch.equal(shuffled_check.index_select(-1, inverse), original_check)
            registry.add(check)
        return {**kwargs, "Angles": shuffled}

    mapping = KernelMap(model, transform)
    mapping.oscillator_permutations = permutations
    return mapping
