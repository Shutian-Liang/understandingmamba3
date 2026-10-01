from __future__ import annotations

from mamba_ssm.ops.interventions.oscillator import OscillatorPhaseMap, WITHIN_TOKEN_OSCILLATOR_SHUFFLE
from .decay_permutation import RoPECoordinateASwap, donor_shifts
CONDITIONS = ("rope_a_mix_1", "phase_within_token_oscillator_shuffle")


def assignment_context(model, condition, sequence_length, seed=1234):
    layers = model.backbone.layers
    if condition == "phase_within_token_oscillator_shuffle":
        return OscillatorPhaseMap(len(layers), WITHIN_TOKEN_OSCILLATOR_SHUFFLE, seed, sequence_length)
    if condition == "rope_a_mix_1":
        heads = layers[0].mixer.nheads
        assert all(layer.mixer.num_rope_angles * 4 == layer.mixer.d_state for layer in layers)
        return RoPECoordinateASwap(model, 1.0, tuple(range(heads)), donor_shifts(len(layers), heads, seed))
    raise ValueError(condition)
