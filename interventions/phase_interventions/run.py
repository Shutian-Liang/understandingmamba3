from .assignment_interventions import assignment_context
from .phase_controls import phase_context as control_context

CONTROL_CONDITIONS = ("phase_reverse", "zero_angles", "phase_oscillator_shuffle")
ASSIGNMENT_CONDITIONS = ("rope_a_mix_1", "phase_within_token_oscillator_shuffle")
CONDITIONS = CONTROL_CONDITIONS + ASSIGNMENT_CONDITIONS


def intervention(model, condition, sequence_length, seed=1234):
    """Return a context manager for one full-sequence phase intervention."""
    if condition in CONTROL_CONDITIONS:
        return control_context(model, condition, sequence_length, seed)
    if condition in ASSIGNMENT_CONDITIONS:
        return assignment_context(model, condition, sequence_length, seed)
    raise ValueError(f"unknown condition: {condition}; choose from {CONDITIONS}")


__all__ = ["ASSIGNMENT_CONDITIONS", "CONDITIONS", "CONTROL_CONDITIONS", "intervention"]
