from .assignment_interventions import assignment_context
from .phase_controls import phase_context as control_context

CONTROL_CONDITIONS = ("phase_reverse", "zero_angles", "phase_oscillator_shuffle")
ASSIGNMENT_CONDITIONS = ("rope_a_mix_1", "phase_within_token_oscillator_shuffle")
CONDITIONS = CONTROL_CONDITIONS + ASSIGNMENT_CONDITIONS
PAPER_TABLE5_CONDITIONS = {
    "decay_permutation": "rope_a_mix_1",
    "phase_shuffle": "phase_within_token_oscillator_shuffle",
    "phase_removal": "zero_angles",
    "phase_reversal": "phase_reverse",
}


def intervention(model, condition, sequence_length, seed=1234):
    """Return a context manager for one full-sequence phase intervention."""
    if condition in CONTROL_CONDITIONS:
        return control_context(model, condition, sequence_length, seed)
    if condition in ASSIGNMENT_CONDITIONS:
        return assignment_context(model, condition, sequence_length, seed)
    raise ValueError(f"unknown condition: {condition}; choose from {CONDITIONS}")


def paper_table5_intervention(model, condition, sequence_length, seed=1234):
    """Return the implementation used for a named Table 5 intervention."""
    if condition not in PAPER_TABLE5_CONDITIONS:
        raise ValueError(
            f"unknown Table 5 condition: {condition}; "
            f"choose from {tuple(PAPER_TABLE5_CONDITIONS)}"
        )
    return intervention(model, PAPER_TABLE5_CONDITIONS[condition], sequence_length, seed)


__all__ = [
    "ASSIGNMENT_CONDITIONS",
    "CONDITIONS",
    "CONTROL_CONDITIONS",
    "PAPER_TABLE5_CONDITIONS",
    "intervention",
    "paper_table5_intervention",
]
