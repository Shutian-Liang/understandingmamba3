from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TaskSpec:
    name: str
    vocab_size: int
    pad_id: int
    answer_classes: int
    default_layers: int
    id_to_token: tuple[str, ...]


@dataclass(frozen=True)
class StateTrackingBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    answer_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    tokens: torch.Tensor


TASK_SPECS = {
    "parity": TaskSpec(
        name="parity",
        vocab_size=3,
        pad_id=0,
        answer_classes=2,
        default_layers=1,
        id_to_token=("[PAD]", "0", "1"),
    ),
    "mod3_counting": TaskSpec(
        name="mod3_counting",
        vocab_size=4,
        pad_id=0,
        answer_classes=3,
        default_layers=1,
        id_to_token=("[PAD]", "0", "1", "2"),
    ),
    "modular_arithmetic_mod3": TaskSpec(
        name="modular_arithmetic_mod3",
        vocab_size=8,
        pad_id=0,
        answer_classes=3,
        default_layers=3,
        id_to_token=("[PAD]", "+", "-", "*", "=", "0", "1", "2"),
    ),
}


def scaled_accuracy(accuracy: float, answer_classes: int) -> float:
    """Map random accuracy to 0 and perfect accuracy to 1.

    This is the paper metric:
    ``(raw_accuracy - 1 / classes) / (1 - 1 / classes)``.
    Consequently, raw 50% maps to scaled 0% for parity and raw 1/3 maps to
    scaled 0% for the two modulo-three tasks.

    Values below random are retained (rather than clipped) so failed runs are
    visible. Multiply by 100 to report percentages.
    """

    random_accuracy = 1.0 / answer_classes
    return (float(accuracy) - random_accuracy) / (1.0 - random_accuracy)


def _make_supervision(
    tokens: np.ndarray,
    answer_indices: np.ndarray,
    lengths: np.ndarray,
    pad_id: int,
) -> StateTrackingBatch:
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError("tokens must have shape (batch_size, context_length >= 2)")
    batch_size, context_length = tokens.shape
    if answer_indices.shape != (batch_size,):
        raise ValueError("each sequence must have one answer index")
    if np.any(answer_indices < 1) or np.any(answer_indices >= context_length):
        raise ValueError("answer indices must be in [1, context_length)")

    rows = np.arange(batch_size, dtype=np.int64)
    masked = tokens.copy()
    masked[rows, answer_indices] = pad_id
    labels = np.full((batch_size, context_length - 1), IGNORE_INDEX, dtype=np.int64)
    output_positions = answer_indices - 1
    labels[rows, output_positions] = tokens[rows, answer_indices]
    return StateTrackingBatch(
        input_ids=torch.from_numpy(masked[:, :-1].astype(np.int64, copy=False)),
        labels=torch.from_numpy(labels),
        answer_positions=torch.from_numpy(output_positions.astype(np.int64, copy=False)),
        sequence_lengths=torch.from_numpy(lengths.astype(np.int64, copy=False)),
        tokens=torch.from_numpy(tokens.astype(np.int64, copy=False)),
    )


def _generate_parity(
    batch_size: int,
    min_sequence_length: int,
    max_sequence_length: int,
    context_length: int,
    rng: np.random.Generator,
) -> StateTrackingBatch:
    spec = TASK_SPECS["parity"]
    lengths = rng.integers(
        min_sequence_length, max_sequence_length + 1, size=batch_size, dtype=np.int64
    )
    tokens = np.full((batch_size, context_length), spec.pad_id, dtype=np.int64)
    answer_indices = lengths - 1

    for row, answer_idx in enumerate(answer_indices):
        # IDs 1 and 2 encode binary values 0 and 1.
        inputs = rng.integers(1, 3, size=int(answer_idx), dtype=np.int64)
        tokens[row, :answer_idx] = inputs
        tokens[row, answer_idx] = int((inputs - 1).sum() % 2) + 1

    return _make_supervision(tokens, answer_indices, lengths, spec.pad_id)


def _generate_mod3_counting(
    batch_size: int,
    min_sequence_length: int,
    max_sequence_length: int,
    context_length: int,
    rng: np.random.Generator,
) -> StateTrackingBatch:
    """Count input ones modulo three, as a direct three-state parity analogue."""

    spec = TASK_SPECS["mod3_counting"]
    lengths = rng.integers(
        min_sequence_length, max_sequence_length + 1, size=batch_size, dtype=np.int64
    )
    tokens = np.full((batch_size, context_length), spec.pad_id, dtype=np.int64)
    answer_indices = lengths - 1

    for row, answer_idx in enumerate(answer_indices):
        # Input IDs 1 and 2 encode binary values 0 and 1. Output ID 3 is used
        # only when the running count has residue two.
        inputs = rng.integers(1, 3, size=int(answer_idx), dtype=np.int64)
        tokens[row, :answer_idx] = inputs
        tokens[row, answer_idx] = int((inputs - 1).sum() % 3) + 1

    return _make_supervision(tokens, answer_indices, lengths, spec.pad_id)


def _evaluate_flat_expression(
    numbers: np.ndarray, operators: np.ndarray, *, modulus: int = 3
) -> int:
    """Evaluate +, -, * with standard multiplication precedence."""

    total = 0
    current_term = int(numbers[0])
    sign = 1
    for operator, number in zip(operators, numbers[1:]):
        number = int(number)
        if operator == 3:  # multiplication
            current_term *= number
        else:
            total += sign * current_term
            current_term = number
            sign = -1 if operator == 2 else 1
    return (total + sign * current_term) % modulus


def _generate_flat_modular_arithmetic(
    *,
    spec: TaskSpec,
    modulus: int,
    batch_size: int,
    min_sequence_length: int,
    max_sequence_length: int,
    context_length: int,
    rng: np.random.Generator,
) -> StateTrackingBatch:
    """Generate unbracketed modular expressions for a specified modulus."""

    # IDs 1:4 are operators, ID 4 is '=', and numeric IDs begin at 5.
    if spec.vocab_size != 5 + modulus or spec.answer_classes != modulus:
        raise ValueError("flat modular-arithmetic TaskSpec is inconsistent with modulus")

    # A valid sequence is: number (op number)* = answer, hence it is odd.
    min_terms = max(1, (min_sequence_length - 1 + 1) // 2)
    max_terms = max(1, (max_sequence_length - 1) // 2)
    if min_terms > max_terms:
        min_terms = max_terms
    term_counts = rng.integers(min_terms, max_terms + 1, size=batch_size, dtype=np.int64)
    lengths = 2 * term_counts + 1
    tokens = np.full((batch_size, context_length), spec.pad_id, dtype=np.int64)
    answer_indices = lengths - 1

    for row, n_terms in enumerate(term_counts):
        n_terms = int(n_terms)
        numbers = rng.integers(0, modulus, size=n_terms, dtype=np.int64)
        operators = rng.integers(1, 4, size=max(0, n_terms - 1), dtype=np.int64)
        expression = np.empty(2 * n_terms - 1, dtype=np.int64)
        expression[0::2] = numbers + 5
        expression[1::2] = operators
        eq_idx = expression.size
        tokens[row, :eq_idx] = expression
        tokens[row, eq_idx] = 4
        tokens[row, eq_idx + 1] = (
            _evaluate_flat_expression(numbers, operators, modulus=modulus) + 5
        )

    return _make_supervision(tokens, answer_indices, lengths, spec.pad_id)


def _generate_modular_arithmetic_mod3(
    batch_size: int,
    min_sequence_length: int,
    max_sequence_length: int,
    context_length: int,
    rng: np.random.Generator,
) -> StateTrackingBatch:
    return _generate_flat_modular_arithmetic(
        spec=TASK_SPECS["modular_arithmetic_mod3"],
        modulus=3,
        batch_size=batch_size,
        min_sequence_length=min_sequence_length,
        max_sequence_length=max_sequence_length,
        context_length=context_length,
        rng=rng,
    )


_GENERATORS = {
    "parity": _generate_parity,
    "mod3_counting": _generate_mod3_counting,
    "modular_arithmetic_mod3": _generate_modular_arithmetic_mod3,
}


def generate_batch(
    task: str,
    *,
    batch_size: int,
    min_sequence_length: int = 3,
    max_sequence_length: int = 40,
    context_length: int = 224,
    seed: int = 42,
) -> StateTrackingBatch:
    """Generate a deterministic online batch for one of the three paper tasks.

    Arithmetic sequences have odd lengths. For an exact even length request,
    the original experiment protocol uses the preceding odd length.
    """
    if task not in _GENERATORS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(_GENERATORS)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if min_sequence_length < 3:
        raise ValueError("min_sequence_length must be at least 3")
    if max_sequence_length < min_sequence_length:
        raise ValueError("max_sequence_length must be >= min_sequence_length")
    if context_length < max_sequence_length:
        raise ValueError("context_length must be >= max_sequence_length")
    return _GENERATORS[task](
        batch_size=batch_size,
        min_sequence_length=min_sequence_length,
        max_sequence_length=max_sequence_length,
        context_length=context_length,
        rng=np.random.default_rng(seed),
    )


def decode_tokens(task: str, token_ids: torch.Tensor | np.ndarray | list[int]) -> list[str]:
    spec = TASK_SPECS[task]
    values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    return [spec.id_to_token[int(token_id)] for token_id in values]
