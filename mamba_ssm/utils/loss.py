"""Exact token-mean LM loss without materializing the full sequence logits."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_cross_entropy(hidden_states, weight, targets, chunk_size):
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    labels = targets.reshape(-1)
    if labels.numel() != hidden.shape[0] or labels.numel() == 0:
        raise ValueError("targets must match the nonempty hidden token dimensions")
    if chunk_size < 1:
        raise ValueError("loss chunk size must be positive")

    def summed_loss(features, projection, target):
        return F.cross_entropy(F.linear(features, projection).float(), target, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for begin in range(0, labels.numel(), chunk_size):
        features = hidden[begin : begin + chunk_size]
        target = labels[begin : begin + chunk_size]
        if torch.is_grad_enabled():
            # Recompute each chunk during backward, retaining only hidden states
            # and labels instead of vocab-sized softmax buffers for every token.
            value = checkpoint(summed_loss, features, weight, target, use_reentrant=False)
        else:
            value = summed_loss(features, weight, target)
        total = total + value
    return total / labels.numel()
