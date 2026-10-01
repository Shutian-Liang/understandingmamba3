"""Bound JIT shape count without adding scored tokens or effective context."""


def hidden_forward(model, ids, inference_params=None):
    import torch.nn.functional as F
    if inference_params is not None:
        # Generation uses the original dense prefill to preserve its exact
        # terminal state. Do not pad tokens into the recurrent cache.
        return model.backbone(ids, inference_params=inference_params)
    actual = ids.shape[1]
    padded = max(128, 1 << (actual-1).bit_length())
    ids = F.pad(ids, (0, padded-actual), value=128001)
    return model.backbone(ids)[:, :actual]
