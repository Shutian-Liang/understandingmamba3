"""LM Evaluation Harness adapter for locally trained Mamba-3 checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch
import transformers
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizerFast

from evaluate.common import ROOT, load_model


class Mamba3HarnessLM(HFLM):
    """Raw-checkpoint adapter matching the paper's causal zero-shot protocol."""

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, checkpoint: str | Path, batch_size: int = 64, max_length: int = 2048, bucket_lengths: bool = False, allow_incomplete: bool = False):
        LM.__init__(self)
        self._model, self.checkpoint_identity = load_model(checkpoint, allow_incomplete)
        self.bucket_lengths = bucket_lengths
        self._device = torch.device("cuda")
        self._config = self._model.config

        tokenizer_file = ROOT / "evaluate/assets/tokenizer/tokenizer.json"
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        self.tokenizer.bos_token = "<|begin_of_text|>"
        self.tokenizer.eos_token = "<|end_of_text|>"
        self.tokenizer.pad_token = "<|end_of_text|>"

        # Mirror HFLM v0.4.3 defaults without asking it to construct an HF model.
        self.vocab_size = self.tokenizer.vocab_size
        self.truncation = False
        self.logits_cache = True
        self.add_bos_token = False
        self._max_length = int(max_length)
        self.batch_size_per_gpu = int(batch_size)
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = int(batch_size)
        self._rank = 0
        self._world_size = 1
        self.custom_prefix_token_id = None
        self.pretrained = str(Path(checkpoint).resolve())
        self.revision = Path(checkpoint).name
        self.delta = None
        self.peft = None

    def _model_call(self, inps, attn_mask=None, labels=None):
        del attn_mask, labels
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.bucket_lengths:
                from evaluate.forward import hidden_forward
                # Causal right padding bounds TileLang sequence shapes. Remove
                # all padding before scoring, preserving the original contexts.
                hidden = hidden_forward(self.model, inps)
                return self.model.lm_head(hidden)
            return self.model(inps).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError("The configured harness tasks use log-likelihood only")
