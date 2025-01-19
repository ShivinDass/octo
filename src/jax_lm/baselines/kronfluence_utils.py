import argparse
import torch as ch
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator

from examples.wikitext.pipeline import construct_gpt2, get_wikitext_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

class LanguageModelingTask(Task):
    def compute_train_loss(self, batch, model, sample=False):
        x, y = batch
        logits = model(x, y)[0]

        shift_logits = logits

        if not sample:
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            summed_loss = F.cross_entropy(reshaped_shift_logits, y.view(-1), reduction="sum")
        else:
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            with torch.no_grad():
                probs = torch.nn.functional.softmax(reshaped_shift_logits, dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()

            summed_loss = F.cross_entropy(reshaped_shift_logits, sampled_labels.detach(), reduction="sum")

        return summed_loss

    # We could also compute the log-likelihood or averaged margin.
    def compute_measurement(self, batch, model):
        return self.compute_train_loss(batch, model)

    def tracked_modules(self) -> List[str]:
        total_modules = []

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")

        return total_modules

    def get_attention_mask(self, batch) -> Optional[torch.Tensor]:
        return ch.ones_like(batch[0])
