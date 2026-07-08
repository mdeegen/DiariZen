import math

import torch

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from diarizen.models.module.conformer import BLSTM, BLSTM_FILM

class diarization_head(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.classifier = nn.Linear(emb_dim, out_dim)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.classifier(x)
        outputs = self.activation(x)
        return outputs


class asr_head(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.classifier = nn.Linear(emb_dim, out_dim)
        self.activation = ...# TODO

    def forward(self, x):
        x = self.classifier(x)
        outputs = self.activation(x)
        return outputs

    # TODO: ASR und SE heads vtl mit task specific decoder und Zeitauflösung , waveform oder masken bei SE?


        # task heads
        self.diar_head = nn.Linear(attention_in, diar_out_dim)

        # ASR: either CTC logits or encoder output for a separate decoder
        self.asr_ctc_head = nn.Linear(attention_in, asr_vocab_size)

        # SE: usually mask or spectral mapping head; adapt output dim to your target
        self.se_head = nn.Linear(attention_in, se_out_dim if se_out_dim > 0 else attention_in)

        # optional task routing / weighting for multi-task training
        self.task_loss_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=False)
