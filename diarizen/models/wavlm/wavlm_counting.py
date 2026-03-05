import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import os

import numpy as np
import torch.nn as nn
import torch.profiler
from functools import lru_cache
import paderbox as pb
import dlp_mpi
import math

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames,
    multi_conv_receptive_field_size,
    multi_conv_receptive_field_center
)


class Model(BaseModel):

    def __init__(self,
         max_speakers_per_chunk: int = 4,
         chunk_size: int = 8,
         num_channels: int = 8,
         selected_channel: int = 0,
         sample_rate: int = 16000,
         random_channel=False,
         wavlm_frozen = True,
         max_num_spk=3,  # silence , 1 ,2 oder mehr als 2 spk
         model_path="/net/vol/deegen/models/wavlm_base_plus",
         proj_size = 256,

         ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk)
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel
        self.random_channel = random_channel
        self.max_num_spk = max_num_spk
        self.wavlm_frozen = wavlm_frozen

        # TODO: feature extraction im preprocessing machen? BZW WEglassen, ist nur ne normierung und padding
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        # TODO: Optimizer und parameter machen, loss
        self.wavlm = WavLMModel.from_pretrained(
            model_path,
            output_hidden_states=True
        )
        # TODO: in trainigns schleife einbauen statt hier
        # model.train()
        # model.wavlm.eval()
        self.wavlm.eval()

        wavlm_feat_dim = self.wavlm.config.hidden_size
        wavlm_layer_num = self.wavlm.config.num_hidden_layers + 1  # 12 + CNN layer
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.proj = nn.Linear(wavlm_feat_dim, proj_size)
        self.lnorm = nn.LayerNorm(proj_size)
        print(f"WavLM loaded from {model_path}")

        # self.classification = nn.Linear(proj_size, self.max_num_spk)

        self.classifier = nn.Sequential(
            nn.Linear(proj_size, proj_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_size, self.max_num_spk)
        )


    def forward(self, waveforms: torch.Tensor):
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, [channel], sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        if waveforms.dim() == 3:
            if self.random_channel:
                selected_channel = np.random.randint(0, waveforms.size(1))
            else:
                selected_channel = self.selected_channel
            waveforms = waveforms[:, selected_channel, :]
        assert waveforms.dim() == 2

        if self.wavlm_frozen:
            with torch.no_grad():
                wavlm_feat = self.wavlm(waveforms)
        else:
            wavlm_feat = self.wavlm(waveforms)

        hidden_states = wavlm_feat.hidden_states
        wavlm_feat_stacked = torch.stack(hidden_states, dim=-1)  # (batch, frames, feat_dim, layers)

        # TODO: evtl. softmax damit weights itnerpretierbarer werden?
        feature_layers = self.weight_sum(wavlm_feat_stacked)
        feature_layers = torch.squeeze(feature_layers, -1)

        outputs = self.proj(feature_layers)
        outputs = self.lnorm(outputs)

        outputs = self.classification(outputs)


        return outputs