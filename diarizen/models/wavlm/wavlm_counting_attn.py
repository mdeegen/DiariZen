import torch
import os
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import numpy as np
import torch.nn as nn
import torch.profiler
from functools import lru_cache
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model
from diarizen.models.module.wavlm_config import get_config
from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames,
    multi_conv_receptive_field_size,
    multi_conv_receptive_field_center
)


class Model(BaseModel):

    def __init__(self,
         chunk_size: int = 8,
         num_channels: int = 8,
         selected_channel: int = 0,
         sample_rate: int = 16000,
         random_channel=False,
         wavlm_frozen = True,
         max_speakers_per_chunk=3,  # silence , 1 ,2 oder mehr als 2 spk
         model_path="/scratch/hpc-prf-nt2/deegen/models/wavlm_base_plus",
         proj_size = 256,
         select_layers = None,
         pruned_model_path = None,
         ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk)
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel
        self.random_channel = random_channel
        self.max_speakers_per_chunk = self.max_num_spk = max_speakers_per_chunk
        self.wavlm_frozen = wavlm_frozen
        self.select_layers = select_layers
        self.pruned_model_path = pruned_model_path

        # TODO: feature extraction im preprocessing machen? BZW WEglassen, ist nur ne normierung und padding
        # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        if not os.path.exists(model_path):
            model_path = "/net/vol/deegen/models/wavlm_base_plus"
        self.wavlm = WavLMModel.from_pretrained(
            model_path,
            output_hidden_states=True
        )

        wavlm_feat_dim = self.wavlm.config.hidden_size
        if isinstance(select_layers, list):
            wavlm_layer_num = len(select_layers)
        elif select_layers is not None:
            if select_layers < 0:
                wavlm_layer_num = -select_layers
            else:
                wavlm_layer_num = select_layers
        else:
            wavlm_layer_num = self.wavlm.config.num_hidden_layers + 1  # 12 + CNN layer
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.proj = nn.Linear(wavlm_feat_dim, proj_size)
        self.lnorm = nn.LayerNorm(proj_size)
        print(f"WavLM loaded from {model_path}")

        # self.classification = nn.Linear(proj_size, self.max_speakers_per_chunk)

        # TODO: 2 self attention heads stattdessen, self attention based calssifier?#
        # TODO: vtl.wavlm vroher extrahieren und hdf5 id:feature:target abspeichern
        # TODO: psotprocessing glätten
        self.attention = nn.MultiheadAttention(
            embed_dim=proj_size,
            num_heads=4,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_size, self.max_speakers_per_chunk)
        )


        # self.activation = self.default_activation()

    def non_wavlm_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.classifier.parameters(),
            *self.attention.parameters(),
            # *self.classifier.parameters(),
        ]

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    @property
    def get_rf_info(self):
        """Return receptive field info to dataset
        """

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        duration = receptive_field_size / self.sample_rate
        step=receptive_field_step / self.sample_rate
        return num_frames, duration, step


    def forward(self, waveforms: torch.Tensor, gccs: torch.Tensor = None, save_features = False) -> torch.Tensor:
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

        # if self.wavlm_frozen:
        with torch.no_grad():
            wavlm_feat = self.wavlm(waveforms)
        # else:
        #     wavlm_feat = self.wavlm(waveforms)

        hidden_states = wavlm_feat.hidden_states
        # TODO: fix path and directiory
        if save_features:
            return hidden_states

        # if self.select_layers is not None:
        #     if self.select_layers < 0:
        #         hidden_states = hidden_states[self.select_layers:]
        #     else:
        #         hidden_states = hidden_states[:self.select_layers]
        #     # hidden_states = [hidden_states[i] for i in self.select_layers]
        if self.select_layers is not None:
            hidden_states = [hidden_states[i - 1] for i in self.select_layers]

        wavlm_feat_stacked = torch.stack(hidden_states, dim=-1)  # (batch, frames, feat_dim, layers)

        # TODO: evtl. softmax damit weights itnerpretierbarer werden?
        feature_layers = self.weight_sum(wavlm_feat_stacked)
        feature_layers = torch.squeeze(feature_layers, -1)

        outputs = self.proj(feature_layers)
        x = self.lnorm(outputs)

        x, _ = self.attention(x, x, x)  # (B,T,D)
        outputs = self.classifier(x)  # (B,T,C)

        return outputs