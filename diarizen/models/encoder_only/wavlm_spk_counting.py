#!/usr/bin/env python3

import os

import numpy as np
import torch
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

from diarizen.models.module.conformer import ConformerEncoder, gcc_encoder, init_as_identity
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model
from diarizen.models.module.wavlm_config import get_config
from diarizen.spatial_features.gcc_phat import (get_gcc_for_all_channel_pairs, channel_wise_activities,
                                                convert_to_frame_wise_activities, get_dominant_time_frequency_mask)


class Model(BaseModel):
    def __init__(
        self,
        wavlm_src: str = "wavlm_base_s80_md",
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        attention_in: int = 256,
        attention_in_aux: int = 200,  # search range for delay
        linear_input_size: int = 399,  # number of frames per channel
        linear_output_size: int = 399,  # number of frames per channel
        num_head_aux: int = 4,
        num_layer_aux: int = 3,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        max_speakers_per_chunk: int = 4,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        sample_rate: int = 16000,
        random_channel = False,
        max_num_spk = 4, # silence , 1 ,2 oder mehr als 2 spk
        sin_cos = False,
        ffn = None,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        self.max_num_spk = max_num_spk
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel
        self.random_channel = random_channel

        # wavlm
        self.wavlm_model = self.load_wavlm(wavlm_src)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.proj = nn.Linear(wavlm_feat_dim, attention_in)
        self.lnorm = nn.LayerNorm(attention_in)
        print(f"WavLM loaded from {wavlm_src}")

        self.classification = nn.Linear(attention_in, self.max_num_spk)
        self.activation = self.default_activation()

    def non_wavlm_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.classification.parameters(),
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

    def sin_cos_representation(self, gcc_features):
        " gets the sin and cos representations of the phase from the complex valued input"
        magnitude = torch.real(gcc_features[:, :, -1, :])
        gcc_features = gcc_features[:, :, :-1, :]  # remove magnitude
        phase = torch.angle(gcc_features)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        gcc_features = torch.concatenate((sin_phase, cos_phase, magnitude[:,:,None,:]), dim=2)  # (batch, frames, 2*channels, freq)
        # pad_size = (0, 3)  # pad 3 zeros at the end to make divisible for att heads
        # gcc_features = torch.nn.functional.pad(gcc_features, pad_size)
        return gcc_features

    def load_wavlm(self, source: str):
        """
        Load a WavLM model from either a config name or a checkpoint file.

        Parameters
        ----------
        source : str
            - If `source` is a config name (e.g., "wavlm_large_md_s80"),
            the model will be initialized using predefined configuration via `get_config()`.
            - If `source` is a file path (e.g., "pytorch_model.bin", "model.ckpt", or any local .pt file),
            the model will be loaded from the checkpoint, using its saved 'config' and 'state_dict'.

        Returns
        -------
        model : nn.Module
            Initialized WavLM model.
        """
        if os.path.isfile(source):
            # Load from checkpoint file
            ckpt = torch.load(source, map_location="cpu")

            if "config" not in ckpt or "state_dict" not in ckpt:
                raise ValueError("Checkpoint must contain 'config' and 'state_dict'.")

            for k, v in ckpt["config"].items():
                if 'prune' in k and v is not False:
                    raise ValueError(f"Pruning must be disabled. Found: {k}={v}")

            model = wavlm_model(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"], strict=False)

        else:
            # Load from predefined config
            config = get_config(source)
            model = wavlm_model(**config)

        return model


    def wav2wavlm(self, in_wav, model):
        """
        transform wav to wavlm features
        """
        layer_reps, _ = model.extract_features(in_wav)
        return torch.stack(layer_reps, dim=-1)

    def forward(self, waveforms: torch.Tensor, gcc_features=None) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        gccs : (batch, frames, channels, delays)

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
        # waveforms = waveforms[:, self.selected_channel, :]

        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)
        wavlm_feat = self.weight_sum(wavlm_feat)
        wavlm_feat = torch.squeeze(wavlm_feat, -1)

        outputs = self.proj(wavlm_feat)
        outputs = self.lnorm(outputs)

        outputs = self.classification(outputs)
        return outputs
