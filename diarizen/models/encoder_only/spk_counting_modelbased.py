#!/usr/bin/env python3

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from functools import lru_cache
import paderbox as pb
import dlp_mpi

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
        wavlm_src: str = "wavlm_base",
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

    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        self.max_num_spk = max_num_spk
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        self.linear_input_size = linear_input_size
        self.linear = nn.Linear(linear_input_size, linear_output_size)

    def non_wavlm_parameters(self):
        return [
            *self.linear.parameters(),
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

    def compute_gcc(self, waveforms_mc, frame_size_gcc=400, frame_shift_gcc=320, avg_len_gcc=4, apply_ifft=True,
                    search_range_gcc=10, f_max_gcc=3500, f_min=125, ths=[], modelbased=False):
        """
        Compute GCC features from multichannel waveforms.
        returns:
            batch_gcc_features: (batch, frame, channel, channel, search_range)
        """
        # TODO: try different stft values for better gcc but need fit frames of WAVLM
        sigs_stft = pb.transform.stft(waveforms_mc, frame_size_gcc, frame_shift_gcc,
                                      pad=False, fading=False)
        # print("STFT shape:", sigs_stft.shape)  # c, t, f
        voice_activity = channel_wise_activities(waveforms_mc, ths=ths)
        frame_wise_voice_activity = convert_to_frame_wise_activities(
            voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
        )
        dominant = get_dominant_time_frequency_mask(sigs_stft)
        gcc_features = get_gcc_for_all_channel_pairs(
            sigs_stft, frame_wise_voice_activity, dominant=dominant, search_range=search_range_gcc, f_min=f_min,
            f_max=f_max_gcc, avg_len=avg_len_gcc, shift=frame_shift_gcc, modelbased=modelbased, apply_ifft=apply_ifft,
        )
        return gcc_features

    def forward(self, waveforms: torch.Tensor, gcc_features=None) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, sample) or (batch, channel, sample)
        gccs : (batch, frames, channels, delays)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        gcc_features = F.one_hot(gcc_features.long(), self.max_num_spk)  # (B, T, num_classes)
        gcc_features = gcc_features.float()

        return gcc_features
