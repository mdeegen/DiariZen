#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)


import os

import numpy as np
import torch
import torch.nn as nn
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
        multichannel_processing=False,
        random_channel = False,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel
        self.random_channel = random_channel
        self.multichannel_processing=multichannel_processing

        # wavlm 
        self.wavlm_model = self.load_wavlm(wavlm_src)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.proj = nn.Linear(wavlm_feat_dim, attention_in)
        self.lnorm = nn.LayerNorm(attention_in)

        self.conformer = ConformerEncoder(
            attention_in=attention_in + attention_in_aux,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            kernel_size=kernel_size,
            dropout=dropout,
            use_posi=use_posi,
            output_activate_function=output_activate_function
        )
        self.gcc_encoder = gcc_encoder(
            attention_in_aux=attention_in_aux,  # search range for delay
            linear_input_size=linear_input_size,  # number of frames per channel
            linear_output_size=linear_output_size,  # number of frames per channel
            num_head_aux=num_head_aux,
            num_layer_aux=num_layer_aux,
            dropout=dropout,
            )

        # self.norm_wavLM = nn.LayerNorm(attention_in)
        # self.norm_gcc = nn.LayerNorm(attention_in_aux)

        # self.merged_linear = nn.Linear(attention_in+attention_in_aux, attention_in)
        # # todo: vtl eye plus noise init? bzw eye und noise für aux
        # init_as_identity(self.merged_linear, noisy=True) # takes smaller dim, so identity for attention in and rest is zeros

        # self.proj_gcc = nn.Linear(attention_in_aux, attention_in)

        self.classifier = nn.Linear(attention_in + attention_in_aux, self.dimension)
        self.activation = self.default_activation()

    def non_wavlm_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.conformer.parameters(),
            *self.gcc_encoder.parameters(),
            *self.classifier.parameters(),
            # *self.merged_linear.parameters(),
            # *self.norm_wavLM.parameters(),
            # *self.norm_gcc.parameters(),
            # *self.proj_gcc.parameters(),
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

    # def compute_gcc(self, waveforms_mc, frame_size_gcc=4096, frame_shift_gcc=1024, avg_len_gcc=4, search_range_gcc=10,
    #                 f_max_gcc=3500, f_min=125,):
    def compute_gcc(self, waveforms_mc, frame_size_gcc=400, frame_shift_gcc=320, avg_len_gcc=4,
                    search_range_gcc=10, f_max_gcc=3500, f_min=125, ths=[]):
        """
        Compute GCC features from multichannel waveforms.
        returns:
            batch_gcc_features: (batch, frame, channel, channel, search_range)
        """
        batch_gcc_features = []
        for sigs in waveforms_mc:   # dlp_mpi.split_managed(waveforms_mc):
            # TODO: try different stft values for better gcc but need fit frames of WAVLM
            sigs_stft = pb.transform.stft(sigs, frame_size_gcc, frame_shift_gcc,
                                          pad=False, fading=False)
            voice_activity = channel_wise_activities(sigs, ths=ths)
            frame_wise_voice_activity = convert_to_frame_wise_activities(
                voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
            )
            dominant = get_dominant_time_frequency_mask(sigs_stft)
            gcc_features = get_gcc_for_all_channel_pairs(
                sigs_stft, frame_wise_voice_activity, dominant=dominant, search_range=search_range_gcc, f_min=f_min,
                f_max=f_max_gcc, avg_len=avg_len_gcc
            )
            # # os makedir data/gccs if not exists
            # path = store_gcc(gcc_features,)
            # def store_gcc(gcc_features, path=None):
            #     """Store gcc features to file and return path"""
            #     np.save(path, gcc_features)

            batch_gcc_features.append(gcc_features)
        # return torch.from_numpy(np.array(batch_gcc_features))
        return np.array(batch_gcc_features)

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

        assert waveforms.dim() == 3 # e.g. torch.Size([16, 8, 128000])
        # if self.multichannel_processing:
        # waveforms_mc = waveforms.clone()
        if self.random_channel:
            random_channel = np.random.randint(0, waveforms.shape[1])
            waveforms = waveforms[:, random_channel, :]
        else:
            waveforms = waveforms[:, self.selected_channel, :]
        # else:
        #     # Here channel selection from MC signal is performed.
        #     waveforms_mc = None
        #     waveforms = waveforms[:, self.selected_channel, :]

        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)
        wavlm_feat = self.weight_sum(wavlm_feat)
        wavlm_feat = torch.squeeze(wavlm_feat, -1)

        outputs = self.proj(wavlm_feat)
        outputs = self.lnorm(outputs)

        # with torch.no_grad():
        #     device = waveforms_mc.device  # oder device = next(self.parameters()).device
        #     gcc_features = self.compute_gcc(waveforms_mc.cpu().numpy(), ths=ths)
        #     gcc_features = torch.from_numpy(gcc_features).float().to(device)
        # # Randeffekte? muss ich etwas beachten an den rändern der chunks? leere gccs oder sowas?
        gcc_embeddings = self.gcc_encoder(gcc_features)

        # # TODO: Layer norm for scaling?
        # outputs = self.norm_wavLM(outputs)
        # gcc_embeddings = self.norm_gcc(gcc_embeddings)

        # Concatenate wavlm and gcc embeddings and project to original shape
        outputs = torch.cat((outputs, gcc_embeddings), dim=-1)  # (batch, frames, attention_in + delays)
        # First project into old shape with zeros on gcc and identiy on x to start from only x going into conformer and be able to load conformer weights.
        # outputs = self.merged_linear(outputs)  # out shape is (batch, frames, attention_in)
        # just go directly different input size in conformer
#         # # test: adding to break model
#         # gcc_embeddings = self.proj_gcc(gcc_embeddings)
#         # outputs = outputs + gcc_embeddings


        outputs = self.conformer(outputs)

        # todo: !!!! Idee halbe frame rate damit größere fenster udn dann jeden wert zweimal nehmen?
        # todo: !! Oder window size kleiner machen aber fft size groß halten für frequenz auflösung?
        #  kann man ja seperat setzen aber nur mehr freq resolution, nicht längeres fenster

        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)

        return outputs

if __name__ == '__main__':
    wavlm_conf_name = 'wavlm_base_md_s80'
    model = Model(wavlm_conf_name=wavlm_conf_name)
    print(model)
    x = torch.randn(2, 1, 32000)
    y = model(x)
    print(f'y: {y.shape}')