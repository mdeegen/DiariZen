#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)


import os
import torch
import torch.nn as nn

from functools import lru_cache
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames, 
    multi_conv_receptive_field_size, 
    multi_conv_receptive_field_center
)

from diarizen.models.module.conformer import ConformerEncoder, BLSTMEncoder
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model
from diarizen.models.module.wavlm_config import get_config

class Model(BaseModel):
    def __init__(
        self,
        wavlm_src: str = "wavlm_base",
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        attention_in: int = 256,
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
        model_path = None,
        hidden_size = 512,
        softmax = False,
        projection=False,
        layer_norm=False,
        downstream_model = "blstm",
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel
        self.model_path = model_path
        # wavlm
        if model_path is None:
            self.wavlm_model = self.load_wavlm(wavlm_src)
        else:
            self.wavlm_model = WavLMModel.from_pretrained(
                model_path,
                output_hidden_states=True
            )
        self.wavlm_model.requires_grad_(False)
        self.wavlm_model.eval()

        if softmax:
            self.apply_softmax_to_weights = True
            self.weight_sum = nn.Parameter(torch.randn(wavlm_layer_num))
        else:
            self.apply_softmax_to_weights = False
            self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.input_size = wavlm_feat_dim
        if projection:
            self.proj = nn.Linear(wavlm_feat_dim, attention_in)
            self.input_size = attention_in
        if layer_norm:
            self.lnorm = nn.LayerNorm(self.input_size)

        self.projection = projection
        self.layer_norm = layer_norm


        if downstream_model == 'conformer':
            self.downstream_model = ConformerEncoder(
                attention_in=self.input_size,
                ffn_hidden=ffn_hidden,
                num_head=num_head,
                num_layer=num_layer,
                kernel_size=kernel_size,
                dropout=dropout,
                use_posi=use_posi,
                output_activate_function=output_activate_function
            )
            self.classifier = nn.Linear(self.input_size, self.dimension)
        elif downstream_model == 'blstm':
            self.downstream_model = BLSTMEncoder(
                input_size=self.input_size, # 768 for wavlm without projection, adapt with projection
                hidden_size=hidden_size,
                num_layers=2,
                dropout=dropout,
                bidirectional=True,
                batch_first=True,
            )
            self.classifier = nn.Linear(2 * hidden_size, self.dimension)
        else:
            raise ValueError(f"Unsupported downstream model: {downstream_model}")

        self.activation = self.default_activation()

    def non_wavlm_parameters(self):
        params = []

        if hasattr(self, 'weight_sum') and self.weight_sum is not None:
            if isinstance(self.weight_sum, nn.Parameter):
                params.append(self.weight_sum)
            else:
                params.extend(self.weight_sum.parameters())

        if hasattr(self, 'proj') and self.projection:
            params.extend(self.proj.parameters())

        if hasattr(self, 'lnorm') and self.layer_norm:
            params.extend(self.lnorm.parameters())

        if hasattr(self, 'downstream_model') and self.downstream_model is not None:
            params.extend(self.downstream_model.parameters())

        params.extend(self.classifier.parameters())

        return params
        # return [
        #     *self.weight_sum.parameters(),
        #     *self.proj.parameters(),
        #     *self.lnorm.parameters(),
        #     *self.conformer.parameters(),
        #     *self.classifier.parameters(),
        # ]

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
        print(num_frames, self.chunk_size, self.sample_rate, receptive_field_size, receptive_field_step, flush=True)
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
    
    def forward(self, waveforms: torch.Tensor, gccs=None) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, sample) or (batch, channel, sample)
        num_spk : (batch, frames, 1)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        assert waveforms.dim() == 2
        waveforms = waveforms[:, :]
        if self.model_path is None:
            wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)
        else:
            with torch.no_grad():
                wavlm_feat = self.wavlm_model(waveforms)
                hidden_states = wavlm_feat.hidden_states
                wavlm_feat = torch.stack(hidden_states, dim=-1)  # (batch, frames, feat_dim, layers)

        if self.apply_softmax_to_weights:
            w = torch.softmax(self.weight_sum, dim=0)
            w = w.view(1, 1, 1, -1)  # broadcast to (1,1,1,layers)
            outputs = (wavlm_feat * w).sum(dim=-1)  # (batch, frames, feat_dim, layers) -> (batch, frames, feat_dim)
        else:
            wavlm_feat = self.weight_sum(wavlm_feat)
            outputs = torch.squeeze(wavlm_feat, -1)

        # TODO: What to keep and what to throw
        if self.projection:
            outputs = self.proj(outputs)
        if self.layer_norm:
            outputs = self.lnorm(outputs)

        # outputs = self.conformer(outputs)
        outputs = self.downstream_model(outputs)

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