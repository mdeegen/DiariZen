#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames, 
    multi_conv_receptive_field_size, 
    multi_conv_receptive_field_center
)

from diarizen.models.module.conformer import BLSTM_FILM, ConformerEncoder, FiLM, ConformerEncoder_film, BLSTM, BLSTM_stateful, BLSTM_torch
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model
from diarizen.models.eend.model_wavlm_conformer_gcc_solo import Model as encoder
from diarizen.models.module.wavlm_config import get_config

from transformers import WavLMModel


from torch.nn.utils import parametrize

class Normalize(nn.Module):
    def forward(self, X):
        return nn.functional.softmax(X, dim=-1)
    

from safetensors.torch import load_file

class Model(BaseModel):

    def __init__(
        self,
        model_path: str = None, 
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768, # not used
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        film_dim:int = 11,
        sin_cos = False,
        ffn = None,
        attention_in_aux = 216,
        num_layer_aux = 3,
        max_num_spk = 4, # silence , 1 ,2 oder mehr als 2 spk

        num_layer_downstream: int = 4,
        num_layer_encoder: int = 4,
        max_speakers_per_chunk: int = 4,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        sample_rate: int = 16000,

        hidden_size: int = 512,
        downstream_model="blstm",
        softmax_weight_norm = True,
        attention_in_downstream = 256,
        use_l_norm = False,
        freeze = False,
    ):
        
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel

        # wavlm 


        assert model_path is not None
        self.wavlm_model = WavLMModel.from_pretrained(
            model_path,
            output_hidden_states=True
        )
        #self.wavlm_model = self.load_wavlm(wavlm_src, model_path)
        #self.wavlm_model.eval()  # Set WavLM to evaluation mode

        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)
        if softmax_weight_norm:
            print("Using softmax weight normalization for WavLM layer weighting.")
            parametrize.register_parametrization(self.weight_sum, "weight", Normalize())
        else:
            print("Using raw weights for WavLM layer weighting (no normalization).")

        if downstream_model == 'blstm':
            self.downstream = BLSTM_FILM(
                input_size=attention_in_downstream,
                hidden_size=hidden_size,
                num_layers=num_layer_downstream,
                film_dim=film_dim,      
                attention_in=attention_in_downstream,)      
            self.classifier = nn.Linear(2*hidden_size, self.dimension)

        elif downstream_model == 'conformer':
            self.downstream = ConformerEncoder_film(
                    attention_in=attention_in_downstream, # needed to compensate for removed projection layer
                    ffn_hidden=ffn_hidden,
                    num_head=num_head,
                    num_layer=num_layer_downstream,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    use_posi=use_posi,
                    output_activate_function=output_activate_function,
                    film_dim=film_dim,
                )
            self.classifier = nn.Linear(attention_in_downstream, self.dimension)
        else:
            raise ValueError(f"Unsupported downstream model: {downstream_model}")
       

        num_params = sum(p.numel() for p in self.downstream.parameters() if p.requires_grad)
        print(f"Number of parameters in downstream model: {num_params}") #11550720 --> paper:11.6 M -> seems ok

        self.activation = self.default_activation()

        # TODO: needed ?!?!
        #self.proj = nn.Linear(wavlm_feat_dim, attention_in)
        self.use_l_norm = use_l_norm
        if use_l_norm:
            print("Using layer normalization before downstream model.")
            self.lnorm = nn.LayerNorm(attention_in_downstream) # needed ?!?!


        print("\ndownstream_model, hidden_size", downstream_model, hidden_size, "\n")





        self.encoder = encoder(
            num_layer_aux=num_layer_aux,
            attention_in = attention_in,
            ffn_hidden = ffn_hidden,
            num_head = num_head,
            num_layer = num_layer_encoder,
            dropout = dropout,
            chunk_size = chunk_size,
            use_posi = use_posi,
            output_activate_function = output_activate_function,
            max_speakers_per_chunk = max_speakers_per_chunk,
            selected_channel = selected_channel,
            max_num_spk = max_num_spk, # silence , 1 ,2 oder mehr als 2 spk
            sin_cos = sin_cos,
            ffn = ffn,
            attention_in_aux = attention_in_aux,
            )
        self.freeze = freeze
        if self.freeze:
            self.encoder.eval()
            print("Freezing encoder parameters.")
            for param in self.encoder.parameters():
                param.requires_grad = False
            


    def non_wavlm_parameters(self):
        if self.use_l_norm:
            parameters = [
            *self.weight_sum.parameters(),
            #*self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.downstream.parameters(),
            *self.classifier.parameters(),
            ]
        else:
            parameters = [
            *self.weight_sum.parameters(),
            #*self.proj.parameters(),
            #*self.lnorm.parameters(),
            *self.downstream.parameters(),
            *self.classifier.parameters(),
            *self.encoder.parameters(),
            ]
        if not self.freeze:
            parameters.extend(self.encoder.parameters())
        return parameters

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

    def load_wavlm(self, source: str, model_path: str = None):
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
            model.load_state_dict(state_dict = load_file(model_path), strict=False)


        return model

    def wav2wavlm(self, in_wav, model):
        """
        transform wav to wavlm features
        """
        layer_reps, _ = model.extract_features(in_wav)
        return torch.stack(layer_reps, dim=-1)

    def forward(self, waveforms: torch.Tensor,  gcpsd_features) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, sample) or (batch, channel, sample)
        gcc : (batch, frames, 1)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        assert waveforms.dim() == 3, f'waveforms.dim() = {waveforms.dim()}, should be 3, shape: {waveforms.shape}'
        waveforms = waveforms[:, self.selected_channel, :]

        
        # print("waveforms", waveforms.shape) # [16, 128000]
        #with torch.no_grad():
        #    wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)


        with torch.no_grad():
            wavlm_feat = self.wavlm_model(waveforms)
            hidden_states = wavlm_feat.hidden_states
            wavlm_feat = torch.stack(hidden_states, dim=-1)  # (batch, frames, feat_dim, layers)

        
        #print("wavlm_feat", wavlm_feat.shape) # [16, 399, 768, 13] B, Time, F, Layer
        wavlm_feat = self.weight_sum(wavlm_feat)
        outputs = torch.squeeze(wavlm_feat, -1)

        # TODO mybe leave out
        # print("wavlm_feat squeezed", wavlm_feat.shape) # [16, 399, 768]
        #outputs = self.proj(outputs)
        # print("out proj", outputs.shape) # [16, 399, 256]
        if self.use_l_norm:
            outputs = self.lnorm(outputs)
        # print("out lnorm", outputs.shape) # [16, 399, 256]


        if self.freeze:
            with torch.no_grad():
                gcpsd_emb = self.encoder(waveforms, gcpsd_features)
        else:
            gcpsd_emb = self.encoder(waveforms, gcpsd_features)


        outputs = self.downstream(outputs, gcpsd_emb)

        outputs = self.classifier(outputs) #needed to project to necessary output dimension (e.g. powerset)
        outputs = self.activation(outputs)


        return outputs
    


    
    def finish(self, out_dir):
        print("Finished")




if __name__ == '__main__':
    wavlm_conf_name = 'wavlm_base_plus'
    model = Model(wavlm_conf_name=wavlm_conf_name)
    print(model)
    x = torch.randn(2, 1, 32000)
    y = model(x)
    print(f'y: {y.shape}')