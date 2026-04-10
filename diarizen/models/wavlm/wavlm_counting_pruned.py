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

        self.wavlm_model = self.load_wavlm("wavlm_base_s80_md")
        print(f"WavLM loaded from {model_path}")

        # wavlm_feat_dim = self.wavlm.config.hidden_size
        # wavlm_layer_num = self.wavlm.config.num_hidden_layers + 1  # 12 + CNN layer
        # self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)
        wavlm_feat_dim = 768 # self.wavlm.encoder_embed_dim
        wavlm_layer_num = 13  # self.wavlm.encoder_layers + 1  # encoder layers + CNN layer
        if isinstance(select_layers, list) and select_layers is not None:
            wavlm_layer_num = len(select_layers)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.proj = nn.Linear(wavlm_feat_dim, proj_size)
        self.lnorm = nn.LayerNorm(proj_size)

        # self.classification = nn.Linear(proj_size, self.max_speakers_per_chunk)

        self.classifier = nn.Sequential(
            nn.Linear(proj_size, proj_size),
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
            # *self.classifier.parameters(),
        ]

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

    def forward(self, waveforms: torch.Tensor, gccs: torch.Tensor = None) -> torch.Tensor:
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
            layer_reps, _ = self.wavlm_model.extract_features(waveforms)

            if self.select_layers is not None:
                layer_reps = [layer_reps[i - 1] for i in self.select_layers]

            wavlm_feat = torch.stack(layer_reps, dim=-1)
        # else:
        #     layer_reps, _ = self.wavlm_model.extract_features(waveforms)
        #
        #     if self.select_layers is not None:
        #         layer_reps = [layer_reps[i - 1] for i in self.select_layers]
        #
        #     wavlm_feat = torch.stack(layer_reps, dim=-1)

        # hidden_states = wavlm_feat.hidden_states
        # if self.select_layers is not None:
        #     if self.select_layers < 0:
        #         hidden_states = hidden_states[self.select_layers:]
        #     else:
        #         hidden_states = hidden_states[:self.select_layers]
        #     # hidden_states = [hidden_states[i] for i in self.select_layers]
        # wavlm_feat = torch.stack(hidden_states, dim=-1)  # (batch, frames, feat_dim, layers)

        # TODO: evtl. softmax damit weights itnerpretierbarer werden?
        feature_layers = self.weight_sum(wavlm_feat)
        feature_layers = torch.squeeze(feature_layers, -1)

        outputs = self.proj(feature_layers)
        outputs = self.lnorm(outputs)

        outputs = self.classifier(outputs)


        return outputs