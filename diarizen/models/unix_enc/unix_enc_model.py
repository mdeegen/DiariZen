# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# S3PRL Team has no contribution to this file
# The file was copied from fairseq to remove the dependency on the entire fairseq package
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..wav2vec2.wav2vec2_model import (
#     EXTRACTOR_MODE_CHOICES,
#     LAYER_TYPE_CHOICES,
#     MASKING_DISTRIBUTION_CHOICES,
#     ChoiceEnum,
#     ConvFeatureExtractionModel,
#     GradMultiply,
#     LayerNorm,
#     compute_mask_indices,
#     get_available_activation_fns,
#     get_activation_fn,
#     SamePad,
# )

from torch.nn.modules.activation import MultiheadAttention

from diarizen.models.wavlm.WavLM import ConvFeatureExtractionModel, compute_mask_indices
from diarizen.models.wavlm.modules import GradMultiply, SamePad, get_activation_fn
from diarizen.models.module.wav2vec2.components import LayerNorm

EXTRACTOR_MODE_CHOICES = ("default", "layer_norm")
MASKING_DISTRIBUTION_CHOICES = ("static", "uniform", "normal", "poisson")
LAYER_TYPE_CHOICES = ("transformer", "conformer")

def get_available_activation_fns():
    return ["relu", "gelu", "gelu_accurate", "tanh", "linear", "glu"]
class ChoiceEnum:
    """Minimal stub used only for dataclass annotations/metadata."""
    def init(self, choices):
        self.choices = list(choices)
    def repr(self):
        return f"ChoiceEnum({self.choices})"

# def get_available_activation_fns():
#     return ["relu", "gelu", "gelu_accurate", "tanh", "linear", "glu"]
#
#
# class ChoiceEnum:
#     def init(self, choices):
#         self.choices = list(choices)
#     def repr(self):
#         return f"ChoiceEnum({self.choices})"

logger = logging.getLogger(__name__)

@dataclass
class UnixEncConfig:
    label_rate: float

    extractor_mode: str = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: str = field(
        default='ccccccffffff', metadata={"help": "c for channel-wise attention, f for frame-wise attention"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: str = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: str = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    ) 
    context_size: int = field(
        default=0, metadata={"help": "context size"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    pred_sec: bool = field(
        default=True,
        metadata={"help": "whether to predict secondary pseudo labels"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )

    pos_enc_channel: bool = field(
        default=False,
        metadata={"help": "whether to add positional encoding in the channel dimension"},
    )
    max_num_channel: int = field(
        default=1,
        metadata={"help": "max number of channels"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})

@dataclass
class UnixEncPretrainingConfig:
    data: str = field(default=None, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    mixing_interfere_prob: float = field(
        default=0.5,
        metadata={"help": "the probability of mixing with interefering speaker"},
    )
    mixing_noise_prob: float = field(
        default=0.5,
        metadata={"help": "the probability of mixing with noise"},
    )
    mixing_additive_noise_prob: float = field(
        default=0.5,
        metadata={"help": "the probability of mixing with additive noise (otherwise point noise)"},
    )
    noise_wav_scp: Optional[str] = field(
        default=None,
        metadata={"help": "noise wav scp file"},
    )
    min_num_chans: int = field(
        default=2,
        metadata={"help": "minimum number of channels"},
    )
    max_num_chans: int = field(
        default=4,
        metadata={"help": "maximum number of channels"},
    )
    min_sir: float = field(
        default=-6.0,
        metadata={"help": "min sir"},
    )
    max_sir: float = field(
        default=6.0,
        metadata={"help": "max sir"},
    )
    min_snr: float = field(
        default=-5.0,
        metadata={"help": "min snr"},
    )
    max_snr: float = field(
        default=20.0,
        metadata={"help": "max snr"},
    )
    min_len: float = field(
        default=0.1,
        metadata={"help": "min length of noises and interfences"},
    )
    max_len: float = field(
        default=0.5,
        metadata={"help": "max length of noises and interfences"},
    )
    rir_dir: Optional[str] = field(
        default=None,
        metadata={"help": "directory of RIR"},
    )

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

class CrossChannelTransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        context_size: int = 0,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(self.activation_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.norm1 = LayerNorm(self.embedding_dim)
        self.linear1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.linear2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.norm2 = LayerNorm(self.embedding_dim)
        self.context_size = context_size

    def forward(
        self,
        src: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        assert not self.layer_norm_first
        B, C, T, D = src.size()
        if self.context_size > 0:
            src_pad = F.pad(src, (0, 0, self.context_size, self.context_size))
            feat_list = [src_pad[:, :, i+self.context_size:i+self.context_size+T, :] for i in range(-self.context_size, self.context_size + 1)]
            src_k_v = torch.stack(feat_list, dim=3)
            src_k_v = (torch.transpose(src_k_v, 1, 2)).reshape(B*T, C*(self.context_size*2+1), D)
            
            #self_attn_padding_mask_pad = F.pad(self_attn_padding_mask, (self.context_size, self.context_size), value=True)
            #pad_mask_list = [self_attn_padding_mask_pad[:, :, i+self.context_size:i+self.context_size+T] for i in range(-self.context_size, self.context_size + 1)]
            #pad_mask = torch.stack(pad_mask_list, dim=3)
            #self_attn_padding_mask = (torch.transpose(pad_mask, 1, 2)).reshape(B*T, C*(self.context_size*2+1))
        else:
            src_k_v = (torch.transpose(src, 1, 2)).reshape(B*T, C, D)
            #self_attn_padding_mask = (torch.transpose(self_attn_padding_mask, 1, 2)).reshape(B*T, C)
        src = (torch.transpose(src, 1, 2)).reshape(B*T, C, D)
        # print(src.shape)
        if need_weights:
            src2, attn_output_weights = self.self_attn(src, src_k_v, src_k_v, attn_mask=self_attn_mask, key_padding_mask=None)
        else:
            src2 = self.self_attn(src, src_k_v, src_k_v, attn_mask=self_attn_mask, key_padding_mask=None)[0]
            attn_output_weights = None
        src = src + self.dropout1(src2)
        src = src.transpose(-2, -1)
        src = self.norm1(src)
        src = src.transpose(-2, -1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.transpose(-2, -1)
        src = self.norm2(src)
        src = src.transpose(-2, -1)
        src = torch.transpose(src.view(B, T, C, D), 1, 2)
        if self_attn_padding_mask is not None:
            non_self_attn_padding_mask = (~self_attn_padding_mask).unsqueeze(-1)
            src = src * non_self_attn_padding_mask
        # non_self_attn_padding_mask = (~self_attn_padding_mask).unsqueeze(-1)
        # src = src * non_self_attn_padding_mask
        src = src.transpose(-2, -1)
        return src, (attn_output_weights, None)

class CrossFrameTransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(self.activation_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.norm1 = LayerNorm(self.embedding_dim)
        self.linear1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.linear2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.norm2 = LayerNorm(self.embedding_dim)

    def forward(
        self,
        src: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        assert not self.layer_norm_first
        B, C, T, D = src.size()
        src = src.reshape(B * C, T, D)
        if self_attn_padding_mask is not None:
            self_attn_padding_mask = self_attn_padding_mask.view(B * C, T)
        else:
            self_attn_padding_mask = None # torch.zeros(B * C, T, dtype=torch.bool, device=src.device)
        if need_weights:
            src2, attn_output_weights = self.self_attn(src, src, src, attn_mask=self_attn_mask,
                                key_padding_mask=self_attn_padding_mask)
        else:
            src2 = self.self_attn(src, src, src, attn_mask=self_attn_mask,
                                key_padding_mask=self_attn_padding_mask)[0]
            attn_output_weights = None
        src = src + self.dropout1(src2)
        src = src.transpose(-2, -1)
        src = self.norm1(src)
        src = src.transpose(-2, -1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.transpose(-2, -1)
        src = self.norm2(src)
        src = src.transpose(-2, -1)
        src = src.view(B, C, T, D)
        src = src.transpose(-2, -1)
        return src, (attn_output_weights, None)

def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor

def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"

def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

def make_conv_pos(e, k, g):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv

class MchTransformerEncoder(nn.Module):
    def build_encoder_layer(self, args: UnixEncConfig, attention_type):
        assert args.layer_type == "transformer"
        if attention_type == "crossframe":
            layer = CrossFrameTransformerLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        elif attention_type == "crosschannel":
            layer = CrossChannelTransformerLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
                context_size=args.context_size,
            )

        #layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args: UnixEncConfig):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )

        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )
        
        if args.pos_enc_channel:
            self.channel_enc = nn.Parameter(torch.randn(args.max_num_channel, args.encoder_embed_dim))
            nn.init.normal_(self.channel_enc, mean=0, std=0.01)
        else:
            self.channel_enc = None

        encoder_layers = []
        for ii in range(len(args.encoder_layers)):
            if args.encoder_layers[ii] == 'c':
                encoder_layers.append(self.build_encoder_layer(args, attention_type='crosschannel'))
            elif args.encoder_layers[ii] == 'f':
                encoder_layers.append(self.build_encoder_layer(args, attention_type='crossframe'))
            else:
                raise ValueError("Layer type not defined.")
                
        self.layers = nn.ModuleList(encoder_layers)
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        #self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None, corpus_key=None):
        x, layer_results = self.extract_features(
            x, padding_mask, layer, corpus_key=corpus_key
        )

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        corpus_key=None,
    ):

        B, C, T, D = x.size()
        #print("x0", x.size())
        #print("padding_mask0", padding_mask.size())
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)
        x = x.view(B * C, T, D)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = x.transpose(-2, -1)
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = x.view(B, C, x.size(1), x.size(2))

        if self.channel_enc is not None:
            channel_enc = (self.channel_enc[:C, :]).unsqueeze(1).unsqueeze(0)
            x = x + channel_enc
 
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        r = None
        #print("x3", x.size())
        #print("padding_mask3", padding_mask.size())
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            # if not self.training or (dropout_probability > self.layerdrop):
            layer_check = layer
            #if isinstance(layer, FullyShardedDataParallel):
            #    layer_check = layer.unwrapped_module
            if (corpus_key is None) or (
                not isinstance(layer_check, (
                    TransformerSentenceEncoderWithAdapterLayer,
                    )
                )
            ):
                x = x.transpose(-2, -1)
                # print("Xshape:", x.shape)
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                # else:
                #     assert False
                #     x, (z, lr) = layer(
                #         x,
                #         self_attn_padding_mask=padding_mask,
                #         need_weights=False,
                #         corpus_key=corpus_key,
                #     )
                # #if i >= min_layer:
                # # layer_results.append((x, z, lr))
            layer_results.append(x)
            # else:
            #     print(self.training,dropout_probability, self.layerdrop, flush=True)
            #
            #     print("NEIN", flush=True)

            if i == tgt_layer:
                r = x
                break
                 
        #print("x4", x.size())

        if r is not None:
            x = r

        ## T x B x C -> B x T x C
        #x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :, :-pad_length, :]

            def undo_pad(a, b=None, c=None):
                a = a[:, :, :-pad_length, :]
                return torch.mean(a, 1)

            layer_results = [undo_pad(u) for u in layer_results]
        x = torch.mean(x, 1)
        #print("x5", x.size())
        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

class UnixEncModel(torch.nn.Module):
    def __init__(
        self,
        cfg: UnixEncConfig,
        task_cfg: UnixEncPretrainingConfig,
        dictionaries: List[Any],
    ) -> None:
        super().__init__()
        logger.info(f"UnixEncModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.pred_sec = cfg.pred_sec

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = MchTransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            if self.pred_sec:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, final_dim * (len(dictionaries) + 1)
                )
            else:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, final_dim * len(dictionaries)
                )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: UnixEncConfig, task):
        """Build a new model instance."""

        model = UnixEncModel(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask(self, x, padding_mask):
        B, C, T, D = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                None,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = np.repeat(np.expand_dims(mask_indices, 1), C, axis=1)
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, D),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, C, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        bs, num_chans, T = source.size()
        source = source.view(bs * num_chans, T)
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        features = features.view(bs, num_chans, features.size(1), features.size(2))
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(-1)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(2) % features.size(2)
        if extra > 0:
            padding_mask = padding_mask[:, :, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), padding_mask.size(1), features.size(2), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        sec_target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        #print("source", source.size())
        features = self.forward_features(source)
        #print("features0", features.size())
        if self.pred_sec and sec_target_list is not None:
            target_list += sec_target_list
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)
        #print("len(target_list)", len(target_list))
        #for i in range(len(target_list)):
        #    print("target_list[{}]".format(i), target_list[i].size())
        #    print("target_list[{}]".format(i), torch.max(target_list[i]), torch.min(target_list[i]))
        #print("features1", features.size())

        features_pen = features.float().pow(2).mean()
        # features: (B, C, D, T), float
        features = self.layer_norm(features)
        features = features.transpose(-2, -1)
        # features: (B, C, T, D), float
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        #print("features2", features.size())
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None
        #print("x", x.size())
        #print("mask_indices", mask_indices.size())
        #print("torch.sum(mask_indices)", torch.sum(mask_indices))
        #print("mask_portion", 100.0 * torch.sum(mask_indices) / mask_indices.numel())

        # feature: (B, C, T, D), float
        # target: (B, T), long
        # x: (B, C, T, D), float
        # padding_mask: (B, C, T), bool
        # mask_indices: (B, C, T), bool
        # features: (B, C, T, D), float
        features = features.transpose(-2, -1)

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        if padding_mask is not None:
            padding_mask = padding_mask[:, 0, :]
        else:
            padding_mask = None
        #print("x", x.size())
        #print("padding_mask", padding_mask.size())

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features, "layer_results":layer_results}

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        mask_indices = mask_indices[:, 0, :]
        #print("-" * 80)
        #print("x", x.size())
        #print("padding_mask", padding_mask.size())
        #print("mask_indices", mask_indices.size())
        #print("target_list", len(target_list))
        #for i in range(len(target_list)):
        #    print("target_list[{}]".format(i), target_list[i].size())

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            #print("x[masked_indices]", x[masked_indices].size())
            proj_x_m = self.final_proj(x[masked_indices])
            #print("proj_x_m", proj_x_m.size())
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
                #for i in range(len(proj_x_m_list)):
                #    print("proj_x_m_list[{}]".format(i), proj_x_m_list[i].size())
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[0])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[0])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]
        #print("logit_m_list", logit_m_list[0].size())
        #print("logit_u_list", logit_u_list[0].size())

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
