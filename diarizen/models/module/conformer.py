import math

import torch

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def init_as_identity(linear, noisy=False):
    with torch.no_grad():
        if not noisy:
            linear.weight.zero_()
            linear.bias.zero_()s
            size = min(linear.in_features, linear.out_features)
            linear.weight[:size, :size] = torch.eye(size)
        else:
            torch.nn.init.xavier_uniform_(linear.weight)
            linear.bias.zero_()
            size = min(linear.in_features, linear.out_features)
            linear.weight[:size, :size] = torch.eye(size)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = nn.Embedding(2*maxlen, d_model)
        if embed_v:
            self.pe_v = nn.Embedding(2*maxlen, d_model)
        self.embed_v = embed_v

    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None


class cross_attn(nn.Module):
    """ Multi head self-attention layer
    """
    def __init__(self, dim_main, dim_aux, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim_main = dim_main
        self.dim_aux = dim_aux

        if dim_main != dim_aux:
            self.proj_aux = nn.Linear(dim_aux, dim_main)
        else:
            self.proj_aux = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim=dim_main, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_main)

    def forward(self, main_feat, aux_feat, mask=None):
        """
        main_feat: (B, T, D_main)  -> z.B. WavLM
        aux_feat: (B, T, D_aux)    -> z.B. GCC
        """
        aux_proj = self.proj_aux(aux_feat)  # (B, T, D_main)
        # MultiheadAttention expects: (B, T, D)
        attn_output, _ = self.attn(query=main_feat, key=aux_proj, value=aux_proj, key_padding_mask=mask)
        # Residual + LayerNorm
        out = self.norm(main_feat + self.dropout(attn_output))
        return out

class MultiHeadSelfAttention(nn.Module):
    """ Multi head self-attention layer
    """
    def __init__(
        self,
        n_units: int,
        h: int,
        dropout: float
    ) -> None:
        super().__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)

        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int, pos_k=None) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)

        q = q.transpose(1, 2)   # (batch, head, time, d_k)
        k = k.transpose(1, 2)   # (batch, head, time, d_k)
        v = v.transpose(1, 2)   # (batch, head, time, d_k)
        att_score = torch.matmul(q, k.transpose(-2, -1))
        
        if pos_k is not None:
            reshape_q = q.reshape(batch_size * self.h, -1, self.d_k).transpose(0,1)
            att_score_pos = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            att_score_pos = att_score_pos.transpose(0, 1).reshape(batch_size, self.h, pos_k.size(0), pos_k.size(1))
            scores = (att_score + att_score_pos) / np.sqrt(self.d_k)
        else:
            scores = att_score / np.sqrt(self.d_k)
            
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        x = torch.matmul(p_att, v)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)
     
class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * inputs.sigmoid()
    
class ConformerMHA(nn.Module):
    """
    Conformer MultiHeadedAttention(RelMHA) module with residule connection and dropout.
    """
    def __init__(
        self,
        in_size: int = 256,
        num_head: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_norm = nn.LayerNorm(in_size)
        self.mha = MultiHeadSelfAttention(
            n_units=in_size, 
            h=num_head, 
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, pos_k=None) -> torch.Tensor:
        """
        x: B, T, N
        """
        bs, time, idim = x.shape
        x = x.reshape(-1, idim)
        res = x
        x = self.ln_norm(x)
        x = self.mha(x, bs, pos_k)    
        x = self.dropout(x)
        x = res + x
        x = x.reshape(bs, time, -1)
        return x   

class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer
                    with scaled residule connection and dropout.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, in_size, ffn_hidden, dropout=0.1, swish=Swish()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.ln_norm = nn.LayerNorm(in_size)
        self.w_1 = nn.Linear(in_size, ffn_hidden)
        self.swish = swish
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(ffn_hidden, in_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward function."""
        res = x
        x = self.ln_norm(x)
        x = self.swish(self.w_1(x))
        x = self.dropout1(x)
        x = self.dropout2(self.w_2(x))
        
        return res + 0.5 * x
    
class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model
                    with residule connection and dropout.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size=31, dropout_rate=0.1, swish=Swish(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        self.ln_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = nn.GLU(dim = 1)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.bn_norm = nn.BatchNorm1d(channels)
        self.swish = swish
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        res = x
        x = self.ln_norm(x)
        x = x.transpose(1, 2)   # B, N, T
        
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        x = self.depthwise_conv(x)
        x = self.swish(self.bn_norm(x))
        x = self.dropout(self.pointwise_conv2(x))

        return res + x.transpose(1, 2)


class ConformerBlock_film(nn.Module):
    def __init__(
            self,
            in_size: int = 256,
            ffn_hidden: int = 1024,
            num_head: int = 2,
            kernel_size: int = 31,
            dropout: float = 0.1,
            film_dim=1,
    ) -> None:
        super().__init__()
        self.ffn1 = PositionwiseFeedForward(
            in_size=in_size,
            ffn_hidden=ffn_hidden,
            dropout=dropout
        )
        self.mha = ConformerMHA(
            in_size=in_size,
            num_head=num_head,
            dropout=dropout
        )
        self.conv = ConvolutionModule(
            channels=in_size,
            kernel_size=kernel_size
        )
        self.ffn2 = PositionwiseFeedForward(
            in_size=in_size,
            ffn_hidden=ffn_hidden,
            dropout=dropout
        )
        self.ln_norm = nn.LayerNorm(in_size)
        self.film = FiLM(in_size, film_dim)

    def forward(self, x: torch.Tensor, pos_k=None, num_spk=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        """
        x = self.ffn1(x)
        x = self.mha(x, pos_k)
        x = self.conv(x)
        x = self.film(x, num_spk)
        x = self.ffn2(x)

        return self.ln_norm(x)

class ConformerBlock(nn.Module):
    def __init__(
        self,
        in_size: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 2,
        kernel_size:int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = PositionwiseFeedForward(
            in_size=in_size,
            ffn_hidden=ffn_hidden,
            dropout=dropout
        )
        self.mha = ConformerMHA(
            in_size=in_size, 
            num_head=num_head, 
            dropout=dropout
        )
        self.conv = ConvolutionModule(
            channels=in_size, 
            kernel_size=kernel_size
        )
        self.ffn2 = PositionwiseFeedForward(
            in_size=in_size,
            ffn_hidden=ffn_hidden,
            dropout=dropout
        )
        self.ln_norm = nn.LayerNorm(in_size)
        
    def forward(self, x: torch.Tensor, pos_k=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        """
        x = self.ffn1(x)
        x = self.mha(x, pos_k)
        x = self.conv(x)
        x = self.ffn2(x)
        
        return self.ln_norm(x)

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        attention_in : int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function="ReLU"
    ) -> None:
        super().__init__()
        
        if not use_posi:
            self.pos_emb = None
        else:
            self.pos_emb = RelativePositionalEncoding(attention_in // num_head)
        
        self.conformer_layer = nn.ModuleList([
            ConformerBlock(
                in_size=attention_in,
                ffn_hidden=ffn_hidden,
                num_head=num_head,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_layer)
        ])

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )
        self.output_activate_function = output_activate_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        """
        if self.pos_emb is not None:
            x_len = x.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(x.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, _ = self.pos_emb(pos_seq)
        else:
            pos_k = None
    
        for layer in self.conformer_layer:
            x = layer(x, pos_k)
        if self.output_activate_function:
            x = self.activate_function(x)
        return x

class ConformerEncoder_film(nn.Module):
    def __init__(
            self,
            attention_in: int = 256,
            ffn_hidden: int = 1024,
            num_head: int = 4,
            num_layer: int = 4,
            kernel_size: int = 31,
            dropout: float = 0.1,
            use_posi: bool = False,
            output_activate_function="ReLU",
            film_dim=1,
    ) -> None:
        super().__init__()

        if not use_posi:
            self.pos_emb = None
        else:
            self.pos_emb = RelativePositionalEncoding(attention_in // num_head)

        self.conformer_layer = nn.ModuleList([
            ConformerBlock(
                in_size=attention_in,
                ffn_hidden=ffn_hidden,
                num_head=num_head,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_layer)
        ])
        self.film_layers = nn.ModuleList([
            FiLM(attention_in, film_dim)
            for _ in range(num_layer)
        ])
        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )
        self.output_activate_function = output_activate_function

    def forward(self, x: torch.Tensor, num_spk) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        """
        if self.pos_emb is not None:
            x_len = x.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(x.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, _ = self.pos_emb(pos_seq)
        else:
            pos_k = None

        for layer, film_layer in zip(self.conformer_layer, self.film_layers):
            x = film_layer(x, num_spk)
            x = layer(x, pos_k)
        if self.output_activate_function:
            x = self.activate_function(x)
        return x

class ConformerEncoder_film_deep(nn.Module):
    def __init__(
            self,
            attention_in: int = 256,
            ffn_hidden: int = 1024,
            num_head: int = 4,
            num_layer: int = 4,
            kernel_size: int = 31,
            dropout: float = 0.1,
            use_posi: bool = False,
            output_activate_function="ReLU",
            film_dim=1,
    ) -> None:
        super().__init__()

        if not use_posi:
            self.pos_emb = None
        else:
            self.pos_emb = RelativePositionalEncoding(attention_in // num_head)

        self.conformer_layer = nn.ModuleList([
            ConformerBlock_film(
                in_size=attention_in,
                ffn_hidden=ffn_hidden,
                num_head=num_head,
                kernel_size=kernel_size,
                dropout=dropout,
                film_dim=film_dim,
            ) for _ in range(num_layer)
        ])
        # self.film_layers = nn.ModuleList([
        #     FiLM(attention_in, film_dim)
        #     for _ in range(num_layer)
        # ])
        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )
        self.output_activate_function = output_activate_function

    def forward(self, x: torch.Tensor, num_spk) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        """
        if self.pos_emb is not None:
            x_len = x.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(x.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, _ = self.pos_emb(pos_seq)
        else:
            pos_k = None

        for layer in self.conformer_layer:
            x = layer(x, pos_k, num_spk=num_spk)
        if self.output_activate_function:
            x = self.activate_function(x)
        return x


class gcc_encoder(nn.Module):
    """ GCC Encoder for auxiliary channel processing.
    This module processes GCC features with a series of ConformerMHA layers,
    linear layers, and normalization layers to extract meaningful representations
    for auxiliary channels, such as delays in GCC features.
    """
    def __init__(
            self,
            attention_in_aux: int = 200, # search range for delay
            linear_input_size: int = 399, # number of frames per channel
            linear_output_size: int = 399, # number of frames per channel # todo: double gcc size and transform linear to 2x frames
            num_head_aux = 4,
            num_layer_aux: int = 3,
            dropout: float = 0.1,
            ffn = None,
            sin_cos = False,
    ) -> None:
        # in_size: int = 256, # anzahl frequenzen wenn ich für einen channel das mache oder?
        # num_head: int = 4, # nur einer oder soll ich mehrere nehmen obwohl ich nur self att auf einem channel machen will?
        # dropout: float = 0.1,
        super().__init__()
        self.sin_cos = sin_cos
        self.encoder_layer = nn.ModuleList([
            ConformerMHA(in_size=attention_in_aux,
                         num_head=num_head_aux,
                         dropout=dropout
            ) for _ in range(num_layer_aux)
        ])
        self.linear_layer = nn.ModuleList([
            nn.Linear(in_features=linear_input_size, out_features= linear_output_size
            ) for _ in range(num_layer_aux)
        ])
        # GCC encoder braucht nicht identiy initialisiert werden sondern nur der part wo die emb ignoriert werden sollen
        # vielleicht initialisieren, sodass anfangs keine channel infos ausgetauscht werden?
        # for layer in self.linear_layer:
        #     init_as_identity(layer)

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(2 * linear_input_size) for _ in range(num_layer_aux)
        ])

        self.linear_layer2 = nn.ModuleList([
            nn.Linear(in_features=2*linear_input_size, out_features=linear_output_size
                      ) for _ in range(num_layer_aux)
        ])
        self.encoder_end_layer = ConformerMHA(in_size=attention_in_aux,
                                 num_head=num_head_aux,
                                 dropout=dropout)

        if ffn is not None:
            self.ffn_layer = nn.ModuleList([
                PositionwiseFeedForward(in_size=attention_in_aux, ffn_hidden=ffn, dropout=dropout) for _ in range(num_layer_aux)
            ])
        else:
            self.ffn_layer = [None for _ in range(num_layer_aux)]

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

    def forward(self, gcc_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
             auxiliary shape: (16, 399, 28, 200) stft params fitted to wavlm frames, 28 channel combinations, 200 delays
        """
        b, f, c, d = gcc_features.shape  # (batch, frames, channels, delays)
        pos_k = None

        if self.sin_cos:
            gcc_features = self.sin_cos_representation(gcc_features)
            c = gcc_features.size(2)


        # # gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(b * c, f, d)

        for encoder_layer, linear_layer, linear_layer2, norm, ffn_layer in zip(self.encoder_layer, self.linear_layer, self.linear_layer2,
                                                              self.norm_layers, self.ffn_layer):

            gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*frames, channels, delay):

            gcc_features = encoder_layer(gcc_features, pos_k)
            if ffn_layer is not None:
                gcc_features = ffn_layer(gcc_features)
            gcc_features = rearrange(gcc_features, "(b c) f d -> b c d f", b=b, c=c)  # (batch,channels, delay, frames)

            # TODO: instead of rearranging try use dim parameter and take first rearrange out of loop and change last one? and then one out of loop?
            gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)

            gcc_features_temp = norm(gcc_features_temp)
            gcc_features_temp = linear_layer2(gcc_features_temp)  # (batch, channels, delay, frames)

            gcc_features = gcc_features + gcc_features_temp  # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            gcc_features = rearrange(gcc_features, "b c d f -> b f c d")  # (batch, frames, channels, delays)

            # # Optimized code:
            # # gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*frames, channels, delay):
            # gcc_features = encoder_layer(gcc_features, pos_k)
            # if ffn_layer is not None:
            #     gcc_features = ffn_layer(gcc_features)
            #
            # gcc_features = gcc_features.view(b, c, f, d)  # .permute(0, 1, 3, 2)  # (batch, channels, , delay)
            # gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            # gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            # gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            # gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)
            #
            # gcc_features_temp = norm(gcc_features_temp)
            # gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)
            #
            # gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            # gcc_features = gcc_features.reshape(b * c, f, d)         # (B, T, C, D)  .permute(0, 1, 3, 2)
            #
            #
            # # # Optimized code:
            # # gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(b * c, f, d)
            # #
            # # gcc_features = encoder_layer(gcc_features, pos_k)
            # # gcc_features = gcc_features.view(b, c, d, f)
            # #
            # # gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            # # gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            # # gcc_features_c = gcc_features_c.repeat(1, gcc_features.size(1), 1, 1)
            # # gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)
            # #
            # # gcc_features_temp = norm(gcc_features_temp)
            # # gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)
            # #
            # # gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            # # gcc_features = gcc_features.permute(0, 3, 1, 2)         # (B, T, C, D)
            #
            #



            # # edited code:
            # gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*frames, channels, delay):
            #
            # gcc_features = encoder_layer(gcc_features, pos_k)
            # if ffn_layer is not None:
            #     gcc_features = ffn_layer(gcc_features)
            # gcc_features = rearrange(gcc_features, "(b c) f d -> b c f d", b=b, c=c)  # (batch,channels, delay, frames)
            # # gcc_features = rearrange(gcc_features, "(b c) f d -> b c d f", b=b, c=c)  # (batch,channels, delay, frames)
            #
            # # TODO: instead of rearranging try use dim parameter and take first rearrange out of loop and change last one? and then one out of loop?
            # gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            # gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            # gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            # gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)
            #
            # gcc_features_temp = norm(gcc_features_temp)
            # gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)
            #
            # gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            # gcc_features = rearrange(gcc_features, "b c f d -> b f c d")  # (batch, frames, channels, delays)
            # # gcc_features = rearrange(gcc_features, "b c d f -> b f c d")  # (batch, frames, channels, delays)

        gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(b * c, f, d)
        gcc_features = self.encoder_end_layer(gcc_features, pos_k)
        gcc_features = gcc_features.view(b, f, c, d)
        gcc_features = gcc_features.mean(dim=2) # # (batch, frames, delays)
        return gcc_features

class gcc_encoder_f(nn.Module):
    """ GCC Encoder for auxiliary channel processing.
    This module processes GCC features with a series of ConformerMHA layers,
    linear layers, and normalization layers to extract meaningful representations
    for auxiliary channels, such as delays in GCC features.
    """
    def __init__(
            self,
            attention_in_aux: int = 200, # search range for delay
            linear_input_size: int = 200, # number of frames per channel
            linear_output_size: int = 200, # number of frames per channel # todo: double gcc size and transform linear to 2x frames
            num_head_aux = 4,
            num_layer_aux: int = 3,
            dropout: float = 0.1,
            ffn = None,
            sin_cos = False,
    ) -> None:
        # in_size: int = 256, # anzahl frequenzen wenn ich für einen channel das mache oder?
        # num_head: int = 4, # nur einer oder soll ich mehrere nehmen obwohl ich nur self att auf einem channel machen will?
        # dropout: float = 0.1,
        super().__init__()
        self.sin_cos = sin_cos
        self.encoder_layer = nn.ModuleList([
            ConformerMHA(in_size=attention_in_aux,
                         num_head=num_head_aux,
                         dropout=dropout
            ) for _ in range(num_layer_aux)
        ])
        self.linear_layer = nn.ModuleList([
            nn.Linear(in_features=linear_input_size, out_features= linear_output_size
            ) for _ in range(num_layer_aux)
        ])
        # GCC encoder braucht nicht identiy initialisiert werden sondern nur der part wo die emb ignoriert werden sollen
        # vielleicht initialisieren, sodass anfangs keine channel infos ausgetauscht werden?
        # for layer in self.linear_layer:
        #     init_as_identity(layer)

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(2 * linear_input_size) for _ in range(num_layer_aux)
        ])

        self.linear_layer2 = nn.ModuleList([
            nn.Linear(in_features=2*linear_input_size, out_features=linear_output_size
                      ) for _ in range(num_layer_aux)
        ])
        self.encoder_end_layer = ConformerMHA(in_size=attention_in_aux,
                                 num_head=num_head_aux,
                                 dropout=dropout)

        if ffn is not None:
            self.ffn_layer = nn.ModuleList([
                PositionwiseFeedForward(in_size=attention_in_aux, ffn_hidden=ffn, dropout=dropout) for _ in range(num_layer_aux)
            ])
        else:
            self.ffn_layer = [None for _ in range(num_layer_aux)]

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

    def forward(self, gcc_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
             auxiliary shape: (16, 399, 28, 200) stft params fitted to wavlm frames, 28 channel combinations, 200 delays
        """
        b, f, c, d = gcc_features.shape  # (batch, frames, channels, delays)
        pos_k = None

        if self.sin_cos:
            gcc_features = self.sin_cos_representation(gcc_features)
            c = gcc_features.size(2)


        # # gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(b * c, f, d)

        for encoder_layer, linear_layer, linear_layer2, norm, ffn_layer in zip(self.encoder_layer, self.linear_layer, self.linear_layer2,
                                                              self.norm_layers, self.ffn_layer):

            # gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*frames, channels, delay):
            #
            # gcc_features = encoder_layer(gcc_features, pos_k)
            # if ffn_layer is not None:
            #     gcc_features = ffn_layer(gcc_features)
            # gcc_features = rearrange(gcc_features, "(b c) f d -> b c d f", b=b, c=c)  # (batch,channels, delay, frames)
            #
            # # TODO: instead of rearranging try use dim parameter and take first rearrange out of loop and change last one? and then one out of loop?
            # gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            # gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            # gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            # gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)
            #
            # gcc_features_temp = norm(gcc_features_temp)
            # gcc_features_temp = linear_layer2(gcc_features_temp)  # (batch, channels, delay, frames)
            #
            # gcc_features = gcc_features + gcc_features_temp  # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            # gcc_features = rearrange(gcc_features, "b c d f -> b f c d")  # (batch, frames, channels, delays)

            # # Optimized code:
            # # gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*frames, channels, delay):
            # gcc_features = encoder_layer(gcc_features, pos_k)
            # if ffn_layer is not None:
            #     gcc_features = ffn_layer(gcc_features)
            #
            # gcc_features = gcc_features.view(b, c, f, d)  # .permute(0, 1, 3, 2)  # (batch, channels, , delay)
            # gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            # gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            # gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            # gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)
            #
            # gcc_features_temp = norm(gcc_features_temp)
            # gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)
            #
            # gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            # gcc_features = gcc_features.reshape(b * c, f, d)         # (B, T, C, D)  .permute(0, 1, 3, 2)
            #
            #
            # # # Optimized code:
            # # gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(b * c, f, d)
            # #
            # # gcc_features = encoder_layer(gcc_features, pos_k)
            # # gcc_features = gcc_features.view(b, c, d, f)
            # #
            # # gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            # # gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            # # gcc_features_c = gcc_features_c.repeat(1, gcc_features.size(1), 1, 1)
            # # gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)
            # #
            # # gcc_features_temp = norm(gcc_features_temp)
            # # gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)
            # #
            # # gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            # # gcc_features = gcc_features.permute(0, 3, 1, 2)         # (B, T, C, D)
            #
            #



            # # edited code:
            gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*frames, channels, delay):

            gcc_features = encoder_layer(gcc_features, pos_k)
            if ffn_layer is not None:
                gcc_features = ffn_layer(gcc_features)
            gcc_features = rearrange(gcc_features, "(b c) f d -> b c f d", b=b, c=c)  # (batch,channels, delay, frames)
            # gcc_features = rearrange(gcc_features, "(b c) f d -> b c d f", b=b, c=c)  # (batch,channels, delay, frames)

            # TODO: instead of rearranging try use dim parameter and take first rearrange out of loop and change last one? and then one out of loop?
            gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)

            gcc_features_temp = norm(gcc_features_temp)
            gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)

            gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            gcc_features = rearrange(gcc_features, "b c f d -> b f c d")  # (batch, frames, channels, delays)
            # gcc_features = rearrange(gcc_features, "b c d f -> b f c d")  # (batch, frames, channels, delays)

        gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(b * c, f, d)
        gcc_features = self.encoder_end_layer(gcc_features, pos_k)
        gcc_features = gcc_features.view(b, f, c, d)
        gcc_features = gcc_features.mean(dim=2) # # (batch, frames, delays)
        return gcc_features

class gcc_encoder_cnn(nn.Module):
    """ GCC Encoder for auxiliary channel processing.
    This module processes GCC features with a series of ConformerMHA layers,
    linear layers, and normalization layers to extract meaningful representations
    for auxiliary channels, such as delays in GCC features.
    """

    def init_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialisiere Transformer-Encoder
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialisiere Output Projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def __init__(
            self,
            cnn_out: int = 128,  # Output channels after CNN
            attention_in_aux = 200,
            num_heads: int = 2,  # Heads for multi-head attention
            num_layer_aux: int = 3,  # Depth of transformer stack
            dropout: float = 0.1,
    ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, cnn_out, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(cnn_out),
                nn.ReLU(),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cnn_out,
                nhead=num_heads,
                dim_feedforward=cnn_out * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer_aux)
            self.out_proj = nn.Linear(cnn_out, attention_in_aux)

            self.init_weights()


    def forward(self, gcc):
        # # gcc: (B, F, C, D)
        B, F, C, D = gcc.shape
        # # CNN over delays to detect patterns and peaks
        # # x = gcc.view(B * F * C, D)[:,:64]       # (B*F*C, 1, D)
        x = self.conv(gcc.reshape(B * F * C, 1, D))       # (B*F*C, cnn_out, D)
        x = x.mean(dim=-1)                  # (B*F*C, cnn_out)
        # # x = x.view(B, F, C, -1)             # (B, F, C, cnn_out)


        # # Attention over C (Channel-Pairs) → for each frame
        # x = x.view(B * F, C, -1)            # (B*F, C, cnn_out)
        x = self.transformer(x.reshape(B * F, C, -1))            # (B*F, C, cnn_out)


        # # Attention über C: Query = global Frame-Repräsentation
        # global_q = x.mean(dim=1, keepdim=True)  # (B*F, 1, cnn_out)
        # attn_scores = (x * global_q).sum(dim=-1)  # (B*F, C)
        # attn_weights = torch.sigmoid(attn_scores)  # (B*F, C)
        # x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B*F, cnn_out)

        x = x.mean(dim=1)                   # (B*F, cnn_out)

        # # x = x.view(B, F, -1)  # (B, F, cnn_out)
        out = self.out_proj(x.reshape(B, F, -1) )  # (B, F, D)
        return out

class gcc_encoder_cnn_tac(nn.Module):
    """ GCC Encoder for auxiliary channel processing.
    This module processes GCC features with a series of ConformerMHA layers,
    linear layers, and normalization layers to extract meaningful representations
    for auxiliary channels, such as delays in GCC features.
    """

    def init_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __init__(
            self,
            cnn_out: int = 128,  # Output channels after CNN
            attention_in_aux = 200,
            num_heads: int = 2,  # Heads for multi-head attention
            num_layer_aux: int = 3,  # Depth of transformer stack
            dropout: float = 0.1,
            linear_input_size = 399,
            linear_output_size = 399,
            linears = False,
    ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, cnn_out, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(cnn_out),
                nn.ReLU(),
            )
            # encoder_layer = nn.TransformerEncoderLayer(
            #     d_model=cnn_out,
            #     nhead=num_heads,
            #     dim_feedforward=cnn_out * 4,
            #     dropout=dropout,
            #     batch_first=True
            # )
            # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer_aux)

            self.encoder_layer = nn.ModuleList([
                ConformerMHA(in_size=cnn_out,
                             num_head=num_heads,
                             dropout=dropout
                             ) for _ in range(num_layer_aux)
            ])
            self.linear_layer = nn.ModuleList([
                nn.Linear(in_features=linear_input_size, out_features=linear_output_size
                          ) for _ in range(num_layer_aux)
            ])
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(2 * linear_input_size) for _ in range(num_layer_aux)
            ])

            self.linear_layer2 = nn.ModuleList([
                nn.Linear(in_features=2 * linear_input_size, out_features=linear_output_size
                          ) for _ in range(num_layer_aux)
            ])
            self.encoder_end_layer = ConformerMHA(in_size=cnn_out,
                                                  num_head=num_heads,
                                                  dropout=dropout)

            self.init_weights()

            self.feedforward_layer1 = nn.ModuleList([
                nn.Linear(cnn_out, cnn_out*4) for _ in range(num_layer_aux)
            ])
            self.linears = linears
            if self.linears:
                self.feedforward_layer2 = nn.ModuleList([
                    nn.Linear(cnn_out * 4, cnn_out) for _ in range(num_layer_aux)
                ])
                self.att_norm = nn.ModuleList([
                    nn.LayerNorm(cnn_out)  for _ in range(num_layer_aux)
                ])

    def forward(self, gcc):
        # # gcc: (B, F, C, D)
        B, F, C, D = gcc.shape
        pos_k = None

        # # CNN over delays to detect patterns and peaks
        # # x = gcc.view(B * F * C, D)[:,:64]       # (B*F*C, 1, D)
        x = self.conv(gcc.reshape(B * F * C, 1, D))       # (B*F*C, cnn_out, D)
        x = x.mean(dim=-1)
        # # Attention over C (Channel-Pairs) → for each frame
        # x = self.transformer(x.reshape(B * F, C, -1))            # (B*F, C, cnn_out)
        gcc_features = x.reshape((B, F, C, -1))
        for encoder_layer, linear_layer, linear_layer2, norm, ffn1, ffn2, norm_att in zip(self.encoder_layer, self.linear_layer, self.linear_layer2,
                                                              self.norm_layers, self.feedforward_layer1, self.feedforward_layer2, self.att_norm):
            gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*channels, frames, cnnout):

            gcc_features = encoder_layer(gcc_features, pos_k)
            # optional linear layers:
            if self.linears:
                gcc_features = gcc_features + ffn2(torch.nn.functional.relu(ffn1(norm_att(gcc_features))))  # (batch*frames, channels, frames, cnn_out)

            gcc_features = rearrange(gcc_features, "(b c) f d -> b c d f", b=B, c=C)  # (batch,channels, delay, frames)

            gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)

            gcc_features_temp = norm(gcc_features_temp)
            gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)

            gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            gcc_features = rearrange(gcc_features, "b c d f -> b f c d")  # (batch, frames, channels, delays)

        gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(B * C, F, -1)
        gcc_features = self.encoder_end_layer(gcc_features, pos_k)
        gcc_features = gcc_features.reshape(B, F, C, -1)
        gcc_features = gcc_features.mean(dim=2)  # # (batch, frames, delays)
        #
        # # x = x.mean(dim=1)                   # (B*F, cnn_out)
        #
        # # # x = x.view(B, F, -1)  # (B, F, cnn_out)
        # out = self.out_proj(x.reshape(B, F, -1) )  # (B, F, D)
        out = gcc_features
        return out

class gcc_encoder_cnn_tac_small(nn.Module):
    """ GCC Encoder for auxiliary channel processing.
    This module processes GCC features with a series of ConformerMHA layers,
    linear layers, and normalization layers to extract meaningful representations
    for auxiliary channels, such as delays in GCC features.
    """

    def init_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __init__(
            self,
            cnn_out: int = 64,  # Output channels after CNN
            attention_in_aux = 200,
            num_heads: int = 2,  # Heads for multi-head attention
            num_layer_aux: int = 3,  # Depth of transformer stack
            dropout: float = 0.1,
            linear_input_size = 399,
            linear_output_size = 399,
            linears = False,
    ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, cnn_out, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(cnn_out),
                nn.ReLU(),
            )
            # encoder_layer = nn.TransformerEncoderLayer(
            #     d_model=cnn_out,
            #     nhead=num_heads,
            #     dim_feedforward=cnn_out * 4,
            #     dropout=dropout,
            #     batch_first=True
            # )
            # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer_aux)

            self.encoder_layer = nn.ModuleList([
                ConformerMHA(in_size=cnn_out,
                             num_head=num_heads,
                             dropout=dropout
                             ) for _ in range(num_layer_aux)
            ])
            self.linear_layer = nn.ModuleList([
                nn.Linear(in_features=linear_input_size, out_features=linear_output_size
                          ) for _ in range(num_layer_aux)
            ])
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(2 * linear_input_size) for _ in range(num_layer_aux)
            ])

            self.linear_layer2 = nn.ModuleList([
                nn.Linear(in_features=2 * linear_input_size, out_features=linear_output_size
                          ) for _ in range(num_layer_aux)
            ])
            self.encoder_end_layer = ConformerMHA(in_size=cnn_out,
                                                  num_head=num_heads,
                                                  dropout=dropout)

            self.init_weights()

            self.feedforward_layer1 = nn.ModuleList([
                nn.Linear(cnn_out, cnn_out*4) for _ in range(num_layer_aux)
            ])
            self.linears = linears
            if self.linears:
                self.feedforward_layer2 = nn.ModuleList([
                    nn.Linear(cnn_out * 4, cnn_out) for _ in range(num_layer_aux)
                ])
                self.att_norm = nn.ModuleList([
                    nn.LayerNorm(cnn_out)  for _ in range(num_layer_aux)
                ])

    def forward(self, gcc):
        # # gcc: (B, F, C, D)
        B, F, C, D = gcc.shape
        pos_k = None

        # # CNN over delays to detect patterns and peaks
        # # x = gcc.view(B * F * C, D)[:,:64]       # (B*F*C, 1, D)
        x = self.conv(gcc.reshape(B * F * C, 1, D))       # (B*F*C, cnn_out, D)
        x = x.mean(dim=-1)
        # # Attention over C (Channel-Pairs) → for each frame
        # x = self.transformer(x.reshape(B * F, C, -1))            # (B*F, C, cnn_out)
        gcc_features = x.reshape((B, F, C, -1))
        for encoder_layer, linear_layer, linear_layer2, norm in zip(self.encoder_layer, self.linear_layer, self.linear_layer2,
                                                              self.norm_layers, ):
            gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*channels, frames, cnnout):

            gcc_features = encoder_layer(gcc_features, pos_k)
            # optional linear layers:
            # if self.linears:
            #     gcc_features = gcc_features + ffn2(torch.nn.functional.relu(ffn1(norm_att(gcc_features))))  # (batch*frames, channels, frames, cnn_out)

            gcc_features = rearrange(gcc_features, "(b c) f d -> b c d f", b=B, c=C)  # (batch,channels, delay, frames)

            gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)

            gcc_features_temp = norm(gcc_features_temp)
            gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)

            gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            gcc_features = rearrange(gcc_features, "b c d f -> b f c d")  # (batch, frames, channels, delays)

        gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(B * C, F, -1)
        gcc_features = self.encoder_end_layer(gcc_features, pos_k)
        gcc_features = gcc_features.reshape(B, F, C, -1)
        gcc_features = gcc_features.mean(dim=2)  # # (batch, frames, delays)
        #
        # # x = x.mean(dim=1)                   # (B*F, cnn_out)
        #
        # # # x = x.view(B, F, -1)  # (B, F, cnn_out)
        # out = self.out_proj(x.reshape(B, F, -1) )  # (B, F, D)
        out = gcc_features
        return out

class gcc_encoder_multi(nn.Module):
    """ GCC Encoder for auxiliary channel processing.
    This module processes GCC features with a series of ConformerMHA layers,
    linear layers, and normalization layers to extract meaningful representations
    for auxiliary channels, such as delays in GCC features.
    """
    def __init__(
            self,
            cnn_out: int = 32,  # Output channels after CNN
            attention_in_aux: int = 200, # search range for delay
            linear_input_size: int = 399, # number of frames per channel
            linear_output_size: int = 399, # number of frames per channel # todo: double gcc size and transform linear to 2x frames
            num_head_aux = 4,
            num_layer_aux: int = 3,
            dropout: float = 0.1,
    ) -> None:
        # in_size: int = 256, # anzahl frequenzen wenn ich für einen channel das mache oder?
        # num_head: int = 4, # nur einer oder soll ich mehrere nehmen obwohl ich nur self att auf einem channel machen will?
        # dropout: float = 0.1,
        super().__init__()
        # todo: 2d conv oder 1d conv?
        self.delay_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=cnn_out, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out),
        )

        self.encoder_layer = nn.ModuleList([
            ConformerMHA(in_size=attention_in_aux,
                         num_head=num_head_aux,
                         dropout=dropout
            ) for _ in range(num_layer_aux)
        ])
        self.linear_layer = nn.ModuleList([
            nn.Linear(in_features=linear_input_size, out_features= linear_output_size
            ) for _ in range(num_layer_aux)
        ])
        # GCC encoder braucht nicht identiy initialisiert werden sondern nur der part wo die emb ignoriert werden sollen
        # vielleicht initialisieren, sodass anfangs keine channel infos ausgetauscht werden?
        # for layer in self.linear_layer:
        #     init_as_identity(layer)

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(2 * linear_input_size) for _ in range(num_layer_aux)
        ])

        self.linear_layer2 = nn.ModuleList([
            nn.Linear(in_features=2*linear_input_size, out_features=linear_output_size
                      ) for _ in range(num_layer_aux)
        ])
        self.encoder_end_layer = ConformerMHA(in_size=attention_in_aux,
                                 num_head=num_head_aux,
                                 dropout=dropout)

    def forward(self, gcc_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
             auxiliary shape: (16, 399, 28, 200) stft params fitted to wavlm frames, 28 channel combinations, 200 delays
        """
        b, f, c, d = gcc_features.shape  # (batch, frames, channels, delays)
        pos_k = None
        # # Alternativ über channel attention machen? oder cross attention?

        for encoder_layer, linear_layer, linear_layer2, norm in zip(self.encoder_layer, self.linear_layer, self.linear_layer2,
                                                              self.norm_layers):
            gcc_features = rearrange(gcc_features, "b f c d -> (b c) f d")  # (batch*frames, channels, delay):

            gcc_features = encoder_layer(gcc_features, pos_k)
            gcc_features = rearrange(gcc_features, "(b c) f d -> b c d f", b=b, c=c)  # (batch,channels, delay, frames)

            gcc_features_c = linear_layer(gcc_features)  # (batch,channels, delay, frames)
            gcc_features_c = torch.mean(gcc_features_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
            gcc_features_c = gcc_features_c.expand(-1, gcc_features.size(1), -1, -1)
            gcc_features_temp = torch.cat((gcc_features, gcc_features_c), dim=-1)  # (batch, channels, delay, 2*frames)

            gcc_features_temp = norm(gcc_features_temp)
            gcc_features_temp = linear_layer2(gcc_features_temp)      # (batch, channels, delay, frames)

            gcc_features = gcc_features + gcc_features_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
            gcc_features = rearrange(gcc_features, "b c d f -> b f c d")  # (batch, frames, channels, delays)

        gcc_features = gcc_features.permute(0, 2, 1, 3).reshape(b * c, f, d)
        gcc_features = self.encoder_end_layer(gcc_features, pos_k)
        gcc_features = gcc_features.view(b, f, c, d)
        gcc_features = gcc_features.mean(dim=2) # # (batch, frames, delays)

        return gcc_features

class gcc_encoder_cnn_linear(nn.Module):
    """ GCC Encoder for auxiliary channel processing.
    This module processes GCC features with a series of ConformerMHA layers,
    linear layers, and normalization layers to extract meaningful representations
    for auxiliary channels, such as delays in GCC features.
    """

    def init_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __init__(
            self,
            cnn_out: int = 64,  # Output channels after CNN
            embedding_dim = 256,  # Dimension of the embedding space
            attention_in_aux = 200,
            num_heads: int = 4,  # Heads for multi-head attention
            num_layer_aux: int = 3,  # Depth of transformer stack
            dropout: float = 0.1,
            linear_input_size = 399,
            linear_output_size = 399,
            linears = False,
    ):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, cnn_out, kernel_size=5, stride=2,  padding=2),
                nn.BatchNorm1d(cnn_out),
                nn.ReLU(),
            )

            self.mlp = nn.Sequential(
                nn.Linear(cnn_out, 128),
                nn.ReLU(),
                nn.Linear(128, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
            self.init_weights()

            self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, gcc):
        # # gcc: (B, F, C, D)
        B, F, C, D = gcc.shape
        x = self.conv(gcc.reshape(B * F * C, 1, D))       # (B*F*C, cnn_out, D)
        x = x.mean(dim=-1)
        x = self.mlp(x)  # (B*F*C, embedding_dim)
        x = x.view(B * F, C, -1)  # treat each frame separately
        attn_out, _ = self.attn(x, x, x)  # (B*F, C, embedding_dim)
        x = attn_out.mean(dim=1)  # (B*F, D)
        return x.view(B, F, -1)




class FiLM(nn.Module):
    def __init__(self, feature_dim, conditioning_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * feature_dim)  # gamma and beta
        )

    def forward(self, x, cond):
        """
        x: (B, T, F) mainfeatures
        cond: (B, T, C) or (B, C) Modulationsignal (z.B. num_spk)
        """
        if cond.dim() == 2:  # time dimension
            cond = cond.unsqueeze(1).expand(-1, x.size(1), -1)

        gamma_beta = self.mlp(cond)          # (B, T, 2F)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        return gamma * x + beta


# class ConformerEncoder_mc(nn.Module):
#     def __init__(
#             self,
#             attention_in: int = 256,
#             attention_in_aux: int = 200, # search range for delay
#             linear_input_size: int = 399, # number of frames per channel
#             linear_output_size: int = 399, # number of frames per channel
#             ffn_hidden: int = 1024,
#             num_head: int = 4,
#             num_head_aux = 4, # only one or two heads for auxiliary channel?
#             num_layer: int = 4,
#             num_layer_aux: int = 3,
#             kernel_size: int = 31,
#             dropout: float = 0.1,
#             use_posi: bool = False,
#             output_activate_function="ReLU"
#     ) -> None:
#         # in_size: int = 256, # anzahl frequenzen wenn ich für einen channel das mache oder?
#         # num_head: int = 4, # nur einer oder soll ich mehrere nehmen obwohl ich nur self att auf einem channel machen will?
#         # dropout: float = 0.1,
#         super().__init__()
#         # self.ln_norm = nn.LayerNorm(in_size)
#         # self.self_attention1 = MultiHeadSelfAttention(n_units = attention_in_aux, h = num_head_aux, dropout = dropout)
#         # self.self_attention2 = MultiHeadSelfAttention(n_units = attention_in_aux, h = num_head_aux, dropout = dropout)
#         # self.self_attention3 = MultiHeadSelfAttention(n_units = attention_in_aux, h = num_head_aux, dropout = dropout)
#         # self.conformer_mha1 = ConformerMHA(in_size=attention_in_aux,num_head=num_head_aux,dropout=dropout)
#         # self.conformer_mha2 = ConformerMHA(in_size=attention_in_aux,num_head=num_head_aux,dropout=dropout)
#         # self.conformer_mha3 = ConformerMHA(in_size=attention_in_aux,num_head=num_head_aux,dropout=dropout)
#         self.encoder_layer = nn.ModuleList([
#             ConformerMHA(in_size=attention_in_aux,
#                          num_head=num_head_aux,
#                          dropout=dropout
#             ) for _ in range(num_layer_aux)
#         ])
#         self.linear_layer = nn.ModuleList([
#             nn.Linear(in_features=linear_input_size, out_features= linear_output_size
#             ) for _ in range(num_layer_aux)
#         ])
#         for layer in self.linear_layer:
#             init_as_identity(layer)
#
#         self.norm_layers = nn.ModuleList([
#             nn.LayerNorm(2 * linear_input_size) for _ in range(num_layer_aux)
#         ])
#
#         self.linear_layer2 = nn.ModuleList([
#             nn.Linear(in_features=2*linear_input_size, out_features=linear_output_size
#                       ) for _ in range(num_layer_aux)
#         ])
#         for layer in self.linear_layer2:
#             init_as_identity(layer)
#
#         if not use_posi:
#             self.pos_emb = None
#         else:
#             self.pos_emb = RelativePositionalEncoding(attention_in // num_head)
#
#         self.conformer_layer = nn.ModuleList([
#             ConformerBlock(
#                 in_size=attention_in,
#                 ffn_hidden=ffn_hidden,
#                 num_head=num_head,
#                 kernel_size=kernel_size,
#                 dropout=dropout
#             ) for _ in range(num_layer)
#         ])
#
#         # Activation function layer
#         if output_activate_function:
#             if output_activate_function == "Tanh":
#                 self.activate_function = nn.Tanh()
#             elif output_activate_function == "ReLU":
#                 self.activate_function = nn.ReLU()
#             elif output_activate_function == "ReLU6":
#                 self.activate_function = nn.ReLU6()
#             elif output_activate_function == "LeakyReLU":
#                 self.activate_function = nn.LeakyReLU()
#             elif output_activate_function == "PReLU":
#                 self.activate_function = nn.PReLU()
#             elif output_activate_function == "Sigmoid":
#                 self.activate_function = nn.Sigmoid()
#             else:
#                 raise NotImplementedError(
#                     f"Not implemented activation function {self.activate_function}"
#                 )
#         self.output_activate_function = output_activate_function
#
#
#     def forward(self, x: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): Input tensor (#batch, time, idim).
#              x shape: torch.Size([16, 399, 256])
#              auxiliary shape: (16, 399, 28, 200) stft params fitted to wavlm frames, 28 channel combinations, 200 delays
#
#         """
#         b, f, c, d = auxiliary.shape  # (batch, frames, channels, delays)
#         if self.pos_emb is not None:
#             x_len = x.shape[1]
#             pos_seq = torch.arange(0, x_len).long().to(x.device)
#             pos_seq = pos_seq[:, None] - pos_seq[None, :]
#             pos_k, _ = self.pos_emb(pos_seq)
#         else:
#             pos_k = None
#
#         # auxiliary = rearrange(auxiliary, "b f c d -> (b c f) d")  # (batch*frames, channels, delay)
#         """self attention soll auf jeden channel einzeln angewandt werden und unabhängig von ch number sein => b*C ist "batch size" """
#         # # Alternativ über channel attention machen? oder cross attention?
#         # # alternativ: ConformerMHA forward aufrufen? dann reshapes und so bisschen weniger, residual ist mit drin
#
#         # TODO: iniliaisieren sodass anfangs alles normal durchgeht also ohne kreuzconnections, leichtes rauschen drauf vtl
#
#         for encoder_layer, linear_layer, linear_layer2, norm in zip(self.encoder_layer, self.linear_layer, self.linear_layer2,
#                                                               self.norm_layers):
#             auxiliary = rearrange(auxiliary, "b f c d -> (b c) f d")  # (batch*frames, channels, delay)
#
#             auxiliary = encoder_layer(auxiliary, pos_k)
#             auxiliary = rearrange(auxiliary, "(b c) f d -> b c d f", b=b, c=c)  # (batch,channels, delay, frames)
#
#             auxiliary_c = linear_layer(auxiliary)  # (batch,channels, delay, frames)
#             auxiliary_c = torch.sum(auxiliary_c, dim=1, keepdim=True)  # (batch, 1, delay, frames)
#             auxiliary_c = auxiliary_c.expand(-1, auxiliary.size(1), -1, -1)
#             auxiliary_temp = torch.cat((auxiliary, auxiliary_c), dim=-1)  # (batch, channels, delay, 2*frames)
#
#             auxiliary_temp = norm(auxiliary_temp)
#             auxiliary_temp = linear_layer2(auxiliary_temp)      # (batch, channels, delay, frames)
#
#             auxiliary = auxiliary + auxiliary_temp      # (batch, channels, delay, frames) + # (batch, channels, delay, frames)
#             auxiliary = rearrange(auxiliary, "b c d f -> b f c d")  # (batch, frames, channels, delays)
#
#         auxiliary = auxiliary.mean(dim=2) # # (batch, frames, delays)
#
#         # x = x + auxiliary  # (batch, frames, idim) + (batch, frames, delays)
#         x = torch.cat((x, auxiliary), dim=-1)  # (batch, frames, idim) + (batch, frames, delays)
#         # todo: !!!! Idee halbe frame rate damit größere fenster udn dann jeden wert zweimal nehmen?
#         # todo: !! Oder window size kleiner machen aber fft size groß halten für frequenz auflösung? kann man ja seperat setzen aber muss ich mal testen
#         # TODO x + aux  oder  x und aux concatenate + Linear
#         for layer in self.conformer_layer:
#             x = layer(x, pos_k)
#         if self.output_activate_function:
#             x = self.activate_function(x)
#         return x