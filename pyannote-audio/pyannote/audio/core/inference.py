# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from pathlib import Path
from typing import Callable, List, Optional, Text, Tuple, Union

import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
# from masterarbeit.diarization_project.diarization_project.utils import num_spk
from paderbox.io import load_json
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pytorch_lightning.utilities.memory import is_oom_error

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model, Specifications
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.reproducibility import fix_reproducibility
import random
import soundfile as sf
import math
import paderbox as pb
from paderbox.io import dump_json, load_json
from diarizen.spatial_features.gcc_phat import compute_vad_th, get_ch_pairs, get_gcc_for_all_channel_pairs_torch
from diarizen.spatial_features.gcc_phat import (get_gcc_for_all_channel_pairs, channel_wise_activities,
                                                convert_to_frame_wise_activities, get_dominant_time_frequency_mask)

class BaseInference:
    pass

def num_frames(L, n_fft, hop):
    return math.floor((L - n_fft) / hop) + 1

def median_filter_torch(x, kernel_size=7, f="median"):
    """
    x: Tensor (batch, frames, 1)
    """
    assert kernel_size % 2 == 1, "kernel_size should be an odd number"
    pad = kernel_size // 2
    x_padded = F.pad(x, (0, 0, pad, pad), mode="replicate")

    unfolded = x_padded.unfold(dimension=1, size=kernel_size, step=1)
    unfolded = unfolded.transpose(2, 3)  # [B, F, 1, K] -> [B, F, K, 1]
    # Shape: [batch, frames, kernel_size, 1]
    if f == "max":
        x_med = unfolded.max(dim=2).values
    if f == "median":
        x_med = unfolded.median(dim=2).values
    # Shape: [batch, frames, 1]
    return x_med

def pad_data(data, size=1024):
    L = data.shape[-1]
    n_fft_wavlm = 400
    hop = 320
    N_ref = num_frames(L, n_fft_wavlm, hop)

    n_fft_big = size
    N_big = num_frames(L, n_fft_big, hop)

    L_target = (N_ref - 1) * hop + n_fft_big
    pad = max(0, L_target - L)

    last_vals = data[:, -1:]  # Shape (C,1)
    pad_block = np.repeat(last_vals, pad, axis=-1)  # Shape (C,pad)
    return np.concatenate([data, pad_block], axis=-1)

class Inference(BaseInference):
    """Inference

    Parameters
    ----------
    model : Model
        Model. Will be automatically set to eval() mode and moved to `device` when provided.
    window : {"sliding", "whole"}, optional
        Use a "sliding" window and aggregate the corresponding outputs (default)
        or just one (potentially long) window covering the "whole" file or chunk.
    duration : float, optional
        Chunk duration, in seconds. Defaults to duration used for training the model.
        Has no effect when `window` is "whole".
    step : float, optional
        Step between consecutive chunks, in seconds. Defaults to warm-up duration when
        greater than 0s, otherwise 10% of duration. Has no effect when `window` is "whole".
    pre_aggregation_hook : callable, optional
        When a callable is provided, it is applied to the model output, just before aggregation.
        Takes a (num_chunks, num_frames, dimension) numpy array as input and returns a modified
        (num_chunks, num_frames, other_dimension) numpy array passed to overlap-add aggregation.
    skip_aggregation : bool, optional
        Do not aggregate outputs when using "sliding" window. Defaults to False.
    skip_conversion: bool, optional
        In case a task has been trained with `powerset` mode, output is automatically
        converted to `multi-label`, unless `skip_conversion` is set to True.
    batch_size : int, optional
        Batch size. Larger values (should) make inference faster. Defaults to 32.
    device : torch.device, optional
        Device used for inference. Defaults to `model.device`.
        In case `device` and `model.device` are different, model is sent to device.
    use_auth_token : str, optional
        When loading a private huggingface.co model, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    """

    def __init__(
        self,
        model: Union[Model, Text, Path],
        window: Text = "sliding",
        duration: Optional[float] = None,
        step: Optional[float] = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        skip_conversion: bool = False,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        use_auth_token: Union[Text, None] = None,
    ):
        # ~~~~ model ~~~~~

        self.model = (
            model
            if isinstance(model, Model)
            else Model.from_pretrained(
                model,
                map_location=device,
                strict=False,
                use_auth_token=use_auth_token,
            )
        )

        if device is None:
            device = self.model.device
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        specifications = self.model.specifications

        # ~~~~ sliding window ~~~~~

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')

        if window == "whole" and any(
            s.resolution == Resolution.FRAME for s in specifications
        ):
            warnings.warn(
                'Using "whole" `window` inference with a frame-based model might lead to bad results '
                'and huge memory consumption: it is recommended to set `window` to "sliding".'
            )
        self.window = window

        training_duration = next(iter(specifications)).duration
        duration = duration or training_duration
        if training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference: this might lead to suboptimal results."
            )
        self.duration = duration

        # ~~~~ powerset to multilabel conversion ~~~~

        self.skip_conversion = skip_conversion

        conversion = list()
        for s in specifications:
            if s.powerset and not skip_conversion:
                c = Powerset(len(s.classes), s.powerset_max_classes)
            else:
                c = nn.Identity()
            conversion.append(c.to(self.device))

        if isinstance(specifications, Specifications):
            self.conversion = conversion[0]
        else:
            self.conversion = nn.ModuleList(conversion)

        # ~~~~ overlap-add aggregation ~~~~~

        self.skip_aggregation = skip_aggregation
        self.pre_aggregation_hook = pre_aggregation_hook

        self.warm_up = next(iter(specifications)).warm_up
        # Use that many seconds on the left- and rightmost parts of each chunk
        # to warm up the model. While the model does process those left- and right-most
        # parts, only the remaining central part of each chunk is used for aggregating
        # scores during inference.

        # step between consecutive chunks
        step = step or (
            0.1 * self.duration if self.warm_up[0] == 0.0 else self.warm_up[0]
        )

        if step > self.duration:
            raise ValueError(
                f"Step between consecutive chunks is set to {step:g}s, while chunks are "
                f"only {self.duration:g}s long, leading to gaps between consecutive chunks. "
                f"Either decrease step or increase duration."
            )
        self.step = step

        self.batch_size = batch_size

    def to(self, device: torch.device) -> "Inference":
        """Send internal model to `device`"""

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        self.model.to(device)
        self.conversion.to(device)
        self.device = device
        return self

    def get_dtype(self, value: int) -> str:
        """Return the most suitable type for storing the
        value passed in parameter in memory.

        Parameters
        ----------
        value: int
            value whose type is best suited to storage in memory

        Returns
        -------
        str:
            numpy formatted type
            (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
        """
        # signe byte (8 bits), signed short (16 bits), signed int (32 bits):
        types_list = [(127, "b"), (32_768, "i2"), (2_147_483_648, "i")]
        filtered_list = [
            (max_val, type) for max_val, type in types_list if max_val > abs(value)
        ]
        if not filtered_list:
            return "i8"  # signed long (64 bits)
        return filtered_list[0][1]

    def compute_gcc(self, waveforms_mc, frame_size_gcc=1024, frame_shift_gcc=320, avg_len_gcc=4,
                    search_range_gcc=10, f_max_gcc=3500, f_min=125, apply_ifft=True):
        """
        Compute GCC features from multichannel waveforms.
        returns:
            batch_gcc_features: (batch, frame, channel, channel, search_range)
        """
        # TODO: try different stft values for better gcc but need fit frames of WAVLM
        sigs_stft = pb.transform.stft(waveforms_mc, frame_size_gcc, frame_shift_gcc,
                                      pad=False, fading=False)
        sigs_stft = torch.from_numpy(sigs_stft).to(self.device)
        gcc_features = get_gcc_for_all_channel_pairs_torch(sigs_stft, f_min=f_min, f_max=f_max_gcc,apply_ifft=apply_ifft)
        return gcc_features


    def extract_wavforms(self, path, start, end, num_channels=4, sample_rate=None ):
        # 4 for debugging and smaller memory and faster dev
        if not sample_rate:
            data, sample_rate = sf.read(path)
            # print(data.shape, sample_rate, path)
            data = np.einsum('tc->ct', data)  # [channel, time]
            ths = compute_vad_th(data)
            return sample_rate, ths
        else:
            start = int(start * sample_rate)
            # TODO' random channel selection
            end = int(end * sample_rate)
            data, sample_rate = sf.read(path, start=start, stop=end)
            assert sample_rate == sample_rate, f"Sample rate mismatch: {sample_rate} != {self.sample_rate}"

            if data.ndim == 1:
                data = data.reshape(1, -1)
            else:
                data = np.einsum('tc->ct', data)
            data = data[:num_channels, :]
            return data

    def get_gccs(self, path, start, end, th, sample_rate, framewise=False, vad=False, frame_size_gcc=400, frame_shift_gcc=320, ):
        data = self.extract_wavforms(path, start, end, sample_rate=sample_rate)
        gccs = self.compute_gcc(data, ths=th, framewise=framewise, frame_size_gcc=frame_size_gcc,
                           frame_shift_gcc=frame_shift_gcc)
        return gccs

    def load_gcc(self, file_stem: str) -> torch.Tensor:
        """Load GCCs for a specific chunk index from a file.
        """
        # load_gcc_dir = "/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/standard_gcc/"
        load_gcc_dir = "/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_no_freq_filt/"
        # load_gcc_dir = "/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_2/"
        # TODO: LOADING CHUNKS DEPENDS ON STFT PARAMS

        # if load_gcc_dir == "base":
        #     return torch.zeros((32, 1, 1, 1))
        load_gcc_dir = Path(load_gcc_dir) / "test"
        index = load_json(load_gcc_dir / 'index.json')
        if not load_gcc_dir:
            raise FileNotFoundError(f"GCC directory {load_gcc_dir} does not exist")
        segment_id = f"{file_stem}"
        if file_stem not in index:
            raise KeyError(f"File stem '{file_stem}' not found in GCC index")
        h5_filename = Path(index[file_stem])
        if not h5_filename.is_absolute():
            h5_filename = load_gcc_dir / h5_filename
        if not h5_filename.exists():
            raise FileNotFoundError(f"HDF5 file {h5_filename} not found")
        with h5py.File(h5_filename, 'r') as f:
            if file_stem not in f:
                raise KeyError(f"Group '{file_stem}' not found in {h5_filename}")
            grp = f[file_stem]
            if segment_id not in grp:
                raise KeyError(f"Segment ID '{segment_id}' not found in group '{file_stem}'")
            gcc_features = grp[segment_id][()]

        if gcc_features.shape[1] == 6:
            return gcc_features

        if gcc_features.shape[1] == 28:  # todo: use 4 channels for development
            ch_pairs_gcc = get_ch_pairs(8)
        elif gcc_features.shape[1] == 21:
            ch_pairs_gcc = get_ch_pairs(7)
        elif gcc_features.shape[1] == 15:
            ch_pairs_gcc = get_ch_pairs(6)
        elif gcc_features.shape[1] == 10:
            ch_pairs_gcc = get_ch_pairs(5)

        ch_pairs_4 = set(get_ch_pairs(4))
        chs = [i for i, pair in enumerate(ch_pairs_gcc) if pair in ch_pairs_4]
        gcc_features = gcc_features[:, chs, :]
        return gcc_features

    def get_chunk_gccs(self, file_stem: str, chunk_index: int, batch_size, gcc_features, frame_shift_gcc=320, sample_rate=16000) -> torch.Tensor:
        """Load GCCs for a specific chunk index from a file.
        """
        gcc_features_batch = []
        for i in range(batch_size):
            start = round((chunk_index + i ) * self.step, 2)
            end = start + self.duration

            # get chunk from gcc frames:
            frame_shift_sec = frame_shift_gcc / sample_rate  # 320 / 16000 = 0.02 s
            start_frame = int(start / frame_shift_sec)
            end_frame = start_frame + 399 # 400 is the frame size for gcc features
            # TODO STFT params 399 anpassen
            gcc_chunk = gcc_features[start_frame:end_frame]
            gcc_features_batch.append(gcc_chunk)


        try:
            gcc_features_batch = torch.tensor(np.array(gcc_features_batch), dtype=torch.float32)
        except ValueError as e:
            print(
                f"Error converting GCC features to tensor: {e}. "
                f"Check if the shape of the loaded GCC features is correct."
            )
            import pdb
            pdb.set_trace()
        return gcc_features_batch

    def rttm2label(self, rttm_file, current_session):
        '''
        SPEAKER train100_306 1 15.71 1.76 <NA> <NA> 5456 <NA> <NA>
        '''
        annotations = []
        unique_label_lst = []
        with open(rttm_file, 'r') as file:
            for seg_idx, line in enumerate(file):
                line = line.split()
                session, start, dur = line[1], line[3], line[4]

                if session != current_session:
                    continue

                start = float(start)
                end = start + float(dur)
                spk = line[-2] if line[-2] != "<NA>" else line[-3]

                if spk not in unique_label_lst:
                    unique_label_lst.append(spk)

                label_idx = unique_label_lst.index(spk)
                annotations.append((
                        start,
                        end,
                        label_idx
                    ))
        segment_dtype = [
            ("start", "f"),
            ("end", "f"),
            ("label_idx", self.get_dtype(max(a[2] for a in annotations)) if annotations else "i4"),
        ]
        return np.array(annotations, dtype=segment_dtype)

    def get_chunk_start_end(self, batch_start_idx, batch_offset, step, duration):
        """
        Berechnet Start- und Endzeitpunkt eines Chunks in Sekunden.

        Args:
            batch_start_idx (int): Startindex des aktuellen Batches (z.B. c).
            batch_offset (int): Offset innerhalb des Batches (z.B. i).
            step (float): Schrittweite zwischen den Chunks in Sekunden.
            duration (float): Dauer eines Chunks in Sekunden.

        Returns:
            tuple: (chunk_start, chunk_end) in Sekunden.
        """
        chunk_start = (batch_start_idx + batch_offset) * step
        chunk_end = chunk_start + duration
        return chunk_start, chunk_end

    def load_num_spk(self, annotations_session, chunk_start, chunk_end):
        model_rf_duration = 0.025
        model_rf_step = 0.02
        model_num_frames = 399

        chunked_annotations = annotations_session[
            (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
            ]

        # discretize chunk annotations at model output resolution
        step = model_rf_step
        half = 0.5 * model_rf_duration

        start = np.maximum(chunked_annotations["start"], chunk_start) - chunk_start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunked_annotations["end"], chunk_end) - chunk_start - half
        end_idx = np.round(end / step).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunked_annotations['label_idx']))
        num_labels = len(labels)

        mask_label = np.zeros((model_num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}
        for start, end, label in zip(
                start_idx, end_idx, chunked_annotations['label_idx']
        ):
            mapped_label = mapping[label]
            mask_label[start: end + 1, mapped_label] = 1


        num_spk = np.sum(mask_label, axis=-1, keepdims=True)
        return num_spk

    def infer(self, chunks: torch.Tensor, gccs, soft=False, spk_accuracy=False) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Forward pass

        Takes care of sending chunks to right device and outputs back to CPU

        Parameters
        ----------
        chunks : (batch_size, num_channels, num_samples) torch.Tensor
            Batch of audio chunks.

        Returns
        -------
        outputs : (tuple of) (batch_size, ...) np.ndarray
            Model output.
        """

        with torch.inference_mode():
            try:
                if gccs is None:
                    outputs = self.model(chunks.to(self.device))
                else:
                    outputs = self.model(chunks.to(self.device), gccs.to(self.device))
                if spk_accuracy:
                    return outputs
            except RuntimeError as exception:
                if is_oom_error(exception):
                    raise MemoryError(
                        f"batch_size ({self.batch_size: d}) is probably too large. "
                        f"Try with a smaller value until memory error disappears."
                    )
                else:
                    raise exception
                
        def __convert(output: torch.Tensor, conversion: nn.Module, **kwargs):
            return conversion(output, soft=soft).cpu().numpy()

        if not isinstance(outputs, tuple):
            return map_with_specifications(
                self.model.specifications, __convert, outputs, self.conversion
            )
        else:
            return map_with_specifications(
                self.model.specifications, __convert, outputs[0], self.conversion
            ), outputs[1].cpu().numpy()

    def get_mic_selection(self, rec):
        if rec.startswith(("S3")):
            mics = [1, 3, 4, 6]  # for NSF
        else:
            mics = [0, 2, 4, 6]  # default
        return mics

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        hook: Optional[Callable],
        path,
        soft: bool = False,
        out_dir=None,
        only_waveforms=False,
        waveform_bf=None,
    ) -> Union[SlidingWindowFeature, Tuple[SlidingWindowFeature]]:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.
        hook: Optional[Callable]
            When a callable is provided, it is called everytime a batch is
            processed with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """
        # TODO: MC for multi channel recordings still mc and get first channel later
        # waveform_mc = waveform.numpy()
        # waveform = torch.unsqueeze(waveform[0], 0)  # force to use the SDM data
        window_size: int = self.model.audio.get_num_samples(self.duration)
        step_size: int = round(self.step * sample_rate)
        _, num_samples = waveform.shape

        def __frames(
                receptive_field, specifications: Optional[Specifications] = None
        ) -> SlidingWindow:
            if specifications.resolution == Resolution.CHUNK:
                return SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return receptive_field

        frames: Union[SlidingWindow, Tuple[SlidingWindow]] = map_with_specifications(
            self.model.specifications, __frames, self.model._receptive_field
        )

        # prepare complete chunks
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            # num_chunks_test = np.floor((num_samples - window_size) / step_size) + 1
            # print(f"Chunks shape: {chunks.shape}, should be chunks: ", num_chunks_test)
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0
        if waveform_bf is not None:
            chunks_bf: torch.Tensor = rearrange(
                waveform_bf.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
        # prepare last incomplete chunk
        has_last_chunk = (num_samples < window_size) or (
                num_samples - window_size
        ) % step_size > 0
        if has_last_chunk:
            # pad last chunk with zeros
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = F.pad(last_chunk, (0, last_pad))
            if waveform_bf is not None:
                last_chunk_bf: torch.Tensor = waveform_bf[:, num_chunks * step_size :]
                last_chunk_bf = F.pad(last_chunk_bf, (0, last_pad))

        def __empty_list(**kwargs):
            return list()

        outputs: Union[
            List[np.ndarray], Tuple[List[np.ndarray]]
        ] = map_with_specifications(self.model.specifications, __empty_list)

        if hook is not None:
            hook(completed=0, total=num_chunks + has_last_chunk)

        def __append_batch(output, batch_output, **kwargs) -> None:
            output.append(batch_output)
            return

        file_stem = Path(path).stem

        # num_correct = 0
        # num_total = 0
        # num_correct_ov = 0
        # num_total_ov = 0
        # over = 0
        # under = 0
        # frame_deltas = []
        rttm_file = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_mc/test/rttm"
        if not Path(rttm_file).exists():
            rttm_file = "/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_no_chime/test/rttm"

        annotations_session = self.rttm2label(rttm_file, file_stem)
        # print(f"Sliding over {num_chunks} chunks of {self.duration:g}s each, ")

        # TODO: (Un-)comment to switch between loading precomputed GCCs and ground truth num spk
        # gcc_features = self.load_gcc(file_stem)


        for c in np.arange(0, num_chunks, self.batch_size):
            # c = c + 5 * self.batch_size
            batch: torch.Tensor = chunks[c : c + self.batch_size]

            if waveform_bf is not None:
                batch_bf: torch.Tensor = chunks_bf[c : c + self.batch_size, 0]
            num_spks = []

            if not only_waveforms:
                for i in range(batch.shape[0]):
                    chunk_start, chunk_end = self.get_chunk_start_end(c, i , self.step, self.duration)
                    num_spk = self.load_num_spk(annotations_session, chunk_start, chunk_end)
                    num_spks.append(num_spk)

                num_spks = torch.from_numpy(np.stack(num_spks).astype(np.float32)).float().to(self.device)
                # TODO get gcc features
                # gccs = self.get_chunk_gccs(file_stem, c, batch_size=batch.shape[0], gcc_features=gcc_features)
                gccs = []
                fmin = 125
                fmax = 3500
                fft_size = 1024
                median = "max"  #  False
                kernel_size = 9     ## 9 for ffn and 25 for 9Layer
                spk_accuracy = False
                apply_ifft = False
                k_min = int(np.round(fmin / (sample_rate / 2) * (fft_size // 2 + 1)))
                k_max = int(np.round(fmax / (sample_rate / 2) * (fft_size // 2 + 1)))
                for i in range(batch.shape[0]):
                    data = batch[i]
                    data = pad_data(data, size=fft_size)

                    sigs_stft = pb.transform.stft(data[0], size=fft_size, shift=320,
                                                  pad=False, fading=False)
                    magnitude = torch.from_numpy(np.abs(sigs_stft)[:, k_min:k_max]).to(self.device)  # (frames, freq))
                    # TODO MC mics bei 2. system fehlt weil pfad vorne getauscht
                    mics = self.get_mic_selection(rec=file_stem)
                    # print(data.shape, mics, session)

                    gcc_features = self.compute_gcc(data[mics], frame_size_gcc=fft_size, frame_shift_gcc=320, f_max_gcc=fmax,
                                                    f_min=fmin, apply_ifft=apply_ifft)

                    gccs.append(torch.concat([gcc_features, magnitude[:, None, :]], dim=1))

                gccs = torch.stack(gccs, dim=0).to(torch.complex64)
                # gccs = num_spks
                if waveform_bf is not None:
                    batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch_bf, gccs=gccs, soft=soft, spk_accuracy=spk_accuracy)
                else:
                    batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch, gccs=gccs, soft=soft, spk_accuracy=spk_accuracy)
            else:
                if waveform_bf is not None:
                    batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch_bf, gccs=None, soft=soft, spk_accuracy=False)
                else:
                    batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(batch, gccs=None, soft=soft, spk_accuracy=False)

            # print(num_spks.shape, batch_outputs.shape)
            # if spk_accuracy:
            #     # for n, (spk_gt, pred) in enumerate(zip(num_spks, batch_outputs)):
            #
            #     pred_labels = torch.argmax(batch_outputs, dim=-1, keepdim=True)
            #     if median:
            #         pred_labels = median_filter_torch(pred_labels, kernel_size=kernel_size, f=median)
            #     pred_labels = pred_labels.squeeze()
            #     gt_labels = num_spks.squeeze()
            #
            #
            #     hits = (pred_labels == gt_labels).sum().item()
            #     total = pred_labels.numel()
            #
            #     # n = 1
            #     # for i in range(40, len(gt_labels[n]), 100):
            #     #     print("t", gt_labels[n, i:i + 20].int().tolist(), flush=True)
            #     #     print("e", pred_labels[n, i:i + 20].int().tolist(), flush=True)
            #     #     print("p", batch_outputs[n, i:i+20], flush=True)
            #
            #     num_correct += hits
            #     num_total += total
            #     frame_deltas.extend((pred_labels - gt_labels).tolist())
            #
            #     under += torch.sum(pred_labels < gt_labels)
            #     over += torch.sum((pred_labels > gt_labels))
            #
            #     hits_ov = ((pred_labels == gt_labels) & (pred_labels >= 2)).sum().item()
            #     total_ov = (gt_labels >= 2).sum().item()
            #     num_correct_ov += hits_ov
            #     num_total_ov += total_ov
            #     # print(f"Chunk {c}: Accuracy: {hits / total if total > 0 else 0.0:.4f} ({hits}/{total})")
            #     # print(f"Chunk {c}: Accuracy ov: {hits_ov / total_ov if total_ov > 0 else 0.0:.4f} ({hits_ov}/{total_ov})")
            #     # assert False
            #
            #     batch_outputs = batch_outputs.cpu().numpy()
            # print(f"Processed chunk {c}/{num_chunks}, ", batch_outputs.shape, flush=True)

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, batch_outputs
            )
            if hook is not None:
                hook(completed=c + self.batch_size, total=num_chunks + has_last_chunk)
        # process orphan last chunk
        if has_last_chunk:

            # chunk_start, chunk_end = self.get_chunk_start_end(num_chunks, 0 , self.step, self.duration)
            # num_spks = self.load_num_spk(annotations_session, chunk_start, chunk_end)
            # num_spks = torch.from_numpy(num_spks.astype(np.float32)).float().to(self.device)
            # # TODO: ground truth then comment next line out
            # # gccs = num_spks
            if not only_waveforms:
                gccs = gccs[-1]
                if waveform_bf is not None:
                    last_outputs = self.infer(last_chunk_bf[0][None], gccs[None], soft=soft, spk_accuracy=False)
                else:
                    last_outputs = self.infer(last_chunk[None], gccs[None], soft=soft, spk_accuracy=False)
            else:
                if waveform_bf is not None:
                    last_outputs = self.infer(last_chunk_bf[0][None], gccs=None, soft=soft, spk_accuracy=False)
                else:
                    last_outputs = self.infer(last_chunk[None], gccs=None, soft=soft, spk_accuracy=False)

            # if spk_accuracy:
            #     # for spk_gt, pred in zip(num_spks, last_outputs):
            #         # pred_labels = torch.argmax(pred, dim=-1)
            #     pred_labels = torch.argmax(last_outputs, dim=-1, keepdim=True)
            #     if median:
            #         pred_labels = median_filter_torch(pred_labels, kernel_size=kernel_size, f=median)
            #
            #     pred_labels = pred_labels.squeeze()
            #     gt_labels = num_spks.squeeze()
            #
            #     hits = (pred_labels == gt_labels).sum().item()
            #     total = pred_labels.numel()
            #     num_correct += hits
            #     num_total += total
            #     hits_ov = ((pred_labels == gt_labels) & (pred_labels >= 2)).sum().item()
            #     total_ov = (gt_labels >= 2).sum().item()
            #     num_correct_ov += hits_ov
            #     num_total_ov += total_ov
            #
            #     # frame_deltas.extend((pred_labels - gt_labels).tolist())
            #     last_outputs = last_outputs.cpu().numpy()

            _ = map_with_specifications(
                self.model.specifications, __append_batch, outputs, last_outputs
            )
            if hook is not None:
                hook(
                    completed=num_chunks + has_last_chunk,
                    total=num_chunks + has_last_chunk,
                )

        # if spk_accuracy:
        #     ### Accuracy for speaker counting
        #     accuracy = num_correct / num_total if num_total > 0 else 0.0
        #     under = under / num_total if num_total > 0 else 0.0
        #     over = over / num_total if num_total > 0 else 0.0
        #     print(f"under: {under:.4f}, over: {over:.4f}")
        #     print(f"Accuracy: {accuracy:.4f}")
        #     accuracy_ov = num_correct_ov / num_total_ov if num_total_ov > 0 else 0.0
        #     print(f"Accuracy: {accuracy_ov:.4f}")
        #     # assert False
        #     if out_dir:
        #         out_dir = Path(out_dir)
        #         out_dir.mkdir(parents=True, exist_ok=True)
        #         acc_file = out_dir / f"{file_stem}_spk_counting_accuracy.json"
        #         dump_json({"accuracy": accuracy, "accuracy_ov": accuracy_ov}, acc_file)
        #
        #         delta_file = out_dir / f"{file_stem}_spk_counting_frame_deltas.npy"
        #         # np.save(delta_file, np.array(frame_deltas))

        def __vstack(output: List[np.ndarray], **kwargs) -> np.ndarray:
            return np.vstack(output)

        if not isinstance(batch_outputs, tuple):
            outputs: Union[np.ndarray, Tuple[np.ndarray]] = map_with_specifications(
                self.model.specifications, __vstack, outputs
            )
        else:
            outputs_0 = [out[0] for out in outputs]
            outputs_0 = map_with_specifications(
                self.model.specifications, __vstack, outputs_0
            )
            outputs_1 = [out[1] for out in outputs]
            outputs_1 = map_with_specifications(
                self.model.specifications, __vstack, outputs_1
            )
            outputs = (outputs_0, outputs_1)

        def __aggregate(
                outputs: np.ndarray,
                frames: SlidingWindow,
                specifications: Optional[Specifications] = None,
        ) -> SlidingWindowFeature:
            # skip aggregation when requested,
            # or when model outputs just one vector per chunk
            # or when model is permutation-invariant (and not post-processed)
            if (
                    self.skip_aggregation
                    or specifications.resolution == Resolution.CHUNK
                    or (
                    specifications.permutation_invariant
                    and self.pre_aggregation_hook is None
            )
            ):
                frames = SlidingWindow(
                    start=0.0, duration=self.duration, step=self.step
                )
                if not isinstance(outputs, tuple):
                    return SlidingWindowFeature(outputs, frames)
                else:
                    return SlidingWindowFeature(outputs[0], frames), outputs[1]

            if self.pre_aggregation_hook is not None:
                outputs = self.pre_aggregation_hook(outputs)

            aggregated = self.aggregate(
                SlidingWindowFeature(
                    outputs,
                    SlidingWindow(start=0.0, duration=self.duration, step=self.step),
                ),
                frames,
                warm_up=self.warm_up,
                hamming=True,
                missing=0.0,
            )

            # remove padding that was added to last chunk
            if has_last_chunk:
                aggregated.data = aggregated.crop(
                    Segment(0.0, num_samples / sample_rate), mode="loose"
                )

            return aggregated

        return map_with_specifications(
            self.model.specifications, __aggregate, outputs, frames
        )

    def __call__(
        self, file: AudioFile, path, hook: Optional[Callable] = None, soft: bool = False, out_dir=None, only_waveforms=False, bf=False,
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference on a whole file

        Parameters
        ----------
        file : AudioFile
            Audio file.
        hook : callable, optional
            When a callable is provided, it is called everytime a batch is processed
            with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        """

        fix_reproducibility(self.device)

        waveform, sample_rate = self.model.audio(file)
        if bf:
            path_bf = path.replace("/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7/wavs",
                                    "/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7/bf")
            waveform_bf, sample_rate = torchaudio.load(path_bf)
        else:
            waveform_bf = None
        if self.window == "sliding":
            return self.slide(waveform, sample_rate, path=path, hook=hook, soft=soft, out_dir=out_dir, only_waveforms=only_waveforms, waveform_bf=waveform_bf)

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(waveform[None], soft=soft)

        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        return map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    def crop(
        self,
        file: AudioFile,
        chunk: Union[Segment, List[Segment]],
        duration: Optional[float] = None,
        hook: Optional[Callable] = None,
    ) -> Union[
        Tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]:
        """Run inference on a chunk or a list of chunks

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : Segment or list of Segment
            Apply model on this chunk. When a list of chunks is provided and
            window is set to "sliding", this is equivalent to calling crop on
            the smallest chunk that contains all chunks. In case window is set
            to "whole", this is equivalent to concatenating each chunk into one
            (artifical) chunk before processing it.
        duration : float, optional
            Enforce chunk duration (in seconds). This is a hack to avoid rounding
            errors that may result in a different number of audio samples for two
            chunks of the same duration.
        hook : callable, optional
            When a callable is provided, it is called everytime a batch is processed
            with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : (tuple of) SlidingWindowFeature or np.ndarray
            Model output, as `SlidingWindowFeature` if `window` is set to "sliding"
            and `np.ndarray` if is set to "whole".

        Notes
        -----
        If model needs to be warmed up, remember to extend the requested chunk with the
        corresponding amount of time so that it is actually warmed up when processing the
        chunk of interest:
        >>> chunk_of_interest = Segment(10, 15)
        >>> extended_chunk = Segment(10 - warm_up, 15 + warm_up)
        >>> inference.crop(file, extended_chunk).crop(chunk_of_interest, returns_data=False)
        """

        fix_reproducibility(self.device)

        if self.window == "sliding":
            if not isinstance(chunk, Segment):
                start = min(c.start for c in chunk)
                end = max(c.end for c in chunk)
                chunk = Segment(start=start, end=end)

            waveform, sample_rate = self.model.audio.crop(
                file, chunk, duration=duration
            )
            outputs: Union[
                SlidingWindowFeature, Tuple[SlidingWindowFeature]
            ] = self.slide(waveform, sample_rate, hook=hook)

            def __shift(output: SlidingWindowFeature, **kwargs) -> SlidingWindowFeature:
                frames = output.sliding_window
            # config und precompute anpassen
                shifted_frames = SlidingWindow(
                    start=chunk.start, duration=frames.duration, step=frames.step
                )
                return SlidingWindowFeature(output.data, shifted_frames)

            return map_with_specifications(self.model.specifications, __shift, outputs)

        if isinstance(chunk, Segment):
            waveform, sample_rate = self.model.audio.crop(
                file, chunk, duration=duration
            )
        else:
            waveform = torch.cat(
                [self.model.audio.crop(file, c)[0] for c in chunk], dim=1
            )

        outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(waveform[None])

        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        return map_with_specifications(
            self.model.specifications, __first_sample, outputs
        )

    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow,
        warm_up: Tuple[float, float] = (0.0, 0.0),
        epsilon: float = 1e-12,
        hamming: bool = False,
        missing: float = np.NaN,
        skip_average: bool = False,
    ) -> SlidingWindowFeature:
        """Aggregation

        Parameters
        ----------
        scores : SlidingWindowFeature
            Raw (unaggregated) scores. Shape is (num_chunks, num_frames_per_chunk, num_classes).
        frames : SlidingWindow
            Frames resolution.
        warm_up : (float, float) tuple, optional
            Left/right warm up duration (in seconds).
        missing : float, optional
            Value used to replace missing (ie all NaNs) values.
        skip_average : bool, optional
            Skip final averaging step.

        Returns
        -------
        aggregated_scores : SlidingWindowFeature
            Aggregated scores. Shape is (num_frames, num_classes)
        """

        num_chunks, num_frames_per_chunk, num_classes = scores.data.shape

        chunks = scores.sliding_window
        frames = SlidingWindow(
            start=chunks.start,
            duration=frames.duration,
            step=frames.step,
        )

        masks = 1 - np.isnan(scores)
        scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)

        # Hamming window used for overlap-add aggregation
        hamming_window = (
            np.hamming(num_frames_per_chunk).reshape(-1, 1)
            if hamming
            else np.ones((num_frames_per_chunk, 1))
        )

        # anything before warm_up_left (and after num_frames_per_chunk - warm_up_right)
        # will not be used in the final aggregation

        # warm-up windows used for overlap-add aggregation
        warm_up_window = np.ones((num_frames_per_chunk, 1))
        # anything before warm_up_left will not contribute to aggregation
        warm_up_left = round(
            warm_up[0] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[:warm_up_left] = epsilon
        # anything after num_frames_per_chunk - warm_up_right either
        warm_up_right = round(
            warm_up[1] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

        # aggregated_output[i] will be used to store the sum of all predictions
        # for frame #i
        num_frames = (
            frames.closest_frame(
                scores.sliding_window.start
                + scores.sliding_window.duration
                + (num_chunks - 1) * scores.sliding_window.step
                + 0.5 * frames.duration
            )
            + 1
        )
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks
        # that contributed to frame #i
        overlapping_chunk_count: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # aggregated_mask[i] will be used to indicate whether
        # at least one non-NAN frame contributed to frame #i
        aggregated_mask: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # loop on the scores of sliding chunks
        for (chunk, score), (_, mask) in zip(scores, masks):
            # chunk ~ Segment
            # score ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            # mask ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray

            start_frame = frames.closest_frame(chunk.start + 0.5 * frames.duration)

            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                score * mask * hamming_window * warm_up_window
            )

            overlapping_chunk_count[
                start_frame : start_frame + num_frames_per_chunk
            ] += (mask * hamming_window * warm_up_window)

            aggregated_mask[
                start_frame : start_frame + num_frames_per_chunk
            ] = np.maximum(
                aggregated_mask[start_frame : start_frame + num_frames_per_chunk],
                mask,
            )
          
        if skip_average:
            average = aggregated_output
        else:
            average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)

        average[aggregated_mask == 0.0] = missing

        return SlidingWindowFeature(average, frames)

    @staticmethod
    def trim(
        scores: SlidingWindowFeature,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        """Trim left and right warm-up regions

        Parameters
        ----------
        scores : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped scores.
        warm_up : (float, float) tuple
            Left/right warm up ratio of chunk duration.
            Defaults to (0.1, 0.1), i.e. 10% on both sides.

        Returns
        -------
        trimmed : SlidingWindowFeature
            (num_chunks, trimmed_num_frames, num_speakers)-shaped scores
        """

        assert (
            scores.data.ndim == 3
        ), "Inference.trim expects (num_chunks, num_frames, num_classes)-shaped `scores`"
        _, num_frames, _ = scores.data.shape

        chunks = scores.sliding_window

        num_frames_left = round(num_frames * warm_up[0])
        num_frames_right = round(num_frames * warm_up[1])

        num_frames_step = round(num_frames * chunks.step / chunks.duration)
        if num_frames - num_frames_left - num_frames_right < num_frames_step:
            warnings.warn(
                f"Total `warm_up` is so large ({sum(warm_up) * 100:g}% of each chunk) "
                f"that resulting trimmed scores does not cover a whole step ({chunks.step:g}s)"
            )
        new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]

        new_chunks = SlidingWindow(
            start=chunks.start + warm_up[0] * chunks.duration,
            step=chunks.step,
            duration=(1 - warm_up[0] - warm_up[1]) * chunks.duration,
        )

        return SlidingWindowFeature(new_data, new_chunks)
