# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
import random
from pathlib import Path
import torch
import numpy as np
import paderbox as pb
import soundfile as sf
from typing import Dict
from paderbox.io import load_json

from diarizen.spatial_features.gcc_phat import compute_vad_th
from torch.utils.data import Dataset
from diarizen.spatial_features.gcc_phat import (get_gcc_for_all_channel_pairs, channel_wise_activities,
                                                convert_to_frame_wise_activities, get_dominant_time_frequency_mask)

def get_mic_selection(rec):
    if rec.startswith(("S3")):
        mics = [1,3,4,6]  # for NSF
    else:
        mics = [0,2,4,6]  # default
    return mics

def get_dtype(value: int) -> str:
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

def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def load_uem(uem_file: str) -> Dict[str, float]:
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(uem_file):
        return None
    lines = [line.strip().split() for line in open(uem_file)]
    return {x[0]: [float(x[-2]), float(x[-1])] for x in lines}
    
def _gen_chunk_indices(
    init_posi: int,
    data_len: int,
    size: int,
    step: int,
) -> None:
    init_posi = int(init_posi + 1)
    data_len = int(data_len - 1)
    cur_len = data_len - init_posi
    assert cur_len > size
    num_chunks = int((cur_len - size + step) / step)
    
    for i in range(num_chunks):
        yield init_posi + (i * step), init_posi + (i * step) + size

def _collate_fn(batch, max_speakers_per_chunk=4) -> torch.Tensor:
    collated_x = []
    collated_y = []
    collated_names = []
    debug = []
    gccs = []

    for x, y, name, path, session_idx, gcc in batch:
        # print(f'Processing {name} | {path} | {session_idx}')
        num_speakers = y.shape[-1]
        if num_speakers > max_speakers_per_chunk:
            # sort speakers in descending talkativeness order
            indices = np.argsort(-np.sum(y, axis=0), axis=0)
            # keep only the most talkative speakers
            y = y[:, indices[: max_speakers_per_chunk]]

        elif num_speakers < max_speakers_per_chunk:
            # create inactive speakers by zero padding
            y = np.pad(
                y,
                ((0, 0), (0, max_speakers_per_chunk - num_speakers)),
                mode="constant",
            )

        else:
            # we have exactly the right number of speakers
            pass
        # print(f'name: {name} | x: {x.shape} | y: {y.shape}')
        collated_x.append(x)
        collated_y.append(y)
        collated_names.append(name)
        debug.append((path, session_idx))
        gccs.append(gcc)

    return {
        'xs': torch.from_numpy(np.stack(collated_x)).float(), 
        'ts': torch.from_numpy(np.stack(collated_y)), 
        'names': collated_names,
        "gccs": gccs,
        "debug": debug,  # for debugging
    }        
        
import math


def num_frames(L, n_fft, hop):
    return math.floor((L - n_fft) / hop) + 1

def pad_data(data, size=4096):
    L = data.shape[-1]
    n_fft_wavlm = 400
    hop = 320
    N_ref = num_frames(L, n_fft_wavlm, hop)

    n_fft_big = size
    N_big = num_frames(L, n_fft_big, hop)

    L_target = (N_ref - 1) * hop + n_fft_big
    pad = max(0, L_target - L)

    last_vals = data[:, -1:]  # Shape (C,1)
    pad_block = np.repeat(last_vals, pad, axis=1)  # Shape (C,pad)
    return np.concatenate([data, pad_block], axis=1)


class DiarizationDataset(Dataset):
    def __init__(
        self, 
        scp_file: str,
        uem_file: str,
        model_num_frames: int,    # default: wavlm_base
        model_rf_duration: float,  # model.receptive_field.duration, seconds
        model_rf_step: float,  # model.receptive_field.step, seconds
        chunk_size: int = 5,  # seconds
        chunk_shift: int = 5, # seconds
        sample_rate: int = 16000,
        channel_mode: str = "multichannel",    # sdm, random, average, multichannel
        load_gcc_dir = None,
        subset = "train",
        frame_size_gcc = 4096,  # default: 2048, for wavlm_base| 400 & 320, 2048 & 316, 4096 & 311
        frame_shift_gcc = 320,  # default: 316, for wavlm_base,
        f_max_gcc = None,       # default: 3500
        verbose: bool = False,
        modelbased = False,
        apply_ifft=True,
    ):
        self.chunk_indices = []
        self.subset = subset

        self.frame_size_gcc = frame_size_gcc
        self.frame_shift_gcc = frame_shift_gcc
        self.f_max_gcc = f_max_gcc

        self.sample_rate = sample_rate
        self.chunk_sample_size = sample_rate * chunk_size

        self.channel_mode = channel_mode

        self.model_rf_step = model_rf_step
        self.model_rf_duration = model_rf_duration
        self.model_num_frames = model_num_frames

        self.rec_scp = load_scp(scp_file)
        self.reco2dur = load_uem(uem_file)
        self.load_gcc_dir = load_gcc_dir
        self.verbose = verbose

        self.modelbased = modelbased
        self.apply_ifft = apply_ifft
        
        for rec, dur_info in self.reco2dur.items():
            start_sec, end_sec = dur_info   
            try:
                # TODO: somehwere in here the error occures =Y unmatched recording somehow.
                if chunk_size > 0:
                    for st, ed in _gen_chunk_indices(
                            start_sec,
                            end_sec,
                            chunk_size,
                            chunk_shift
                    ):
                        self.chunk_indices.append((rec, self.rec_scp[rec], st, ed))      # seconds
                else:
                    self.chunk_indices.append((rec, self.rec_scp[rec], start_sec, end_sec))
            except:
                print(f'Un-matched recording: {rec}')

        self.energy_th = {}

        self.num_correct =0
        self.num_total =0
        self.under =0
        self.over =0

    def get_session_idx(self, session):
        """
        convert session to session idex
        """
        session_keys = list(self.rec_scp.keys())
        return session_keys.index(session)

    # def compute_gcc(self, waveforms_mc, frame_size_gcc=4096, frame_shift_gcc=1024, avg_len_gcc=4, search_range_gcc=10,
    #                 f_max_gcc=3500, f_min=125,):
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
        # # print("STFT shape:", sigs_stft.shape)  # c, t, f
        # voice_activity = channel_wise_activities(waveforms_mc, ths=ths)
        # frame_wise_voice_activity = convert_to_frame_wise_activities(
        #     voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
        # )
        # dominant = get_dominant_time_frequency_mask(sigs_stft)
        print(f_max_gcc, flush=True)

        dominant = None
        avg_len_gcc = 1
        frame_wise_voice_activity = np.ones((sigs_stft.shape[0], sigs_stft.shape[1]))  # (channels , frames)

        gcc_features = get_gcc_for_all_channel_pairs(
            sigs_stft, frame_wise_voice_activity, dominant=dominant, search_range=search_range_gcc, f_min=f_min,
            f_max=f_max_gcc, avg_len=avg_len_gcc, shift=frame_shift_gcc, modelbased=modelbased, apply_ifft=apply_ifft,
        )
        # # os makedir data/gccs if not exists
        # path = store_gcc(gcc_features,)
        # def store_gcc(gcc_features, path=None):
        #     """Store gcc features to file and return path"""
        #     np.save(path, gcc_features)

        return gcc_features

    def extract_wavforms(self, path, start, end, session, num_channels=4, ):
        # 4 for debugging and smaller memory and faster dev
        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        # if path not in self.energy_th.keys():
        #     data, sample_rate = sf.read(path)
        #     # print(data.shape, sample_rate, path)
        #     data = np.einsum('tc->ct', data)  # [channel, time]
        #     ths = compute_vad_th(data)
        #     self.energy_th[path] = ths
        self.energy_th[path] = 0

        data, sample_rate = sf.read(path, start=start, stop=end)
        assert sample_rate == self.sample_rate, f"Sample rate mismatch: {sample_rate} != {self.sample_rate}"

        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = np.einsum('tc->ct', data) 

        if self.channel_mode == "sdm":
            return np.expand_dims(data[0, :], 0)
        elif self.channel_mode == "random":
            channel_idx = random.randint(0, data.shape[0] - 1)
            return np.expand_dims(data[channel_idx, :], 0)
        elif self.channel_mode == "average":
            return np.mean(data, 0, keepdims=True)
        elif self.channel_mode == "multichannel":

            mic_selection = get_mic_selection(session)
            print("MIC SELECTION: ", mic_selection, flush=True)
            data = data[mic_selection, :]  # take only first 4 channels
            return data
        else:
            raise ValueError(f"Unsupported channel_mode: {self.channel_mode}")

    def load_gccs(self, path, start, end, data, ths, load_gcc_dir):
        load_gcc_dir = Path(load_gcc_dir)
        index = load_json(load_gcc_dir / 'index.json')
        if not load_gcc_dir:
            raise FileNotFoundError(f"GCC directory {load_gcc_dir} does not exist")

        file_stem = Path(path).stem
        segment_id = f"{file_stem}_{start}_{end}"
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

        return gcc_features

    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        while True:
            session, path, chunk_start, chunk_end = self.chunk_indices[idx]
            path = self.fix_prefix(path)
            print(session, path, chunk_start, chunk_end, flush=True)
            data = self.extract_wavforms(path, chunk_start, chunk_end, session=session)          # [start, end)
            if data.shape[1] == self.chunk_sample_size:
                break
            if data.shape[1] < self.chunk_sample_size:   # mainly for CHiME6
                idx = random.randint(0, len(self.chunk_indices) - 1)

        # chunked annotations
        session_idx = self.get_session_idx(session)
        rttm_file = f"/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_no_chime/{self.subset}/rttm"
        self.annotations = self.rttm2label(rttm_file)

        annotations_session = self.annotations[self.annotations['session_idx'] == session_idx]
        chunked_annotations = annotations_session[
            (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
            ]
        # discretize chunk annotations at model output resolution
        step = self.model_rf_step
        half = 0.5 * self.model_rf_duration
        # print(chunk_start, chunk_end, step, half, flush=True)
        start = np.maximum(chunked_annotations["start"], chunk_start) - chunk_start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunked_annotations["end"], chunk_end) - chunk_start - half
        end_idx = np.round(end / step).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunked_annotations['label_idx']))
        num_labels = len(labels)

        mask_label = np.zeros((self.model_num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}
        for start, end, label in zip(
                start_idx, end_idx, chunked_annotations['label_idx']
        ):
            mapped_label = mapping[label]
            mask_label[start: end + 1, mapped_label] = 1




        ths = self.energy_th[path]

        # TODO: set STFT params accordingly
        frame_size_gcc =  1024 # 4096
        frame_shift_gcc = 320  # 311
        f_max_gcc = None
        f_min = None
        data = pad_data(data, size=frame_size_gcc)
        # print("DATA", data.shape, frame_size_gcc, frame_shift_gcc, self.f_max_gcc, flush=True)
        gcc_features = self.compute_gcc(data, ths=ths, frame_size_gcc=frame_size_gcc, apply_ifft=self.apply_ifft,
                                        frame_shift_gcc=frame_shift_gcc, f_max_gcc=f_max_gcc, f_min=f_min, modelbased=self.modelbased)


        # # rttm_file = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_no_chime/train/rttm"
        # # annotations_session = self.rttm2label(rttm_file, session_idx)
        # # # chunk_start, chunk_end = self.get_chunk_start_end(c, i, self.step, self.duration)
        # # num_spk = self.load_num_spk(annotations_session, chunk_start, chunk_end).squeeze()
        # num_spk = np.sum(mask_label, axis=-1)
        # from scipy.ndimage import median_filter, maximum_filter
        # # gcc_features = median_filter(gcc_features, size=7)
        # # gcc_features = maximum_filter(gcc_features, size=5)
        # # for i in range(0, len(gcc_features), 10):
        # print("t", num_spk[100:140], flush=True)
        # print("e", gcc_features[100:140], flush=True)
        #
        # # # mask = (num_spk == 0) | (num_spk == 1) | (num_spk == 2)
        # hits = np.sum((gcc_features == num_spk)[mask])
        # total = np.sum(mask)
        # self.num_correct += hits
        # self.num_total += total
        # self.under += np.sum((gcc_features < num_spk))
        # self.over += np.sum((gcc_features > num_spk))
        # # assert False
        # # frame_deltas.extend((gcc_features - num_spk).tolist())
        # #
        # #
        # # pred_labels = gcc_features
        # # gt_labels = num_spk.squeeze()
        # #
        # # hits = (pred_labels == gt_labels).sum()
        # # total = len(pred_labels)
        # #
        # # acc = hits/total
        # # print(acc, flush=True)
        # # assert False


        if self.verbose:
            return data, ths, session, path, session_idx, gcc_features, self.f_max_gcc
        return data, ths, session, path, session_idx, gcc_features

    def fix_prefix(self, path):
        old_prefix = "/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/"
        ihan_prefix = "/mnt/matylda3/ihan/project/diarization/dataset/NOTSOFAR1/multi-channel/wavs/"
        new_prefix = "/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/wavs/"

        if path.startswith(old_prefix):
            return path.replace(old_prefix, new_prefix)
        # elif path.startswith(ihan_prefix):
        #     return path.replace(ihan_prefix, new_prefix)
        else:
            print(f"Path does not start with expected prefix: {old_prefix} while it is {path}")
            return path

    def rttm2label(self, rttm_file):
        '''
        SPEAKER train100_306 1 15.71 1.76 <NA> <NA> 5456 <NA> <NA>
        '''
        annotations = []
        session_lst = []
        with open(rttm_file, 'r') as file:
            for seg_idx, line in enumerate(file):
                line = line.split()
                session, start, dur = line[1], line[3], line[4]

                start = float(start)
                end = start + float(dur)
                spk = line[-2] if line[-2] != "<NA>" else line[-3]

                # new nession
                if session not in session_lst:
                    unique_label_lst = []
                    session_lst.append(session)

                if spk not in unique_label_lst:
                    unique_label_lst.append(spk)

                label_idx = unique_label_lst.index(spk)

                annotations.append(
                    (
                        self.get_session_idx(session),
                        start,
                        end,
                        label_idx
                    )
                )

        segment_dtype = [
            (
                "session_idx",
                get_dtype(max(a[0] for a in annotations)),
            ),
            ("start", "f"),
            ("end", "f"),
            ("label_idx", get_dtype(max(a[3] for a in annotations))),
        ]

        return np.array(annotations, dtype=segment_dtype)

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