# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
import math
import os
import random
from pathlib import Path
import re

import h5py

import torch
import numpy as np
import paderbox as pb
import soundfile as sf
from typing import Dict

from paderbox.io import dump_json, load_json
from diarizen.spatial_features.gcc_phat import compute_vad_th, get_gcc_for_all_channel_pairs_torch
from diarizen.spatial_features.segmentation import spatial_segmentation
from torch.utils.data import Dataset
from diarizen.spatial_features.gcc_phat import (get_gcc_for_all_channel_pairs, channel_wise_activities,
                                                convert_to_frame_wise_activities, get_dominant_time_frequency_mask)


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


def augment_with_noise(num_spk: np.ndarray, noise_prob) -> np.ndarray:
    """
    Fügt zufälliges -1/+1 Rauschen zu einer Auswahl von Elementen hinzu.
    Entspricht der Torch-Variante mit randperm + randint*2-1 + clamp(min=0).

    Args:
        num_spk (np.ndarray): Array der Labels (wird nicht inplace verändert).
        num_noisy_frames (int): Anzahl der zu verändernden Frames.

    Returns:
        np.ndarray: Neues Array mit Rauschaugmentation.
    """
    t, n = num_spk.shape
    num_noisy_frames = int(noise_prob * t)
    idx = np.random.choice(t, num_noisy_frames, replace=False)
    # TODO: zahlen werte die ich sample anpassen an histogram
    # noise = np.random.randint(0, 2, size=num_noisy_frames) * 2 - 1
    noise = np.random.randint(-2, 2, size=num_noisy_frames)
    num_spk[idx] = np.maximum(num_spk[idx, :] + noise[:, None], 0)
    return num_spk

def _collate_fn(batch, max_speakers_per_chunk=4, noisy_labels=False, noise_prob=0.2, gcpsd=False) -> torch.Tensor:
    collated_x = []
    collated_y = []
    collated_names = []
    gccs = []
    num_spks = []

    for x, y, name, gcc in batch:
        # print(f'Processing {name} | {path} | {session_idx}')
        num_speakers = y.shape[-1]
        num_spk = np.sum(y, axis=-1, keepdims=True)

        if noisy_labels:
            num_spk = augment_with_noise(num_spk, noise_prob)

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
        gccs.append(gcc)
        num_spks.append(num_spk)

    if gcpsd:
        return {
            'xs': torch.from_numpy(np.stack(collated_x)).float(),
            'ts': torch.from_numpy(np.stack(collated_y)),
            'names': collated_names,
            "gccs": torch.stack(gccs).to(torch.complex64),
            "num_spks": torch.from_numpy(np.stack(num_spks).astype(np.float32)).float(),
        }

    else:
        return {
            'xs': torch.from_numpy(np.stack(collated_x)).float(),
            'ts': torch.from_numpy(np.stack(collated_y)),
            'names': collated_names,
            "gccs": torch.from_numpy(np.stack(gccs)).float(),
            "num_spks": torch.from_numpy(np.stack(num_spks).astype(np.float32)).float(),
        }
        
        
class DiarizationDataset(Dataset):
    def __init__(
        self, 
        scp_file: str, 
        rttm_file: str,
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
        num_channels = 4,  # number of channels for multichannel mode
        num_spk = False,
        modelbased = False,
        gcpsd = False,
    ):
        self.chunk_indices = []
        self.subset = subset

        self.sample_rate = sample_rate
        self.chunk_sample_size = sample_rate * chunk_size

        self.channel_mode = channel_mode      
        
        self.model_rf_step = model_rf_step
        self.model_rf_duration = model_rf_duration
        self.model_num_frames = model_num_frames
        
        self.rec_scp = load_scp(scp_file)
        self.reco2dur = load_uem(uem_file)
        self.load_gcc_dir = load_gcc_dir
        self.num_channels = num_channels
        self.num_spk = num_spk
        self.modelbased = modelbased
        self.gcpsd = gcpsd

        for rec, dur_info in self.reco2dur.items():
            start_sec, end_sec = dur_info   
            try:
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

        # todo: build groups with shuffled but grouped that 3 of the same audio always in the same batch to minimize data loading
        # self.chunk_indices = self.build_grouped_index(group_chunks=3)
        print(f'DiarizationDataset {subset} with {len(self.chunk_indices)} chunks initialized from {scp_file} and {rttm_file}.')
        self.annotations = self.rttm2label(rttm_file)
        self.energy_th = {}

    def get_session_idx(self, session):
        """
        convert session to session idex
        """
        session_keys = list(self.rec_scp.keys())
        return session_keys.index(session)
            
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

    # def compute_gcc(self, waveforms_mc, frame_size_gcc=4096, frame_shift_gcc=1024, avg_len_gcc=4, search_range_gcc=10,
    #                 f_max_gcc=3500, f_min=125,):
    def compute_gcc(self, waveforms_mc, frame_size_gcc=4096, frame_shift_gcc=311, avg_len_gcc=4,
                    search_range_gcc=10, f_max_gcc=None, f_min=125, apply_ifft=True):
        """
        Compute GCC features from multichannel waveforms.
        returns:
            batch_gcc_features: (batch, frame, channel, channel, search_range)
        """
        # TODO: try different stft values for better gcc but need fit frames of WAVLM
        sigs_stft = pb.transform.stft(waveforms_mc, frame_size_gcc, frame_shift_gcc,
                                      pad=False, fading=False)
        # voice_activity = channel_wise_activities(waveforms_mc, ths=ths)
        # frame_wise_voice_activity = convert_to_frame_wise_activities(
        #     voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
        # )
        # dominant = get_dominant_time_frequency_mask(sigs_stft)

        sigs_stft = torch.from_numpy(sigs_stft)  # (frames, channels, freq)
        gcc_features = get_gcc_for_all_channel_pairs_torch(sigs_stft, f_min=f_min, f_max=f_max_gcc, apply_ifft=apply_ifft)


        # gcc_features = get_gcc_for_all_channel_pairs(
        #     sigs_stft, frame_wise_voice_activity, dominant=dominant, search_range=search_range_gcc, f_min=f_min,
        #     f_max=f_max_gcc, avg_len=avg_len_gcc
        # )
        # # os makedir data/gccs if not exists
        # path = store_gcc(gcc_features,)
        # def store_gcc(gcc_features, path=None):
        #     """Store gcc features to file and return path"""
        #     np.save(path, gcc_features)

        return gcc_features

    def get_mic_selection(self, rec):
        if rec.startswith(("S3")):
            mics = [1, 3, 4, 6]  # for NSF
        else:
            mics = [0, 2, 4, 6]  # default
        return mics

    def extract_wavforms(self, path, start, end, num_channels=4, ):
        # 4 for debugging and smaller memory and faster dev
        start = int(start * self.sample_rate)
        # TODO' random channel selection
        end = int(end * self.sample_rate)
        # if (not self.load_gcc_dir=="base") and not self.load_gcc_dir and path not in self.energy_th.keys():
        #     data, sample_rate = sf.read(path)
        #     # print(data.shape, sample_rate, path)
        #     data = np.einsum('tc->ct', data)  # [channel, time]
        #     ths = compute_vad_th(data)
        #     self.energy_th[path] = ths
        #     del data
        # # if system is noctua, change path to noctua2
        if not os.path.exists(path):
            # path = path.replace("/mnt/*/AMI_AIS_ALI_NSF_CHiME7",
            #                     "/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7")
            path = re.sub(r"^/mnt/[^/]+/AMI_AIS_ALI_NSF_CHiME7", "/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7", path)

        try:
            data, sample_rate = sf.read(path, start=start, stop=end)
        except Exception as e:
            print(f"Error reading {path} from {start} to {end}: {e}")
            raise RuntimeError(f"Error reading {path} from {start} to {end}: {e}")
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
            # if num_channels >= 1:
                # current_channels = data.shape[0]
                # if current_channels < num_channels:
                #     pad_width = ((0, num_channels - current_channels), (0, 0))  # (channel, time)
                #     data = np.pad(data, pad_width, mode='constant')
                # else:
                #     data = data[:num_channels, :]
            data = data[self.get_mic_selection(Path(path).stem), :]
            return data
        else:
            raise ValueError(f"Unsupported channel_mode: {self.channel_mode}")

    def load_gccs(self, path, start, end, load_gcc_dir):
        load_gcc_dir = Path(load_gcc_dir) / self.subset
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

    def get_num_frames(self, L, n_fft, hop):
        return math.floor((L - n_fft) / hop) + 1

    def pad_data(self, data, size=4096):
        L = data.shape[-1]
        n_fft_wavlm = 400
        hop = 320
        N_ref = self.get_num_frames(L, n_fft_wavlm, hop)

        n_fft_big = size
        N_big = self.get_num_frames(L, n_fft_big, hop)

        L_target = (N_ref - 1) * hop + n_fft_big
        pad = max(0, L_target - L)

        last_vals = data[:, -1:]  # Shape (C,1)
        pad_block = np.repeat(last_vals, pad, axis=1)  # Shape (C,pad)
        return np.concatenate([data, pad_block], axis=1)

    def __getitem__(self, idx):
        while True:
            session, path, chunk_start, chunk_end = self.chunk_indices[idx]

            data = self.extract_wavforms(path, chunk_start, chunk_end, num_channels=self.num_channels)          # [start, end)
            if data.shape[1] == self.chunk_sample_size:
                break
            if data.shape[1] < self.chunk_sample_size:   # mainly for CHiME6
                idx = random.randint(0, len(self.chunk_indices) - 1)

        # chunked annotations
        session_idx = self.get_session_idx(session)
        annotations_session = self.annotations[self.annotations['session_idx'] == session_idx]
        chunked_annotations = annotations_session[
            (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
        ]
        
        # discretize chunk annotations at model output resolution
        step = self.model_rf_step
        half = 0.5 * self.model_rf_duration

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
            mask_label[start : end + 1, mapped_label] = 1


        if self.load_gcc_dir == "base":
            gcc_features = torch.zeros(1, 1, 1)
        elif self.load_gcc_dir:

            if self.gcpsd:
                apply_ifft = False
                fmin = 125
                fmax = 3500
                fft_size = 1024
                k_min = int(np.round(fmin / (self.sample_rate / 2) * (fft_size // 2 + 1)))
                k_max = int(np.round(fmax / (self.sample_rate / 2) * (fft_size // 2 + 1)))
                # gcc_features = gcc_features[:, :, k_min:k_max]   # 216 freq bins long

                data_pad = self.pad_data(data, size=fft_size)
                sigs_stft = pb.transform.stft(data_pad[0], size=fft_size, shift=320,
                                              pad=False, fading=False)
                magnitude = torch.from_numpy(np.abs(sigs_stft)[:, k_min:k_max])  # (frames, freq))
                gcc_features = self.compute_gcc(data_pad, frame_size_gcc=fft_size, frame_shift_gcc=320, f_max_gcc=fmax, f_min=fmin,
                                                apply_ifft=apply_ifft)
            else:
                magnitude = None
                fmin = 125
                fft_size = 4096
                fmax = None
                apply_ifft = True
                data_pad = self.pad_data(data, size=fft_size)
                gcc_features = self.load_gccs(path, chunk_start, chunk_end, self.load_gcc_dir)
            #
            # gcc_features = self.compute_gcc(data_pad, frame_size_gcc=fft_size, frame_shift_gcc=320, f_max_gcc=fmax, f_min=fmin,
            #                                 apply_ifft=apply_ifft)

            if magnitude is not None:
                # TODO: MAGNITUDE IN GCC ÜBERGEBEN und überall anpassen
                gcc_features = torch.concat([gcc_features, magnitude[:, None, :]], dim=1)
            # gcc_features2 = self.load_gccs(path, chunk_start, chunk_end, self.load_gcc_dir, kmin=k_min, kmax=k_max)
            # print(gcc_features.shape)
            # print(f"Load GCC time: {time.time() - start_time:.2f}s", flush=True)
            if self.modelbased:
                gcc_features = spatial_segmentation(gcc_features, avg_len=4, shift=320)

        else:
            if self.gcpsd:
                apply_ifft = False
                fmin = 125
                fmax = 3500
                fft_size = 1024
                k_min = int(np.round(fmin / (self.sample_rate / 2) * (fft_size // 2 + 1)))
                k_max = int(np.round(fmax / (self.sample_rate / 2) * (fft_size // 2 + 1)))
                # gcc_features = gcc_features[:, :, k_min:k_max]   # 216 freq bins long

                data_pad = self.pad_data(data, size=fft_size)
                sigs_stft = pb.transform.stft(data_pad[0], size=fft_size, shift=320,
                                              pad=False, fading=False)
                magnitude = torch.from_numpy(np.abs(sigs_stft)[:, k_min:k_max])  # (frames, freq))
            else:
                magnitude = None
                fft_size = 4096
                fmin = 125
                fmax = None
                apply_ifft = True
                data_pad = self.pad_data(data, size=fft_size)

            gcc_features = self.compute_gcc(data_pad, frame_size_gcc=fft_size, frame_shift_gcc=320, f_max_gcc=fmax, f_min=fmin,
                                            apply_ifft=apply_ifft)
            if magnitude is not None:
                gcc_features = torch.concat([gcc_features, magnitude[:, None, :]], dim=1)

        return data, mask_label, session, gcc_features