import os
import pdb

import torchaudio
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import argparse
import toml
import h5py
from einops import rearrange
import paderbox as pb
from typing import Dict
from paderbox.io import dump_json, load_json
from diarizen.spatial_features.gcc_phat import compute_vad_th
from diarizen.spatial_features.gcc_phat import (get_gcc_for_all_channel_pairs, channel_wise_activities,
                                                convert_to_frame_wise_activities, get_dominant_time_frequency_mask)
import math

def get_mic_selection(rec):
    if rec.startswith(("S3")):
        mics = [1,3,4,6]  # for NSF
    else:
        mics = [0,2,4,6]  # default
    return mics


def num_frames(L, n_fft, hop):
    return math.floor((L - n_fft) / hop) + 1

def pad_data(data):
    L = data.shape[-1]
    n_fft_wavlm = 400
    hop = 320
    N_ref = num_frames(L, n_fft_wavlm, hop)

    n_fft_big = 4096
    N_big = num_frames(L, n_fft_big, hop)

    L_target = (N_ref - 1) * hop + n_fft_big
    pad = max(0, L_target - L)

    last_vals = data[:, -1:]  # Shape (C,1)
    pad_block = np.repeat(last_vals, pad, axis=1)  # Shape (C,pad)
    return np.concatenate([data, pad_block], axis=1)

def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def compute_gcc(waveforms_mc, frame_size_gcc=4096, frame_shift_gcc=311, avg_len_gcc=4, apply_ifft=True,
                search_range_gcc=10, f_max_gcc=3500, f_min=125, ths=[], framewise=False, eval=False, modelbased=False):
    """
    Compute GCC features from multichannel waveforms.
    returns:
        batch_gcc_features: (batch, frame, channel, channel, search_range)
    """
    # # TODO: try different stft values for better gcc but need fit frames of WAVLM

    sigs_stft = pb.transform.stft(waveforms_mc, frame_size_gcc, frame_shift_gcc,
                                  pad=False, fading=False)
    print("STFT shape:", sigs_stft.shape)  # c, t, f

    # voice_activity = channel_wise_activities(waveforms_mc, ths=ths)
    # frame_wise_voice_activity = convert_to_frame_wise_activities(
    #     voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
    # )
    # dominant = get_dominant_time_frequency_mask(sigs_stft)
    # # TODO optimize memory usage for eval
    if eval:
        del sigs_stft
        sigs_stft = np.zeros((waveforms_mc.shape[0], 1, frame_size_gcc//2 + 1))  #  c, t, f

    dominant = None
    avg_len_gcc = 1
    frame_wise_voice_activity = np.ones((sigs_stft.shape[0], sigs_stft.shape[1]))  # (channels , frames)

    gcc_features = get_gcc_for_all_channel_pairs(
        sigs_stft, frame_wise_voice_activity, dominant=dominant, search_range=search_range_gcc, f_min=f_min, apply_ifft=apply_ifft,
        f_max=f_max_gcc, avg_len=avg_len_gcc, framewise=framewise,modelbased=modelbased)# eval=eval, audio=waveforms_mc, shift=frame_shift_gcc,)

    # # os makedir data/gccs if not exists
    # path = store_gcc(gcc_features,)
    # def store_gcc(gcc_features, path=None):
    #     """Store gcc features to file and return path"""
    #     np.save(path, gcc_features)

    return gcc_features

def precompute_gccs(config, scp_file, uem_file, out_dir, dataset, f_max_gcc, modelbased=False):
    print("PYTHON: Precomputing GCCS", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    worker_id = os.environ.get("SGE_TASK_ID", "0")
    if modelbased:
        h5_filename = out_dir / f"num_spk{worker_id}.h5"
        json_filename = out_dir / f"num_spk_index{worker_id}.json"
    else:
        h5_filename = out_dir / f"gcc_features{worker_id}.h5"
        json_filename = out_dir / f"gcc_features_index{worker_id}.json"
    index = {}

    duration = 8
    num_channels = 4
    rec_scp = load_scp(scp_file)
    energy_th = {}
    for rec in rec_scp.keys():
        path = rec_scp[rec]
        data, sample_rate = sf.read(path)
        data = np.einsum('tc->ct', data)  # [channel, time]

        # frame_size_gcc =  4096
        # frame_shift_gcc = 320  # 311

        frame_size_gcc =  1024 # 4096
        frame_shift_gcc = 320  # 311
        f_max_gcc = None
        f_min = None
        modelbased = False  # True for model debug
        apply_ifft = False

        mic_selection = get_mic_selection(rec)
        print("MIC SELECTION: ", mic_selection, flush=True)
        data = data[mic_selection, :]  # take only first 4 channels
        data = pad_data(data)
        # ths = compute_vad_th(data)
        ths = 0

        # TODO: SET GCC STFT PARAMETERS
        print("DATA SHAPE: ", data.shape, frame_size_gcc, frame_shift_gcc, f_max_gcc,apply_ifft, flush=True)
        gcc_features = compute_gcc(data, f_min=f_min, f_max_gcc=f_max_gcc,  frame_size_gcc=frame_size_gcc, frame_shift_gcc=frame_shift_gcc,
                                   framewise=True, modelbased=modelbased, apply_ifft=apply_ifft)  # [frame, channel, search_range

        # a = 54
        # h5_filename = out_dir / f"gcc_features{a}.h5"
        # with h5py.File(h5_filename, 'r') as f:
        #     file_stem = Path(path).stem
        #     segment_id = f'{file_stem}'
        #     print(segment_id)
        #     if not isinstance(gcc_features, np.ndarray):
        #         gcc_features = gcc_features.cpu().numpy()
        #     # grp = f.require_group(file_stem)  # sturctue file_name: file_name_chunk_start_chunk_end...
        #     if file_stem in f:
        #         grp = f[file_stem]
        #     else:
        #         print(f"[Warning] {file_stem} not in f", flush=True)
        #         print("Groups in f:", list(f.keys()))
        #     if segment_id in grp:
        #         existing_gcc = grp[segment_id][:]
        #         if not np.allclose(existing_gcc, gcc_features, atol=1e-6):
        #             print(f"[Warning] GCC features unterscheiden sich für {segment_id}", flush=True)
        #         else:
        #             print(f"GCC features stimmen überein für {segment_id}", flush=True)
        #         print(np.array_equal(existing_gcc, gcc_features))
        #         print(np.allclose(existing_gcc, gcc_features, atol=1e-9))
        #         print("HALLO")
        #         print(np.allclose(existing_gcc, gcc_features, atol=1e-8))
        # assert False

        with h5py.File(h5_filename, 'a') as f:
            file_stem = Path(path).stem
            segment_id = f'{file_stem}'
            print(segment_id)

            if not isinstance(gcc_features, np.ndarray):
                gcc_features = gcc_features.cpu().numpy()
            if gcc_features is None or gcc_features.size == 0:
                print(f"[Warning] Empty gcc_features for {segment_id}", flush=True)
                raise ValueError(f"Empty gcc_features for {segment_id}")
            grp = f.require_group(file_stem)  # sturctue file_name: file_name_chunk_start_chunk_end...
            if segment_id in grp:
                print(f"[Warning] segment exists in grp: {grp} {segment_id}", flush=True)
                raise ValueError(f"segment_id {segment_id} already exists in group {grp}")

            grp.create_dataset(segment_id, data=gcc_features, compression="gzip")
            if file_stem not in index:
                print(f"Adding {file_stem} to json", flush=True)
                index[file_stem] = str(h5_filename)

    print(f"writing index file to {json_filename}", flush=True)
    dump_json(index, json_filename)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config toml file")
    parser.add_argument("--scp_file", type=str, required=True, help="Path to scp file")
    parser.add_argument("--uem_file", type=str, required=True, help="Path to uem file")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to out_dir")
    parser.add_argument("--dataset", type=str, required=True, help="which ddataset, train, test, dev")
    parser.add_argument("--f_max_gcc", required=True, help="which freq to filter, None for no upper limit")
    args = parser.parse_args()


    config_path = Path(args.config).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    scp_file = Path(args.scp_file)
    out_dir = Path(args.out_dir)
    uem_file = Path(args.uem_file)
    dataset = Path(args.dataset)

    if isinstance(args.f_max_gcc, str):
        if args.f_max_gcc.lower() == "none":
            f_max_gcc = None
        elif args.f_max_gcc.replace('.', '', 1).isdigit() or args.f_max_gcc.isdigit():
            f_max_gcc = float(args.f_max_gcc)
        else:
            assert False, f"Invalid value for f_max_gcc: {args.f_max_gcc}"
    else:
        f_max_gcc = None if args.f_max_gcc is None else float(args.f_max_gcc)


    precompute_gccs(config, scp_file, uem_file, out_dir, dataset, f_max_gcc)
