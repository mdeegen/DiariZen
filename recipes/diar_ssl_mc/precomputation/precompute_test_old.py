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


def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def compute_gcc(waveforms_mc, frame_size_gcc=400, frame_shift_gcc=320, avg_len_gcc=4,
                search_range_gcc=10, f_max_gcc=3500, f_min=125, ths=[], framewise=False):
    """
    Compute GCC features from multichannel waveforms.
    returns:
        batch_gcc_features: (batch, frame, channel, search_range)
    """
    # TODO: try different stft values for better gcc but need fit frames of WAVLM
    sigs_stft = pb.transform.stft(waveforms_mc, frame_size_gcc, frame_shift_gcc,
                                  pad=False, fading=False)
    voice_activity = channel_wise_activities(waveforms_mc, ths=ths)
    frame_wise_voice_activity = convert_to_frame_wise_activities(
        voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
    )
    dominant = get_dominant_time_frequency_mask(sigs_stft)
    gcc_features = get_gcc_for_all_channel_pairs(
        sigs_stft, frame_wise_voice_activity, dominant=dominant, search_range=search_range_gcc, f_min=f_min,
        f_max=f_max_gcc, avg_len=avg_len_gcc, framewise=framewise
    )
    # # os makedir data/gccs if not exists
    # path = store_gcc(gcc_features,)
    # def store_gcc(gcc_features, path=None):
    #     """Store gcc features to file and return path"""
    #     np.save(path, gcc_features)

    return gcc_features

def precompute_gccs(config, scp_file, uem_file, out_dir, dataset):
    print("PYTHON: Precomputing GCCS", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    worker_id = os.environ.get("SGE_TASK_ID", "0")
    h5_filename = out_dir / f"gcc_features{worker_id}.h5"
    json_filename = out_dir / f"gcc_features_index{worker_id}.json"
    index = {}

    duration = 8
    step = 0.1
    num_channels = 4

    rec_scp = load_scp(scp_file)
    energy_th = {}
    for rec in rec_scp.keys():
        import time
        start = time.perf_counter()

        path = rec_scp[rec]
        data, sample_rate = sf.read(path)
        data = np.einsum('tc->ct', data)  # [channel, time]
        data = data[:4, :]  # take only first 4 channels # Todo: num channels
        ths = compute_vad_th(data)

        # TODO: SET GCC STFT PARAMETERS
        gcc_features = compute_gcc(data, ths=ths, framewise=True)

        # Your code here
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.4f} seconds", flush=True)

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
            # Your code here
            end2 = time.perf_counter()
            print(f"ROund time: {end2 - end:.4f} seconds", flush=True)

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

    precompute_gccs(config, scp_file, uem_file, out_dir, dataset)
