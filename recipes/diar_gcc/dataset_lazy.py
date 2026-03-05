# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
import math
import os
import random
from pathlib import Path
import re

import h5py

import itertools
import torch
import numpy as np
import paderbox as pb
import soundfile as sf
from typing import Dict

from torch.utils.data import get_worker_info
from paderbox.io import dump_json, load_json
from diarizen.spatial_features.gcc_phat import compute_vad_th, get_gcc_for_all_channel_pairs_torch
from diarizen.spatial_features.segmentation import spatial_segmentation
from torch.utils.data import Dataset, IterableDataset
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

def load_uem(uem_file: str, debug=False) -> Dict[str, float]:
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(uem_file):
        return None
    lines = [line.strip().split() for line in open(uem_file)]
    if debug:
        return {x[0]: [float(x[-2]), 120] for x in lines[:300]}
    else:
        return {x[0]: [float(x[-2]), float(x[-1])] for x in lines}
    
def _gen_chunk_indices(
    init_posi: int,
    data_len: int,
    size: int,
    step: int,
) -> None:
    # init_posi = int(init_posi + 1)
    # data_len = int(data_len - 1)
    init_posi = int(init_posi)
    data_len = int(data_len)
    cur_len = data_len - init_posi
    assert cur_len >= size, f"Data length {data_len} - {init_posi} too short for chunk size {size}."
    num_chunks = int((cur_len - size + step) / step)

    for i in range(num_chunks):
        yield init_posi + (i * step), init_posi + (i * step) + size


def _gen_segment_indices(
        init_posi: int,
        data_len: int,
        size: int,
        step: int,
) -> None:
    start = int(init_posi)
    end_limit = int(data_len)
    while start < end_limit:
        end = start + int(size)
        if end >= end_limit:
            # Rest-Chunk (kürzer als size) auch liefern
            yield start, end_limit
            break
        yield start, end
        start += int(step)

def _collate_fn(batch, max_speakers_per_chunk=4, th=0.15, gcpsd=False) -> torch.Tensor:
    collated_x = []
    collated_y = []
    collated_names = []
    gccs = []
    num_spks = []
    ids = []
    # print(acc.process_index, f'Collate fn called with batch size {len(batch)}', flush=True)
    # print(acc.device, flush=True)

    # # move to device manually because we use lazy dataset
    # batch = {k: v.to(accelerator.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # for x, y, name, gcc in batch:
    for b in batch:
        # print("Example ID:", f"recording_{b['rec']}_{b['start']}", flush=True)
        id = f"{b['name']}_{b['start']}"
        x = b['data']#.to(accelerator.device)
        y = b['mask_label']#.to(accelerator.device)
        name = b['name'][0]
        gcc = b['gcc_features']#.to(accelerator.device)
        # print(f'Processing {name} | {path} | {session_idx}')
        num_speakers = y.shape[-1]
        num_spk = np.sum(y, axis=-1, keepdims=True)


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
        ids.append(id)
        gccs.append(gcc)
        num_spks.append(num_spk)
    # print(acc.process_index, collated_names[0], flush=True)
    if gcpsd:
        try:
            return {
                'xs': torch.from_numpy(np.stack(collated_x)).float(),
                'ts': torch.from_numpy(np.stack(collated_y)),
                "ids": ids,
                # 'names': collated_names,
                # "names": {"items": collated_names},
                "gccs": torch.stack(gccs).to(torch.complex64),
                "num_spks": torch.from_numpy(np.stack(num_spks).astype(np.float32)).float(),
            }
        except Exception as e:
            print(f"Error in collate_fn with gcpsd: {e}")
            print(f"collated_x shapes: {[x.shape for x in collated_x]}")
            raise e

    else:
        # try:
        #     tmp = np.stack(collated_x)
        # except Exception as e:
        #     print(f"Error in collate_fn with gcpsd: {e}")
        #     print(f"collated_x shapes: {[x.shape for x in collated_x]}")
        #     raise e
        return {
            'xs': torch.from_numpy(np.stack(collated_x)).float(),
            'ts': torch.from_numpy(np.stack(collated_y)),
            "ids": ids,
            # 'names': collated_names,
            # "names": {"items": collated_names},
            "gccs": torch.from_numpy(np.stack(gccs)).float(),
            "num_spks": torch.from_numpy(np.stack(num_spks).astype(np.float32)).float(),
        }


class IterableWrapper(IterableDataset):
    def __init__(self, dataset, len=None, rec_list=None):
        self.dataset = dataset
        self.get_my_length = len
        self.distributed = False
        self.rec_list = rec_list

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is not None and worker_info.num_workers > 1:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # # Debug: Zähle Aufrufe
            # if worker_id not in self._iter_count:
            #     self._iter_count[worker_id] = 0
            # self._iter_count[worker_id] += 1
            #
            # print(f"DDP Rank {self.dataset.rank}/{self.dataset.world}__Worker {worker_id}: __iter__ aufgerufen (Aufruf #{self._iter_count[worker_id]})",
            #       flush=True)
            #
            # # Initialisiere nur einmal pro Worker
            # if worker_id not in self._worker_datasets:
            #     print(f"Worker {worker_id}: Initialisiere Dataset", flush=True)
            #     self._worker_datasets[worker_id] = self.dataset.init_ds(
            #         worker_id, num_workers, self.rec_list
            #     ).dataset
            #
            # ds = self._worker_datasets[worker_id]
            print(f"Worker {worker_id}",flush=True) # !! doppelter aufruf der iter funktion aber nur einmal in trainings liste, nicht wundern
            ds = self.dataset.init_ds(worker_id, num_workers, self.rec_list).dataset
        else:
            ds = self.dataset
        rank_iter = iter(ds)
        return rank_iter
        # return iter(self.dataset)

    def __len__(self):
        # if self.cut is not None:
        #     self.get_my_length = self.cut
        return self.get_my_length


from lazy_dataset import from_list

class DiarizationLazy:
    def __init__(
            self,
            scp_file: str,
            rttm_file: str,
            uem_file: str,
            model_num_frames: int,  # default: wavlm_base
            model_rf_duration: float,  # model.receptive_field.duration, seconds
            model_rf_step: float,  # model.receptive_field.step, seconds
            chunk_size: int = 5,  # seconds
            chunk_shift: int = 5,  # seconds
            sample_rate: int = 16000,
            channel_mode: str = "multichannel",  # sdm, random, average, multichannel
            load_gcc_dir=None,
            subset="train",
            num_channels=4,  # number of channels for multichannel mode
            num_spk=False,
            modelbased=False,
            gcpsd=False,
            acc = None,
            sub_rec = True,
            buffer_size = 500, # for shuffling # 300 first time
            segment_size = 10 * 60,   # in seconds
            segment_overlap = 0,         # in seconds
            resample = False,
            rotate = False,
            debug=False,
            num_workers=0,
            gradient_accumulation_steps=1,
    ):
        self.chunk_indices = []
        self.subset = subset
        self.buffer_size = buffer_size
        self.debug = debug
        self.sub_rec = sub_rec
        self.segment_size = segment_size
        self.segment_overlap = segment_overlap

        self.sample_rate = sample_rate
        self.chunk_sample_size = sample_rate * chunk_size
        self.chunk_size = chunk_size
        self.chunk_shift = chunk_shift

        self.channel_mode = channel_mode

        self.model_rf_step = model_rf_step
        self.model_rf_duration = model_rf_duration
        self.model_num_frames = model_num_frames

        self.rec_scp = load_scp(scp_file)
        self.reco2dur = load_uem(uem_file, debug=debug)
        self.load_gcc_dir = load_gcc_dir
        self.num_channels = num_channels
        self.num_spk = num_spk
        self.modelbased = modelbased
        self.gcpsd = gcpsd
        self.resample = resample
        self.rotate = rotate

        self.annotations = self.rttm2label(rttm_file)
        self.energy_th = {}

        self.num_workers = num_workers
        self._length = 0
        # if acc is not None:
        self.world = acc.num_processes
        self.rank = acc.process_index
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # else:
        #     world = 1
        #     rank = 0

        if self.num_workers > 1:
            if self.sub_rec:
                # TODO: hier potentiell sub-recordings machen
                chunk_indices = []
                chunk_size_tmp = self.segment_size  # 10 min chunks jeweils
                chunk_shift_tmp = self.segment_size - self.segment_overlap
                # durations = {x:0 for x in range(world)}
                # counter = 0
                for rec, dur_info in self.reco2dur.items():
                    start_sec, end_sec = dur_info
                    start_sec = start_sec + 1  # remove first s to avoid hard zeros
                    end_sec = end_sec - 1  # remove last s to avoid hard zeros
                    # print(end_sec - start_sec, flush=True)
                    if end_sec - start_sec >= chunk_size_tmp:
                        try:
                            for st, ed in _gen_segment_indices(start_sec, end_sec, chunk_size_tmp, chunk_shift_tmp):
                                chunk_indices.append((rec, (st, ed)))  # seconds
                                # print(f'asdChunked {rec} from {st} to {ed}', flush=True)
                                # durations[counter % world] += ed - st
                                # counter +=1
                                # print(durations, sum(durations.values()), flush=True)
                        except Exception as e:
                            print(f'Un-matched recording: {rec}', e)
                    else:
                        assert end_sec - start_sec >= self.chunk_size, f"Recording {rec} too short even for sub recording: {end_sec}, - {start_sec} <= {self.chunk_size}"
                        chunk_indices.append((rec, (start_sec, end_sec)))  # seconds
                        # durations[counter % world] += end_sec - start_sec
                        # counter +=1
                    # assert False
            else:
                chunk_indices = self.reco2dur.items()

            rec_list = [(rec, dur) for i, (rec, dur) in enumerate(chunk_indices) if (i % self.world == self.rank and dur[1] - dur[0] >= self.chunk_size)]  # List of (sub-) recordings, each as (rec, dur)

            # TODO: Seed festlegen, damit reproduzierbar geshufflet wird
            if self.subset == "train":
                random.shuffle(rec_list)
            dataset_length = sum([
                max(0, int((dur[1] - dur[0] - self.chunk_size + self.chunk_shift) / self.chunk_shift))  ##  - 2
                for rec, dur in rec_list
            ])
            lengths = [None] * self.world
            if self.world > 1:
                torch.distributed.all_gather_object(lengths, dataset_length)  # len(rec_list) ) #
            else:
                lengths = [dataset_length]
            print(lengths, flush=True)
            self.max_len = max(lengths)
            num = sum(lengths)
            delta = self.max_len  - dataset_length
            print("Max length:", self.max_len , "number of chunks:", num, flush=True)

            rec_list = self.extend_by_recording(rec_list, delta)
            self.lazy = IterableWrapper(dataset=self, len=self.max_len * self.gradient_accumulation_steps, rec_list=rec_list)
            return
        else:
            self.lazy = self.init_ds()
            return

    def init_ds(self, worker_id=0, total_num_workers=0, rec_list=None):
        if not total_num_workers > 1:
            if self.sub_rec:
                # TODO: hier potentiell sub-recordings machen
                chunk_indices = []
                chunk_size_tmp = self.segment_size  # 10 min chunks jeweils
                chunk_shift_tmp = self.segment_size - self.segment_overlap
                # durations = {x:0 for x in range(world)}
                # counter = 0
                for rec, dur_info in self.reco2dur.items():
                    start_sec, end_sec = dur_info
                    start_sec = start_sec + 1 # remove first s to avoid hard zeros
                    end_sec = end_sec - 1   # remove last s to avoid hard zeros
                    # print(end_sec - start_sec, flush=True)
                    if end_sec - start_sec >= chunk_size_tmp:
                        try:
                            for st, ed in _gen_segment_indices(start_sec,end_sec, chunk_size_tmp,chunk_shift_tmp):
                                chunk_indices.append((rec, (st, ed)))  # seconds
                                # print(f'asdChunked {rec} from {st} to {ed}', flush=True)
                                # durations[counter % world] += ed - st
                                # counter +=1
                                # print(durations, sum(durations.values()), flush=True)
                        except Exception as e:
                            print(f'Un-matched recording: {rec}', e)
                    else:
                        assert end_sec - start_sec >= self.chunk_size , f"Recording {rec} too short even for sub recording: {end_sec}, - {start_sec} <= {self.chunk_size}"
                        chunk_indices.append((rec, (start_sec,end_sec)))  # seconds
                        # durations[counter % world] += end_sec - start_sec
                        # counter +=1
                    # assert False
            else:
                chunk_indices = self.reco2dur.items()


            rec_list = [(rec,  dur) for i, (rec, dur) in enumerate(chunk_indices) if (i % self.world == self.rank and dur[1] - dur[0] >= self.chunk_size)]   # List of (sub-) recordings, each as (rec, dur)

            # TODO: Seed festlegen, damit reproduzierbar geshufflet wird
            if self.subset == "train":
                random.shuffle(rec_list)
            # # # shorten for testing pruposes
            # rec_list = rec_list[::2]

            # # skip
            # skipping = True
            # if skipping:
            #     skip = int(2250  * 16 / 99)
            #     rec_list = rec_list[skip:]

            # len3 = sum([
            #     max(1, int(math.floor(((dur[1] - dur[0]) - self.chunk_size) / self.chunk_shift) +1))
            #     for rec, dur in rec_list
            # ])
            # len2 = sum([
            #     int((dur[1] - dur[0] - self.chunk_size + self.chunk_shift) / self.chunk_shift)
            #     for rec, dur in rec_list
            # ])
            # len4 = sum([
            #     int((dur[1] - dur[0] -2 - self.chunk_size + self.chunk_shift) / self.chunk_shift)
            #     for rec, dur in rec_list
            # ]) + self.chunk_size
            # len3 = sum([
            #     max(1,int((dur[1] - dur[0] -2 - self.chunk_size + self.chunk_shift) / self.chunk_shift))
            #     for rec, dur in rec_list
            # ])
            dataset_length = sum([
                max(0,int((dur[1] - dur[0] - self.chunk_size + self.chunk_shift) / self.chunk_shift))  ##  - 2
                for rec, dur in rec_list
            ])

            # # # ------------------------------------
            # # OLD TEMP JUST TO COMPARE LENGTHS
            # # # Chunk the segments into 8 second chunks,
            # rec_list_tmp, durations_chunks, counter = self.chunk_recordings(rec_list)
            # assert dataset_length >= len(rec_list)
            # print(f"Rank {rank}: {counter} chunks, computed len: {dataset_length}", flush=True)
            # # # print("alles gleich?", rank, counter*8, len(rec_list) * 8, durations_chunks, flush=True)


            lengths = [None] * self.world
            if self.world > 1:
                torch.distributed.all_gather_object(lengths, dataset_length) # len(rec_list) ) #
            else:
                lengths = [dataset_length]
            print(lengths, flush=True)
            self.max_len  = max(lengths)
            num = sum(lengths)
            delta = self.max_len  - dataset_length
            print("Max length:", self.max_len , "number of chunks:", num, flush=True)

            rec_list = self.extend_by_recording(rec_list, delta)

        else: # total_num_workers > 1:
            # print(f"worker number {worker_id}", rec_list[:10], flush=True)
            rec_list = [(rec, dur) for i, (rec, dur) in enumerate(rec_list)
                        if (i % total_num_workers == worker_id)]
            # print(f"worker number {worker_id}", rec_list[:10], flush=True)
            # print(f"\n{self.rank}___{worker_id}__RECORDING LIST:\n", rec_list, flush=True)
            # # Länge muss berechnet werden damit die über die
            # rec_list_0 = [(rec, dur) for i, (rec, dur) in enumerate(rec_list)
            #
            #
            #             if (i % self.num_workers == 0 and dur[1] - dur[0] >= self.chunk_size)]
            # rec_list_1 = [(rec, dur) for i, (rec, dur) in enumerate(rec_list)
            #             if (i % self.num_workers == 1 and dur[1] - dur[0] >= self.chunk_size)]

        # # rec_list.extend(rec_list[:delta])
        # rec_list_tmp, durations_chunks, counter = self.chunk_recordings(rec_list)
        #
        # lengths2 = [None] * world
        # torch.distributed.all_gather_object(lengths2, len(rec_list_tmp) ) #
        # print(lengths2, flush=True)

        # ------------------------------------

        # # old
        # filtered_lazy = from_list(rec_list)
        # filtered_lazy = filtered_lazy.shuffle(buffer_size=300, reshuffle=True)
        # filtered_lazy = filtered_lazy.map(self.extract_wavforms)
        # filtered_lazy = filtered_lazy.map(self.get_chunk_labels)
        # filtered_lazy = filtered_lazy.map(self.get_spatial_features)
        # self.lazy = filtered_lazy.map(self.to_dict)
        # self.lazy = IterableWrapper(self.lazy, len=world * len(rec_list))
        filtered_lazy = from_list(rec_list)
        filtered_lazy = filtered_lazy.map(self.extract_wavforms_and_chunk)
        filtered_lazy = filtered_lazy.unbatch()
        filtered_lazy = filtered_lazy.map(self.get_chunk_labels)
        filtered_lazy = filtered_lazy.map(self.get_spatial_features)
        lazy = filtered_lazy.map(self.to_dict)
        if self.subset == "train" and not self.debug:
            lazy = lazy.shuffle(buffer_size=self.buffer_size, reshuffle=True)
        # print('Final length:', self.max_len , flush=True)
        if self.rotate:
            lazy = lazy.map(self.rotate_channels)
        self.lazy = IterableWrapper(lazy, len=self.max_len * self.gradient_accumulation_steps) #, cut_to=32) # world *
        # self.lazy = MapWrapper(lazy, len=world)
        self._length = self.max_len

        return self.lazy




        # # lazy = lazy.shuffle()
        #
        # # # filter out to keep only those examples whose index fits the gpu rank
        # # filtered_lazy = lazy.filter(lambda x, i: x[0] == rank)
        # # filtered_lazy = filtered_lazy.map(self.get_chunks)
        # # filtered_lazy = filtered_lazy.unbatch()
        #
        # # TODO: Vtl RAM erhöhen wenn shuffle buffer RAM zieht, RAM explodiert wenn alle 300 verschiedene rec ursprünge haben??
        # filtered_lazy = filtered_lazy.shuffle(buffer_size=300, reshuffle=True)

        # filtered_lazy = filtered_lazy.map(self.extract_wavforms)
        # filtered_lazy = filtered_lazy.map(self.get_chunk_labels)
        # filtered_lazy = filtered_lazy.map(self.get_spatial_features)
        # self.lazy = filtered_lazy.map(self.to_dict)
        #
        # self.lazy = IterableWrapper(self.lazy, len=world * len(rec_list))

    def __len__(self):
        return self._length

    def rotate_channels(self, example):
        if self.rotate:
            rotation = random.randint(0, self.num_channels -1)
            ch_idx = (np.arange(example['data'].shape[0]) + rotation) % example['data'].shape[0]
            example['data'] = example['data'][ch_idx, :]
            example['ch_idx'] = ch_idx
        return example

    def extend_by_recording(self, rec_list, delta):
        extended = rec_list.copy()
        current_delta = 0
        idx = 0
        n_rec = len(rec_list)
        while current_delta < delta:
            rec, dur = rec_list[idx % n_rec]
            L = dur[1] - dur[0]
            num_chunks = max(0, int((L - self.chunk_size + self.chunk_shift) / self.chunk_shift))  #  -2
            space_left = delta - current_delta

            if num_chunks <= space_left:
                # append full recording
                extended.append((rec, dur))
                current_delta += num_chunks
            else:
                # append only as much of this recording as needed to reach exactly `delta` chunks
                needed = space_left
                # minimal length to produce `needed` chunks:
                L_needed = (needed - 1) * self.chunk_shift + self.chunk_size  # + 2
                new_end = dur[0] + L_needed
                # never exceed original end
                new_end = min(new_end, dur[1])
                extended.append((rec, (dur[0], new_end)))
                assert new_end - dur[0] >= self.chunk_size , f"Not enough length to produce needed chunks: {new_end} - {dur[0]} < {self.chunk_size}, needed still {needed}"
                current_delta += needed
                break

            idx += 1

        print(f'Extended by {current_delta} chunks to match max length {delta}, rank: ', self.rank, flush=True)
        return extended


    def extract_wavforms_and_chunk(self, example):
        # TODO: get list of starts and end and then [chunk for st,end in ...] und unbtahc
        # rec , path, start, end = example["rec"], example["audio_path"], example["start"], example["end"]
        rec, (start, end) = example
        path = self.rec_scp[rec]
        # # 4 for debugging and smaller memory and faster dev
        # start = int(start * self.sample_rate)
        # # TODO' random channel selection and augmentation
        # end = int(end * self.sample_rate)
        # # # if system is noctua, change path to noctua2
        if not os.path.exists(path):
            # IF not on Merlin try Noctua2
            # path = path.replace("/mnt/*/AMI_AIS_ALI_NSF_CHiME7",
            #                     "/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7")
            path = re.sub(r"^/mnt/[^/]+/AMI_AIS_ALI_NSF_CHiME7", "/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7", path)
        if not os.path.exists(path):
            # IF not on noctua try on NT network
            path = re.sub(r"/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7", "/net/vol/deegen/db/AMI_AIS_ALI_NSF_CHiME7", path)
        try:
            # load sub recording segment
            start_samples = int(start * self.sample_rate)
            end_samples = int(end * self.sample_rate)
            rec_len = end- start
            data, sample_rate = sf.read(path, start=start_samples, stop=end_samples)
        except Exception as e:
            print(f"Error reading {path} from {start} to {end}: {e}")
            raise RuntimeError(f"Error reading {path} from {start} to {end}: {e}")
        assert sample_rate == self.sample_rate, f"Sample rate mismatch: {sample_rate} != {self.sample_rate}"

        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = np.einsum('tc->ct', data)
        # p(data.shape)
        assert data.shape[-1] <= rec_len * self.sample_rate, f"Data length mismatch: {data.shape[-1]} != {rec_len * self.sample_rate}, {path}, {start}, {end}"
        data = data[self.get_mic_selection(Path(path).stem), :]
        # generate chunks for this sub_recording
        # if end - start < self.chunk_size+2 :
        #     print(f"{self.rank} Sub-recording too short for chunking: {rec}, {start}, {end}, length: {end - start}", flush=True)
        if end - start >= self.chunk_size :   #  TODO: remove +2 and adjust the same in get_chunk_indices or find reason why chunks should be 2 seconds longer alwyys?
            examples = []
            # TODO: start und end sind noch absolut, muss auf neue segmente bezogen werden!!
            for st, ed in _gen_chunk_indices(0, rec_len, self.chunk_size, self.chunk_shift):
                # print(f'Chunked {rec} from {st} to {ed}', flush=True)
                # TODO: START UND END SIND NICHT MIT GLOBAL SONDEN NUR LOKAL RICHTIG!!!
                # if self.rotate:
                #     rotation = random.randint(0, self.num_channels -1)
                #     # data_rotated = np.roll(data, shift=rotation, axis=0)
                # else:
                #     # data_rotated = data
                #     rotation = 0
                # ch_idx = (np.arange(data.shape[0]) + rotation) % data.shape[0]
                examples.append({
                    "rec": rec,
                    "audio_path": path,
                    "global_start": start,
                    "start": start + st,
                    "end": start + ed,
                    "data": data[:, int(st*self.sample_rate):int(ed*self.sample_rate)],
                    # "data": data_rotated[:, int(st*self.sample_rate):int(ed*self.sample_rate)],
                    # "ch_idx": ch_idx,
                })
            self._length += len(examples)

        # else:
        #     # TODO: neccessary or not?
        #     examples = [{"rec": rec, "audio_path": path, "start": start, "end": start+self.chunk_size,
        #                  "data": data[:, int(start*self.sample_rate):int((start+self.chunk_size)*self.sample_rate)]}]
        # # rec_list, durations_chunks, counter
        # self._length += len(examples)

        # if self.debug:
        #     # get random example for debugging, with fixed seed for reproducibility
        #     random.seed(42)
        #     examples = [random.choice(examples)]
        return examples

    def chunk_recordings(self, rec_list):
        chunk_indices = []
        durations_chunks = 0
        counter = 0
        for rec, dur_info in rec_list:
            start_sec, end_sec = dur_info
            # if end_sec - start_sec >= self.chunk_size:
            try:
                for st, ed in _gen_chunk_indices(start_sec,end_sec, self.chunk_size, self.chunk_shift):
                    chunk_indices.append({"rec": rec, "audio_path" : self.rec_scp[rec], "start": st, "end": ed})  # seconds
                    durations_chunks += ed - st
                    assert ed - st == self.chunk_size, f"Chunk size mismatch: {ed - st} != {self.chunk_size}, {rec}, {st}, {ed}"
                    counter +=1
            except Exception as e:
                # print(end_sec - start_sec, start_sec, end_sec, ed-st, ed, st, self.chunk_size, self.chunk_shift)
                print(f'456Un-matched recording: {rec}', end_sec , start_sec, e)
            # else:
            #     chunk_indices.append(
            #         {"rec": rec, "audio_path": self.rec_scp[rec], "start": start_sec, "end": end_sec})  # seconds
            #     durations_chunks += end_sec - start_sec
            #     counter +=1
        return chunk_indices, durations_chunks, counter

    def count_active_speakers_intervals(self, chunk_start, chunk_end, chunked_annotations):
        """
        chunk_start, chunk_end: float (Sekunden)
        chunked_annotations: np.array mit Feldern start, end, label_idx
                             (bereits auf Session und Chunk gefiltert)
        """
        # Intervalle auf Chunk zuschneiden
        intervals = np.stack([
            np.maximum(chunked_annotations["start"], chunk_start),
            np.minimum(chunked_annotations["end"], chunk_end),
        ], axis=1)

        # Events für Sweep-Line
        events = []
        for s, e in intervals:
            if e > s:  # nur wenn noch Länge bleibt
                events.append((s, +1))
                events.append((e, -1))

        # Sweep-Line Algorithmus
        events.sort()
        active = 0
        max_active = 0
        for _, delta in events:
            active += delta
            max_active = max(max_active, active)

        return max_active

    def get_chunks(self, example):
        i, rec, dur_info = example
        chunk_indices = []
        start_sec, end_sec = dur_info
        try:
            if self.chunk_size > 0:
                for st, ed in _gen_chunk_indices(start_sec, end_sec, self.chunk_size, self.chunk_shift):
                    chunk_indices.append((rec, self.rec_scp[rec], st, ed))
            else:
                chunk_indices.append((rec, self.rec_scp[rec], start_sec, end_sec))  # recording id, audio path(?), seconds
        except Exception as e:
            print(f'Un-matched recording: {rec}', e)
        chunked_example = [{
            "rec" : rec,
            "audio_path" : audio_path,
            "start_sec" : start_sec,
            "end_sec" : end_sec}
            for rec, audio_path, start_sec, end_sec in chunk_indices]
        return chunked_example

    def get_chunk_labels(self, example):
        session, chunk_start, chunk_end = example['rec'], example['start'], example['end'] # , example['global_start']
        # chunk_start = global_start + chunk_start
        # chunk_end = global_start + chunk_end
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
            mask_label[start: end + 1, mapped_label] = 1

        example['mask_label'] = mask_label

        # # for debug plot the labels and save the audio on disk
        # import matplotlib.pyplot as plt
        # import os
        # os.makedirs('exp/spk_count_ref_n2_lazy_global_starts/debug_labels', exist_ok=True)
        # plt.figure(figsize=(10, 4))
        # plt.imshow(mask_label.T, aspect='auto', origin='lower', interpolation='nearest')
        # plt.title(f'Labels for {session} from {chunk_start} to {chunk_end}')
        # plt.xlabel('Frame index')
        # plt.ylabel('Speaker index')
        # plt.colorbar(label='Activity')
        # plt.tight_layout()
        # plt.savefig(f'exp/spk_count_ref_n2_lazy_global_starts/debug_labels/{session}_{chunk_start}_{chunk_end}_labels_{self.rank}.png')
        # plt.close()
        # sf.write(f'exp/spk_count_ref_n2_lazy_global_starts/debug_labels/{session}_{chunk_start}_{chunk_end}_audio_{self.rank}.wav',
        #          example['data'].T, self.sample_rate)

        return example

    def get_spatial_features(self, example):
        session, path, chunk_start, chunk_end, data = example['rec'], example['audio_path'], example['start'], example['end'], example['data']#, example['ch_idx']
        # data = data[ch_idx, :]  # apply channel rotation if any

        if self.load_gcc_dir == "base":
            gcc_features = torch.zeros(1, 1, 1)
        elif self.load_gcc_dir:
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
            gcc_features = self.compute_gcc(data_pad, frame_size_gcc=fft_size, frame_shift_gcc=320, f_max_gcc=fmax,
                                            f_min=fmin,
                                            apply_ifft=apply_ifft)


            if magnitude is not None:
                # TODO: MAGNITUDE IN GCC ÜBERGEBEN und überall anpassen
                gcc_features = torch.concat([gcc_features, magnitude[:, None, :]], dim=1)
        else:
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


            gcc_features = self.compute_gcc(data_pad, frame_size_gcc=fft_size, frame_shift_gcc=320, f_max_gcc=fmax,
                                            f_min=fmin,
                                            apply_ifft=apply_ifft)
            if magnitude is not None:
                gcc_features = torch.concat([gcc_features, magnitude[:, None, :]], dim=1)
        example['gcc_features'] = gcc_features
        return example

    def to_dict(self, ex):
        # print(f"DDP Worker: {self.rank}", ex["rec"], flush=True)
        return {
            "data": ex["data"],
            "mask_label": ex["mask_label"],
            "name": ex["rec"],
            "gcc_features": ex["gcc_features"],
            "start": ex["start"],
        }

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
        sigs_stft = torch.from_numpy(sigs_stft)  # (frames, channels, freq)
        gcc_features = get_gcc_for_all_channel_pairs_torch(sigs_stft, f_min=f_min, f_max=f_max_gcc, apply_ifft=apply_ifft)
        return gcc_features


    def get_mic_selection(self, rec):
        if rec.startswith(("S3")):
            mics = [1, 3, 4, 6]  # for NSF
        else:
            mics = [0, 2, 4, 6]  # default
        return mics


    def extract_wavforms(self, example):
        num_channels = self.num_channels
        rec , path, start, end = example["rec"], example["audio_path"], example["start"], example["end"]
        # 4 for debugging and smaller memory and faster dev
        start = int(start * self.sample_rate)
        # TODO' random channel selection and augmentation
        end = int(end * self.sample_rate)
        assert end-start == self.chunk_sample_size, f"Chunk size mismatch: {end - start} != {self.chunk_sample_size}, {rec}, {start}, {end}"
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

        example["data"] = data[self.get_mic_selection(Path(path).stem), :]
        assert example["data"].shape[1] == self.chunk_sample_size, f"Data shape mismatch: {example['data'].shape[1]} != {self.chunk_sample_size}"
        return example


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
