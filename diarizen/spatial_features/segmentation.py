import itertools
from copy import deepcopy

import numpy as np
import os
import paderbox as pb
from paderwasn.synchronization.utils import VoiceActivityDetector
import pdb

from scipy.signal import fftconvolve
# import dlp_mpi
from einops import rearrange
from numpy.fft import rfft, irfft
from scipy import signal

import sys
import inspect
import torch
import psutil
from tqdm import tqdm
from scipy.signal import find_peaks
from spatiospectral_diarization.spatial_diarization.utils import get_position_candidates
from spatiospectral_diarization.spatial_diarization.cluster import temporally_constrained_clustering


def spatial_segmentation(gcc_features, avg_len=4, shift=320):
    tdoa_candidates = get_candidates(gcc_features, num_peaks=5,max_diff=2, upsampling=10, p_th=0.75,
                            max_concurrent=3, distributed=False)
    segments = temporally_constrained_clustering(tdoa_candidates, **{'max_dist': 0.75, 'peak_ratio_th':.5,  # 0.75, .5, 50
                               'max_temp_dist':50})
    segments = segments[::-1]  # ouput of temporal clustering begins with the last segment in the meeting
    assert shift is not None
    num_frames = gcc_features.shape[0]
    segments, segment_tdoas = merge_overlapping_segments(segments, num_frames, avg_len_gcc=avg_len, min_cl_segment=3,
                                                         distributed=False, max_diff_tmp_cl=1, shift=shift)
    spks = np.zeros((num_frames))
    for seg in segments:
        for interval in seg.normalized_intervals:
            spks[interval[0]:(interval[1]+1)] += 1
    return np.array(spks, int)

def get_ch_pairs(num_chs):
    ch_pairs = []
    for i in range(num_chs):
        for j in range(i + 1, num_chs):
            ch_pairs.append((i, j))
    return ch_pairs

def get_candidates(gcc_features, num_peaks=5,max_diff=2, upsampling=10, p_th=0.75,
                            max_concurrent=3, distributed=False):
    assert num_peaks >= max_concurrent

    num_ch_pairs = gcc_features.shape[1]
    search_range = gcc_features.shape[-1]

    if num_ch_pairs == 6:
        num_chs = 4
    elif num_ch_pairs == 28:
        num_chs = 8
    else:
        raise NotImplementedError

    ch_pairs = get_ch_pairs(num_chs)
    lags = np.arange(-search_range, search_range + 1 / upsampling, 1 / upsampling)
    candidates = []
    for l in range(gcc_features.shape[0]):
        gccs = []
        peak_tdoas = []
        peaks = []
        for k, (ref_ch, ch) in enumerate(ch_pairs):

            # todo: hier search range einfach laden
            search_area = gcc_features[l, k]

            th =  np.maximum(p_th * np.max(search_area), 0)# 2 * np.sqrt(np.mean(search_area[search_area > 0] ** 2))    #

            peaks_pair, _ = find_peaks(search_area)
            peaks_pair = np.asarray(peaks_pair)
            peaks_pair = peaks_pair[search_area[peaks_pair] >= th]
            choice = np.argsort(search_area[peaks_pair])[::-1][:num_peaks]
            peaks_pair = peaks_pair[choice]
            peaks.append(peaks_pair)
            for p, peak in enumerate(peaks_pair):
                if p+1 > len(gccs):
                    peak_tdoas.append(-1000*np.ones((num_chs, num_chs)))
                    gccs.append(np.zeros((num_chs, num_chs)))
                peak_tdoas[p][ref_ch, ch] = lags[peak]
                peak_tdoas[p][ch, ref_ch] = -lags[peak]
                gccs[p][ref_ch, ch] = gccs[p][ch, ref_ch] = search_area[peak]
        srps = []
        for combination in itertools.product(*[np.arange(len(p)) for p in peaks]):
            t = np.zeros((num_chs, num_chs))
            for k, (ref_ch, ch) in enumerate(ch_pairs):
                t[ref_ch, ch] = peak_tdoas[combination[k]][ref_ch, ch]
                t[ch, ref_ch] = - t[ref_ch, ch]
            valid = True
            for k, (ref_ch, ch) in enumerate(ch_pairs):
                for m in range(num_chs):
                    if m== ref_ch or m== ch:
                        continue
                    if np.max(abs(t[m, ch] + t[ref_ch, m] - t[ref_ch, ch])) > max_diff:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                srp = 0
                taus = []
                for k, (ref_ch, ch) in enumerate(ch_pairs):
                    srp += gccs[combination[k]][ref_ch, ch]
                for ref_ch in range(num_chs):
                    for ch in range(ref_ch+1, num_chs):
                        taus.append(t[ref_ch, ch])
                if distributed:
                    srps.append((taus,srp))
                elif np.any(np.abs(taus) >= .5):
                    srps.append((taus,srp))
        srps = sorted(srps, key=lambda ex: ex[-1], reverse=True)
        spk_pos = []
        for i in range(max_concurrent):
            if len(srps) == 0:
                break
            new_pos = srps[0]
            spk_pos.append(new_pos)
            taus = new_pos[0].copy()
            to_keep = []
            for srp in srps[1:]:
                t, _ = srp
                if np.sum(abs(np.asarray(t) - np.asarray(taus)) <= .3) <= 2:
                    to_keep.append(srp)
            srps = to_keep
        candidates.append((l, spk_pos))
    return candidates

def merge_overlapping_segments(temp_diary, recording_length, avg_len_gcc, min_cl_segment, distributed, max_diff_tmp_cl, shift):
    """
    Merges overlapping segments from the same direction. For each segment, the corresponding activity interval and the
    median TDOA (Time Difference of Arrival).
    Args:
        temp_diary (list): List of segment entries, each containing TDOA values and frame indices.
        sig_len (int): Length of the signal in samples.
        avg_len_gcc (int): Average length of GCC (Generalized Cross-Correlation)-Buffer calculation.
        min_cl_segment (int): Minimum number of frames required for a segment to be considered.
        distributed (bool): Whether the setup is distributed (affects segment filtering).
        max_diff_tmp_cl (float): Maximum allowed difference between median TDOAs for merging.
    Returns:
        tuple: (segments, seg_tdoas)
            segments (list): List of activity intervals for each merged segment.
            seg_tdoas (list): List of median TDOA values for each merged segment.
    """
    temp_diary_ = deepcopy(temp_diary)
    seg_tdoas = []
    segments = []
    for i, entry in enumerate(temp_diary_):
        if not distributed:
            if np.all(abs(np.median(entry[0], 0)) < .2):
                continue # skip noise position "above" the microphone
        if len(entry[1]) <= min_cl_segment:
            continue
        med_tdoa = np.median(entry[0], 0)
        act = pb.array.interval.zeros(recording_length)
        onset = np.maximum((np.min(entry[1]) - avg_len_gcc), 0)
        offset = np.max(entry[1])  # * shift + 4096
        act.add_intervals([slice(onset, offset), ])
        to_remove = []
        for o, other in enumerate(temp_diary_[i + 1:]):
            if np.linalg.norm(np.median(other[0], 0) - med_tdoa) <= max_diff_tmp_cl:
                other_act = pb.array.interval.zeros(recording_length)
                onset = np.maximum((np.min(other[1]) - avg_len_gcc), 0)
                offset = np.max(other[1])# * shift + 4096
                other_act.add_intervals([slice(onset, offset), ])
                if np.sum(np.array(act) * np.array(other_act)) > 0:
                    for t in other[0]:
                        entry[0].append(t)
                    for t in other[1]:
                        entry[1].append(t)
                    to_remove.append(i + 1 + o)
        for remove_id in to_remove[::-1]:
            temp_diary_.pop(remove_id)
        med_tdoa = np.median(entry[0], 0)
        act = pb.array.interval.zeros(recording_length)
        onset = np.maximum((np.min(entry[1]) - avg_len_gcc), 0)
        offset = np.max(entry[1]) #* shift + 4096
        act.add_intervals([slice(onset, offset), ])
        segments.append(act)
        seg_tdoas.append(med_tdoa)
    return segments, seg_tdoas
