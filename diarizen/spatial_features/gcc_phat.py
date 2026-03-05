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

def print_mem_usage(n=10):
    frames = inspect.stack()
    all_vars = []
    for frame_info in frames:
        frame = frame_info.frame
        for name, obj in frame.f_locals.items():
            try:
                # PyTorch Tensor
                if torch.is_tensor(obj):
                    size = obj.element_size() * obj.nelement()
                # NumPy Array
                elif isinstance(obj, np.ndarray):
                    size = obj.nbytes
                # Sonstiges Python-Objekt
                else:
                    size = sys.getsizeof(obj)
                all_vars.append((name, size, type(obj), frame_info.function))
            except Exception:
                pass

    # Größte N Variablen sortieren
    all_vars.sort(key=lambda x: -x[1])
    print(f"\nTop {n} Speicherfresser (über Callstack hinweg):")
    for name, size, typ, func in all_vars[:n]:
        print(f"{func:15s} | {name:20s} | {typ} | {size / 1024 ** 2:.2f} MB")


def get_gcc_for_all_channel_pairs_torch(sigs_stft, f_min=125, f_max=3500,sample_rate=16000, ups_fact=10, search_range=10,
                                        apply_ifft=True, framewise=False):
    if not framewise:
        fft_size = frame_size = (sigs_stft.shape[-1] - 1) * 2
        if f_min is None:
            k_min = None
        else:
            k_min = int(torch.round(torch.tensor(f_min / (sample_rate / 2) * (fft_size // 2 + 1))).item())
        if f_max is None:
            k_max = None
        else:
            k_max = int(torch.round(torch.tensor(f_max / (sample_rate / 2) * (fft_size // 2 + 1))).item())

        gcpsd = get_gcpsd_matrix(sigs_stft)
        # gcpsd = np.array([gcpsd[i, j, :, :] for (i,j) in get_ch_pairs(gcpsd.shape[0])])  # (num_ch_pairs, num_frames, fft_bins)
        gcpsd = torch.stack(
            [gcpsd[i, j, :, :] for (i, j) in get_ch_pairs(gcpsd.shape[0])])  # (num_ch_pairs, num_frames, fft_bins)

        if k_max is not None:
            gcpsd = gcpsd[:, :, :k_max]
        if k_min is not None:
            gcpsd = gcpsd[:, :, k_min:]

        output = gcpsd.permute(1, 0, 2)  # (num_frames, num_ch_pairs, fft_bins)

        if apply_ifft:
            c, t, f = gcpsd.shape
            gcpsd = torch.cat([
                    gcpsd[:, :, :-1],
                    torch.zeros((c, t, (ups_fact - 1) * (f - 1) * 2), dtype=gcpsd.dtype, device=gcpsd.device),
                    torch.conj(gcpsd).flip(dims=[-1])[:, :, :-1]
                ], dim=-1)

            """Centering and backtransformation for GCC-PHAT"""
            gcc_temp = torch.fft.ifft(gcpsd, dim=-1)
            gcc = torch.fft.fftshift(gcc_temp.real,
                                     dim=-1)  # (num_ch_pairs, num_frames, delays= 2 * ups_fact * search_range)
            search_area = \
                gcc[:, :, gcc.shape[-1] // 2 - search_range * ups_fact: gcc.shape[-1] // 2 + search_range * ups_fact]
            gccs = rearrange(search_area, 'c t d -> t c d')  # (num_frames, num_ch_pairs, delays)
            output = gccs
    else:
        raise NotImplementedError
    return output


def get_ch_pairs(num_chs):
    ch_pairs = []
    for i in range(num_chs):
        for j in range(i + 1, num_chs):
            ch_pairs.append((i, j))
    return ch_pairs

def get_gcpsd(fft_seg, fft_ref_seg):
    cpsd = np.conj(fft_ref_seg) * fft_seg
    phat = np.abs(fft_seg) * np.abs(fft_ref_seg)
    gcpsd = cpsd / np.maximum(phat, 1e-9)
    return gcpsd

def get_gcpsd_matrix(sigs_stft):
    cpsd = sigs_stft * sigs_stft[:, None].conj()
    phat = torch.abs(sigs_stft[None, :]) * torch.abs(sigs_stft[:, None])
    gcpsd = cpsd / torch.maximum(phat, torch.tensor(1e-9, dtype=phat.dtype, device=phat.device))
    return gcpsd

# def get_gcpsd_matrix(sigs_stft):
#     cpsd = sigs_stft * sigs_stft[:, None].conj()
#     phat = np.abs(sigs_stft[None,:]) * np.abs(sigs_stft[:, None])
#     gcpsd = cpsd / np.maximum(phat, 1e-9)
#     return gcpsd

def get_gcc_for_all_channel_pairs(sigs_stft, frame_wise_activities, dominant, f_min=125, f_max=3500, search_range=10,
                                  avg_len=4, sample_rate=16000, ups_fact=10, framewise=True, eval=False, audio=None,
                                  shift=None, modelbased=False, apply_ifft=True,):
    """
     sigs_stft : np.ndarray
         STFT of input signals, shape (num_chs, num_frames, fft_bins).
     frame_wise_activities : np.ndarray
         Activity mask per channel and frame, shape (num_chs, num_frames).
     f_min : int, optional
         Minimum frequency (Hz) for GCC-PHAT, by default 125.
     f_max : int, optional
         Maximum frequency (Hz) for GCC-PHAT, by default 3500.
     avg_len : int, optional
         Number of frames to average for GCC-PHAT, by default 4.
     sample_rate : int, optional
         Sampling rate of the signals, by default 16000.
     search_range: int, optional
        Range for possible delays, default 10. (compact 5-10, distributed ~200)
     """
    num_chs = len(sigs_stft)
    fft_size = frame_size = (sigs_stft.shape[-1] -1) * 2
    """freq-bins k for freq filtering"""
    if f_min is None:
        k_min = None
    else:
        k_min = int(np.round(f_min / (sample_rate / 2) * (fft_size // 2 + 1)))
    if f_max is None:
        k_max = None
    else:
        k_max = int(np.round(f_max / (sample_rate / 2) * (fft_size // 2 + 1)))
    ch_pairs = get_ch_pairs(num_chs)
    # ch_pairs are all combinations (upper triangle) of channels
    gcpsd_buffer = \
        np.zeros((len(ch_pairs), avg_len, frame_size // 2 + 1), np.complex128)
    gccs = []



#     # np.triu(X\*X\[:, None\].conj()) # statt for loops und kein buffer

    if not framewise:
        # print("MATRIX GCC")
        #                               (num_chs, num_frames, fft_bins).
        # np.triu(rearrange(sigs_stft * sigs_stft[:, None].conj(), 'c d t f -> t f c d'), k=1) # . Dann sind alle schleifen ersetzt worden.
        gcpsd = get_gcpsd_matrix(sigs_stft)


        gcpsd = np.array([gcpsd[i, j, :, :] for (i,j) in get_ch_pairs(gcpsd.shape[0])])  # (num_ch_pairs, num_frames, fft_bins)
        c, t, f = gcpsd.shape
        if dominant:
            gcpsd = gcpsd * dominant[None]  # Multiply dominant to filter out noise frequencies
        # gcpsd = np.triu(rearrange(gcpsd, 'c d t f -> t f c d'), k=1)
        #
        vad = np.sum(frame_wise_activities, axis=0)
        # gcpsd[vad==0] = 0  # Set gcpsd to 0 where vad is 0
        # # TODO vtl vad hinter buffer erst?
        #
        # gcpsd = np.array([gcpsd[:,:,i,j] for (i,j) in get_ch_pairs(gcpsd.shape[-1])])  # (num_ch_pairs, num_frames, fft_bins)
        # c, t, f = gcpsd.shape
        # gcpsd = gcpsd * dominant[None]  # Multiply dominant to filter out noise frequencies

        # # # # Buffer for smoothing, for now leave out
        # gcpsd_smooth = gcpsd.copy()
        # for i in range(gcpsd.shape[1]):
        #     if i < avg_len:
        #         gcpsd_smooth[:, i] = np.mean(gcpsd[:, :i + 1], axis=1)
        #     elif i < gcpsd.shape[1]:
        #         gcpsd_smooth[:, i] = np.mean(gcpsd[:, i - avg_len:i], axis=1)
        #     elif i == gcpsd.shape[1]:
        #         gcpsd_smooth[:, i] = np.mean(gcpsd[:, i - avg_len:], axis=1)
        # gcpsd = gcpsd_smooth


        gcpsd[gcpsd > 0.5 / avg_len] /= np.abs(gcpsd[gcpsd > 0.5 / avg_len])

        if k_min is not None:
            gcpsd[:, :, :k_min] = 0.
        if k_max is not None:
            gcpsd[:, :, k_max:] = 0.
        # und interpolation  auch

        gcpsd = np.concatenate(
            [gcpsd[:, :, :-1],
             np.zeros((c, t, (ups_fact - 1) * (f - 1) * 2)),
             np.conj(gcpsd)[:, :, ::-1][:, :, :-1]],
            axis = -1
        )

        """Centering and backtransformation for GCC-PHAT"""
        gcc_temp = np.fft.ifft(gcpsd, axis=-1)
        gcc = np.fft.ifftshift(gcc_temp.real, axes=-1)  # (num_ch_pairs, num_frames, delays= 2 * ups_fact * search_range)

        # print(gcc.shape)
        # return gcc[0, 0]
        # TODO: check if search range is big enough or if peaks outside are loking good?
        # search_range =10*search_range
        search_area = \
            gcc[:, :, gcc.shape[-1] // 2 - search_range * ups_fact: gcc.shape[-1] // 2 + search_range * ups_fact]
        gccs = rearrange(search_area, 'c t d -> t c d')  # (num_frames, num_ch_pairs, delays)

        gccs[vad==0, :, :] = 0      #  Set gcpsd to 0 for frames where vad is 0
    elif framewise:
        # TODO: GCC AUF 0 setzen, wenn negativ??
        # print("FOR LOoP GCC", frame_wise_activities.shape[-1])

        if modelbased:
            tdoa_candidates = get_position_candidates(sigs_stft, frame_wise_activities, dominant, f_max=f_max, search_range=search_range,
                                    avg_len=avg_len, num_peaks=5, sample_rate=sample_rate, max_diff=1, upsampling=ups_fact,
                                    max_concurrent=4, distributed=False, p_th=0.75)  # p_th = 0.75
            segments = temporally_constrained_clustering(tdoa_candidates, **{'max_dist': 0.75, 'peak_ratio_th':.5,  # 0.75, .5, 50
                                       'max_temp_dist':50})
            segments = segments[::-1]  # ouput of temporal clustering begins with the last segment in the meeting
            assert shift is not None
            num_frames = sigs_stft.shape[1]
            segments, segment_tdoas = merge_overlapping_segments(segments, num_frames, avg_len_gcc=avg_len, min_cl_segment=3,
                                                                 distributed=False, max_diff_tmp_cl=1, shift=shift)
            spks = np.zeros((num_frames))
            for seg in segments:
                for interval in seg.normalized_intervals:
                    spks[interval[0]:(interval[1]+1)] += 1
            return np.array(spks, int)

        else:
            for l in tqdm(range(frame_wise_activities.shape[-1]), desc="GCC-PHAT Progress"):
                gccs.append([])
                used = psutil.virtual_memory().used
                available = psutil.virtual_memory().available
                # if available < 70 * 1024 ** 3:
                #     print(f"Verwendet:     {used / (1024 ** 3):.2f} GB")
                #     print(f"Frei:          {available / (1024 ** 3):.2f} GB")

                for k, (i, j) in enumerate(ch_pairs):
                    gcpsd_buffer[k] = np.roll(gcpsd_buffer[k], -1, axis=0)
                    gcpsd_buffer[k, -1] = 0
                if np.sum(frame_wise_activities[:, l]) == 0:
                    print(f"Frame {l} inactive")
                    if apply_ifft:
                        gccs[l] = [np.zeros((2 * ups_fact * search_range ,)) for _ in range(len(ch_pairs))]
                    else:
                        gccs[l] = [np.zeros((sigs_stft.shape[-1],)) for _ in range(len(ch_pairs))]
                        # gccs[l] = [np.zeros((2 * ups_fact * search_range +1 ,)) for _ in range(len(ch_pairs))]  # +1 makes attention not divisble to 4 heads
                    continue
                if eval:
                    chunk = audio[:, l * shift : (l * shift + frame_size)]
                    sigs_stft = pb.transform.stft(chunk, frame_size, shift,
                                      pad=False, fading=False)
                    sigs_stft = sigs_stft[:, 0, :]  # only one frame

                    # ref = pb.transform.module_stft._get_window(signal.windows.blackman, symmetric_window=False, window_length=frame_size)
                    # chunk_win = chunk * ref
                    # sigs_stft_win = rfft(chunk_win, n=frame_size)
                for k, (ref_ch, ch) in enumerate(ch_pairs):
                    if eval:
                        fft_seg = sigs_stft[ch]
                        fft_ref_seg = sigs_stft[ref_ch]
                    else:
                        fft_seg = sigs_stft[ch, l]
                        fft_ref_seg = sigs_stft[ref_ch, l]
                    gcpsd = get_gcpsd(fft_seg, fft_ref_seg)
                    if dominant is not None:
                        gcpsd_buffer[k, -1] = gcpsd * dominant[l] # Multiply dominant to filter out noise frequencies
                    else:
                        gcpsd_buffer[k, -1] = gcpsd

                    use_buffer = True
                    if use_buffer:
                        avg_gcpsd = np.mean(gcpsd_buffer[k], 0)
                    elif not use_buffer:
                        avg_gcpsd = gcpsd_buffer[k, -1]

                    # # avg_gcpsd = np.mean(gcpsd_buffer[k], 0)
                    # avg_gcpsd[avg_gcpsd > 0.5 / avg_len] /= np.abs(avg_gcpsd[avg_gcpsd > 0.5 / avg_len])


                    if k_min is not None:
                        avg_gcpsd[:k_min] = 0.
                    if k_max is not None:
                        avg_gcpsd[k_max:] = 0.

                    # print(avg_gcpsd.shape, "frame", l)
                    if not apply_ifft:
                        gccs[l].append(avg_gcpsd)
                        continue

                    """Calculate complete spectrum GCPSD"""
                    avg_gcpsd = np.concatenate(
                        [avg_gcpsd[:-1],
                         np.zeros((ups_fact - 1) * (len(avg_gcpsd) - 1) * 2),
                         np.conj(avg_gcpsd)[::-1][:-1]]
                    )

                    """Centering and backtransformation for GCC-PHAT"""
                    gcc_temp = np.fft.ifft(avg_gcpsd)
                    gcc = np.fft.ifftshift(gcc_temp.real)
                    # todo: testen wenn ifftshift erst auf vollem array gemacht wird?
                    # print(gcc.shape)
                    # return gcc

                    search_area = \
                        gcc[len(gcc) // 2 - search_range * ups_fact:len(gcc) // 2 + search_range * ups_fact]
                    # search_area = \
                    #     gcc[len(gcc) // 2 - search_range * ups_fact:len(gcc) // 2 + search_range * ups_fact + 1]
                    # 3D => (Frame, Chpairs, Delay)
                    gccs[l].append(search_area)
    else:
        raise ValueError("framewise must be True or False")



    try:
        gccs = np.array(gccs)
    except Exception as e:
        print(f'Error converting gccs to array: {e}')
        import pdb
        pdb.set_trace()
        # print(f'GCC-PHAT: {len(gccs)} active frames, {len(ch_pairs)} channel pairs, {len(search_area)} delays')

    return gccs # shape: (num_frames, num_ch_pairs, len(search_area))

def compute_vad_th(sigs, frame_size=1024, frame_shift=256):
    ths = []
    for ch_id, sig in enumerate(sigs):
        try:
            energy = np.sum(
                pb.array.segment_axis(
                    sig[sig > 0], frame_size, frame_shift, end='cut'   #  tobi: sig[sig!=0]
                ) ** 2,
                axis=-1
            )
            th = np.min(energy[energy > 0])
        except:
            import pdb
            pdb.set_trace()
        ths.append(th)
    return ths

def channel_wise_activities(sigs, ths):
    # import pdb
    # pdb.set_trace()
    activities = np.zeros_like(sigs, bool)
    for ch_id, (sig, th) in enumerate(zip(sigs, ths)):
    #     if sig[sig > 0]:
    #         try:
    #             energy = np.sum(
    #                 pb.array.segment_axis(
    #                     sig[sig > 0], frame_size, frame_shift, end='cut'   #  tobi: sig[sig!=0]
    #                 ) ** 2,
    #                 axis=-1
    #             )
    #             th = np.min(energy[energy > 0])
    #         except:
    #             import pdb
    #             pdb.set_trace()
        # 8 sek windows => hinten mehrere minuten stille => ganzes "SIGNAL" ist 0 also signal ist nur 8 sek window
        # cd /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_mc/
        #         [1, 0, 0, 0]]], device='cuda:1', dtype=torch.uint8),
        #         'names': ['20200712_M_R002S05C01', '20200712_M_R002S05C01', '20200712_M_R002S05C01', '20200712_M_R002S05C01', '20200712_M_R002S05C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01', '20200704_M_R002S06C01'],
        #         'debug': [['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200712_M_R002S05C01.wav', 27], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200712_M_R002S05C01.wav', 27], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200712_M_R002S05C01.wav', 27], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200712_M_R002S05C01.wav', 27], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200712_M_R002S05C01.wav', 27], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28], ['/mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/wavs/dev/20200704_M_R002S06C01.wav', 28]]}

        vad = VoiceActivityDetector(7 * th, len_smooth_win=0) # oder 0?
        act = vad(sig)
        act = np.array(dilate(pb.array.interval.ArrayInterval(act), 3201))
        act = np.array(erode(pb.array.interval.ArrayInterval(act), 3201))
        activities[ch_id] = act[:len(sig)]
    return activities

def convert_to_frame_wise_activities(
        activities, th=.5, frame_size=1024, frame_shift=256
):
    frame_wise_activities = np.sum(
        pb.array.segment_axis(
            activities, length=frame_size, shift=frame_shift, end='cut'
        ), -1
    ) > th * frame_size
    return frame_wise_activities

def erode(activity, kernel_size):
    activity_eroded = pb.array.interval.zeros(shape=activity.shape)
    for (onset, offset) in activity.normalized_intervals:
        onset += (kernel_size - 1) // 2
        onset = np.maximum(onset, 0)
        offset -= (kernel_size - 1) // 2
        offset = np.minimum(offset, activity.shape)
        activity_eroded.add_intervals([slice(onset, offset)])
    return activity_eroded


def dilate(activity, kernel_size):
    activity_dilated = pb.array.interval.zeros(shape=activity.shape)
    for (onset, offset) in activity.normalized_intervals:
        onset -= (kernel_size - 1) // 2
        onset = np.maximum(onset, 0)
        offset += (kernel_size - 1) // 2
        offset = np.minimum(offset, activity.shape)
        activity_dilated.add_intervals([slice(onset, offset)])
    return activity_dilated


def get_dominant_time_frequency_mask(sigs_stft, kernel_size_scm_smoothing=3, eig_val_ratio_th=0.9):
    """
      Computes a dominant time-frequency mask for multichannel STFT signals.

      For each time frame, this function calculates spatial covariance matrices (SCMs) over a local window,
      smooths them, and determines the dominance of the principal eigenvalue compared to the second largest.
      A time-frequency bin is marked as dominant if the ratio of the second to the largest eigenvalue is below a threshold,
      and the largest eigenvalue exceeds a minimum threshold.
      Args:
          sigs_stft (np.ndarray): Multichannel STFT signals with shape (channels, time, frequency).
      Returns:
          np.ndarray: Boolean mask of shape (time, frequency) indicating dominant time-frequency bins.
      """
    dominant = np.zeros_like(sigs_stft[0], bool)
    eig_val_mem = np.zeros_like(sigs_stft[0])
    sigs_stft_ = np.pad(sigs_stft, ((0, 0), (1, 1), (0, 0)), mode='edge')
    for i in range(1, sigs_stft_.shape[1] - 1):
        scms = np.einsum('ctf, dtf -> fcd', sigs_stft_[:, i - 1:i + 2], sigs_stft_[:, i - 1:i + 2].conj())
        scms = fftconvolve(
            np.pad(
                scms,
                (
                    (kernel_size_scm_smoothing // 2, kernel_size_scm_smoothing // 2),
                    (0, 0),
                    (0, 0)),
                mode='edge'
            ),
            1 / kernel_size_scm_smoothing * np.ones(
                (kernel_size_scm_smoothing, len(sigs_stft), len(sigs_stft))
            ),
            axes=0,
            mode='valid'
        )
        eig_vals, _ = np.linalg.eigh(scms)
        dominance = 1 - eig_vals[..., -2] / (eig_vals[..., -1] + 1e-9)
        dominant[i - 1] = (dominance >= eig_val_ratio_th)
        eig_val_mem[i - 1] = eig_vals[..., -1]
    eig_val_th = 10 * np.min(eig_val_mem)
    dominant *= (eig_val_mem > eig_val_th)
    return dominant

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
