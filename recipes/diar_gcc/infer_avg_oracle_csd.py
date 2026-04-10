# Licensed under the MIT license.
# Adopted from https://github.com/espnet/espnet/blob/master/egs2/chime8_task1/diar_asr1/local/pyannote_diarize.py
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
import os
import argparse
import toml
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import torchaudio

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import linear_sum_assignment
from diarizen.utils import instantiate
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

# from pyannote.database.util import load_rttm
from pyannote.metrics.segmentation import Annotation, Segment
# from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
# from pyannote.audio.utils.signal import Binarize

from diarizen.ckpt_utils import load_metric_summary


def plot_cm(cm, num_classes, normalized=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    if normalized:
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f"Pred {i}" for i in range(num_classes)],
                    yticklabels=[f"True {i}" for i in range(num_classes)],
                    ax=ax)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[f"Pred {i}" for i in range(num_classes)],
                    yticklabels=[f"True {i}" for i in range(num_classes)],
                    ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Validation)")
    return fig

def plot_cms(cm, num_classes, out_dir):
    fig = plot_cm(cm, num_classes )
    fig_path = out_dir / f"confusion_matrix.png"
    fig.savefig(fig_path.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    # cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    # cm = np.nan_to_num(cm_normalized)  # for division by 0
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(
        cm.astype(np.float32),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float32),
        where=row_sums != 0,
    )

    fig = plot_cm(cm, num_classes, normalized=True)
    fig_path = out_dir / f"confusion_matrix_normalized.png"
    fig.savefig(fig_path.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return

def oracle_HA_per_window(ann, hyp_segmentations, shift=0.1):
    """ann - pyannote.annotation, hyp_segmentations - EEND output, shift of local EEND windows"""
    frame_res = 0.020  # seconds
    num_chunks, num_frames = hyp_segmentations.shape[:2]

    W = (num_frames + 1) * frame_res
    hop = shift*W
    total = W + (num_chunks - 1) * hop 
    tseg = Segment(0, total + W)  # add W, so REF is atleast of same size as HYP  

    gt = ann.discretize(support=tseg, resolution=frame_res).crop(focus=tseg).astype(float)
    
    SF = np.round((num_frames + 1)*shift).astype(int)  # shift in frames
    gt_segmentations = np.array([gt[i:i+num_frames] 
                                 for i in np.arange(num_chunks) * SF])

    FA_label = gt_segmentations.shape[-1] # extra garbage speaker label in window
    hyp_labels = -2 + np.zeros(hyp_segmentations.shape[0::2])
    hyp_labels[hyp_segmentations.sum(1) > 0] = FA_label

    _cidx, _sidx = np.where(gt_segmentations.sum(1))

    windows = (-gt_segmentations.swapaxes(1, 2) @ hyp_segmentations) # hyp_segmentations)
    for i, cost_matrix in enumerate(windows):
        present_speakers = _sidx[_cidx == i]
    
        # HA for single window:
        for _gt_idx, _ref_idx in zip(*linear_sum_assignment(cost_matrix)):
            if _gt_idx in present_speakers:
                hyp_labels[i, _ref_idx] = _gt_idx

    return hyp_labels.astype(int)

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

def rttm2label(rttm_file, rec_scp):
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
                    get_session_idx(session,rec_scp),
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

def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def get_session_idx(session,rec_scp):
    """
    convert session to session idex
    """
    session_keys = list(rec_scp.keys())
    return session_keys.index(session)

def get_chunk_labels(session, chunk_start, chunk_end, rec_scp, rttm_file, model_num_frames, model_rf_duration, model_rf_step):
    # session, chunk_start, chunk_end = example['rec'], example['start'], example['end'] # , example['global_start']
    # chunk_start = global_start + chunk_start
    # chunk_end = global_start + chunk_end
    # chunked annotations
    session_idx = get_session_idx(session,rec_scp)
    annotations = rttm2label(rttm_file, rec_scp)
    annotations_session = annotations[annotations['session_idx'] == session_idx]

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

    return mask_label

def compute_metrics(all_targets, all_preds, numpy=False):
    if not numpy:
        all_preds = torch.cat(all_preds).cpu().numpy().reshape(-1)
        all_targets = torch.cat(all_targets).cpu().numpy().reshape(-1)
    num_classes = max(all_targets.max(), all_preds.max()) + 1
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    # F1 Score
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)

    precision_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)

    recall_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    return cm, f1_macro, f1_weighted, f1_per_class, precision_macro, precision_per_class, recall_macro, recall_per_class, num_classes, all_preds, all_targets

# def compute_correct_predictions_numpy(target_spk_count, y_pred):
#
#     num_correct = 0
#     num_total = 0
#     num_correct_ov = 0
#     num_total_ov = 0
#
#     pred_labels = np.argmax(y_pred, axis=-1)
#
#     gt_labels = target_spk_count
#     # all_preds.append(pred_labels)
#     # all_targets.append(gt_labels)
#
#     hits = int(np.sum(pred_labels == gt_labels))
#     total = int(pred_labels.size)
#     num_correct += hits
#     num_total += total
#
#     ov_mask = gt_labels >= 2
#     hits_ov = int(np.sum((pred_labels == gt_labels) & ov_mask))
#     total_ov = int(np.sum(ov_mask))
#     total_active = int(np.sum(gt_labels >= 1))
#     num_correct_ov += hits_ov
#     num_total_ov += total_ov
#
#     return num_correct, num_total, num_correct_ov, num_total_ov, total_active

def compute_correct_predictions_numpy(target_spk_count, y_pred):
    # target_spk_count = np.asarray(target_spk_count)
    # y_pred = np.asarray(y_pred)
    gt_labels = np.squeeze(target_spk_count).astype(np.int64)

    pred_labels = y_pred.astype(np.int64)

    if pred_labels.shape != gt_labels.shape:
        raise ValueError(
            f"Shape mismatch: pred_labels{pred_labels.shape} vs gt_labels{gt_labels.shape}"
        )

    hits = int(np.sum(pred_labels == gt_labels))
    total = int(gt_labels.size)

    ov_mask = gt_labels >= 2
    hits_ov = int(np.sum((pred_labels == gt_labels) & ov_mask))
    total_ov = int(np.sum(ov_mask))
    total_active = int(np.sum(gt_labels >= 1))

    return hits, total, hits_ov, total_ov, total_active

def compute_correct_predictions(target_spk_count, y_pred):
    all_preds = []
    all_targets = []
    num_correct = 0
    num_total = 0
    num_correct_ov = 0
    num_total_ov = 0

    pred_labels = torch.argmax(y_pred, dim=-1)

    gt_labels = target_spk_count.squeeze()  # (F) num spk per frame
    all_preds.append(pred_labels.cpu())
    all_targets.append(gt_labels.cpu())

    hits = (pred_labels == gt_labels).sum().item()
    total = pred_labels.numel()
    num_correct += hits
    num_total += total

    hits_ov = ((pred_labels == gt_labels) & (gt_labels >= 2)).sum().item()
    total_ov = (gt_labels >= 2).sum().item()
    total_active = (gt_labels >= 1).sum().item()
    num_correct_ov += hits_ov
    num_total_ov += total_ov

    return all_preds, all_targets, num_correct, num_total, num_correct_ov, num_total_ov, total_active

def diarize_session(
    sess_name,
    rttm_file,
    in_wav,
    # pipeline,
    # min_speakers=1,
    # max_speakers=20,
    # apply_median_filtering=True,
    out_dir=None,
    model=None,
    scp=None,
):
    all_preds = []
    all_targets = []
    num_correct = 0
    num_total = 0
    num_correct_ov = 0
    num_total_ov = 0
    total_active = 0

    print('Extracting segmentations...')
    if not os.path.exists(in_wav):
        # in_wav = in_wav.replace("/mnt/matylda3/ihan/project/diarization/", "prefix_new/")
        # in_wav = re.sub(r"^/mnt/[^/]+/*.wav", "/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7/wavs/test/", in_wav)
        in_wav = os.path.join("/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7/wavs/test/", os.path.basename(in_wav))
        # '/mnt/matylda3/ihan/project/diarization/dataset/NOTSOFAR1/multi-channel/wavs/test/S32000107.wav'
    waveform, sample_rate = torchaudio.load(in_wav)
    waveform = waveform.to(device)

    chunk_duration = 8  # seconds
    chunk_size = chunk_duration * sample_rate
    num_samples = waveform.shape[1]
    rec_scp = load_scp(scp)

    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info

    for start_sample in range(0, num_samples, chunk_size):
        end_sample = min(start_sample + chunk_size, num_samples)

        chunk = waveform[0, start_sample:end_sample]  # (channels, samples)
        # Pad last chunk if shorter than chunk_size
        if chunk.shape[0] < chunk_size:
            pad_size = chunk_size - chunk.shape[0]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))

        # Chunk time in seconds (RTTM is in seconds)
        chunk_start_sec = start_sample / float(sample_rate)
        chunk_end_sec = end_sample / float(sample_rate)
        labels = get_chunk_labels(sess_name, chunk_start_sec, chunk_end_sec, rec_scp, rttm_file, model_num_frames, model_rf_duration,
                         model_rf_step)

        target_spk_count = labels.sum(axis=-1)  # (F,)

        max_num_spk = getattr(model, 'module', model).max_num_spk
        target_spk_count = np.clip(target_spk_count, 0, max_num_spk - 1)
        # target_spk_count = torch.clamp(target_spk_count, min=0, max=max_num_spk - 1)

        with torch.no_grad():
            prediction = model(chunk[None])
        # TODO: model frame auflösung zu samples bzw sekunden damit auf ground truth mappen kann? oder truth in frames holen?
        # TODO: pro chunk eval oder am ende einmal? Aber F1Score, recall etc alles speichern
        preds, targets, num_correct_tmp, num_total_tmp, num_correct_ov_tmp, num_total_ov_tmp, total_active_tmp = compute_correct_predictions(torch.as_tensor(target_spk_count.astype(np.int64), device=prediction.device), prediction)
        all_preds.extend(preds)
        all_targets.extend(targets)
        num_correct += num_correct_tmp
        num_total += num_total_tmp
        num_correct_ov += num_correct_ov_tmp
        num_total_ov += num_total_ov_tmp
        total_active += total_active_tmp


    accuracy = num_correct / num_total if num_total > 0 else 0.0
    accuracy_ov = num_correct_ov / num_total_ov if num_total_ov > 0 else 0.0

    cm, f1_macro, f1_weighted, f1_per_class, precision_macro, precision_per_class, recall_macro, recall_per_class, num_classes, all_preds, all_targets = compute_metrics(all_targets, all_preds)
    plot_cms(cm, num_classes, out_dir)
    # save predictions, targets and metrics
    np.savez_compressed(
        (out_dir / "preds_targets.npz").as_posix(),
        preds=all_preds,
        targets=all_targets,
    )
    summary_path = out_dir / "metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy: {accuracy:.6f}\n")
        f.write(f"accuracy_ov: {accuracy_ov:.6f}\n")
        f.write(f"Relative OV-Time: {num_total_ov / num_total :.6f}\n")
        f.write(f"Relative active Time: {total_active / num_total:.6f}\n")
        f.write(f"f1_macro: {f1_macro:.6f}\n")
        f.write(f"f1_weighted: {f1_weighted:.6f}\n")
        f.write(f"precision_macro: {precision_macro:.6f}\n")
        f.write(f"recall_macro: {recall_macro:.6f}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write("\nclass-wise metrics:\n")
        for c in range(num_classes):
            f1_c = float(f1_per_class[c]) if c < len(f1_per_class) else float("nan")
            p_c = float(precision_per_class[c]) if c < len(precision_per_class) else float("nan")
            r_c = float(recall_per_class[c]) if c < len(recall_per_class) else float("nan")
            f.write(f"class {c}: f1={f1_c:.6f}, precision={p_c:.6f}, recall={r_c:.6f}\n")
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required arguments
    parser.add_argument(
        "-C",
        "--configuration",
        type=str,
        required=True,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-i", 
        "--in_wav_scp",
        type=str,
        required=True,
        help="test wav.scp.",
        dest="in_wav_scp",
    )
    parser.add_argument(
        "-o", 
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model.",
    )

    parser.add_argument(
        "--rttm_file",
        type=str,
        help="rttm_file",
    )

    # Optional arguments
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--avg_ckpt_num",
        type=int,
        default=5,
        help="the number of chckpoints of model averaging",
    )
    parser.add_argument(
        "--val_metric",
        type=str,
        default="Loss",
        help="validation metric",
        choices=["Loss", "DER", "F1score", "DE0_Rov"],
    )
    parser.add_argument(
        "--val_mode",
        type=str,
        default="best",
        help="validation metric mode",
        choices=["best", "prev", "center"],
    )
    parser.add_argument(
        "--val_metric_summary",
        type=str,
        default="",
        help="val_metric_summary",
    )
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="",
        help="Path to pretrained segmentation model.",
    )

    # Inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # Clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )

    args = parser.parse_args()
    # print(args)

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())
    
    ckpt_path = config_path.parent / 'checkpoints'
    segmentation = args.segmentation_model
    if args.val_metric_summary:
        val_metric_lst = load_metric_summary(args.val_metric_summary, ckpt_path)
        metric = args.val_metric
        if metric == "F1score":
            val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[metric], reverse=True)
        else:
            val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[metric])
        best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
        if args.val_mode == "best":
            segmentation = val_metric_lst_sorted[:args.avg_ckpt_num]
        elif args.val_mode == "prev":
            segmentation = val_metric_lst[
                best_val_metric_idx - args.avg_ckpt_num + 1 :
                best_val_metric_idx + 1
            ]
        else:
            segmentation = val_metric_lst[
                best_val_metric_idx - args.avg_ckpt_num // 2 :
                best_val_metric_idx + args.avg_ckpt_num // 2 + 1
            ]
        assert len(segmentation) == args.avg_ckpt_num

    best_ckp = val_metric_lst_sorted[0]["bin_path"]

    # create, instantiate and apply the pipeline
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    audio_dict = load_scp(args.in_wav_scp)

    if best_ckp is None:
        raise ValueError("Kein Checkpoint-Pfad in best_ckp gefunden.")

    state_dict = torch.load(best_ckp, map_location="cpu")

    model = instantiate(config['model']["path"], args=config['model']["args"])

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    for sess, in_wav in audio_dict.items():
        # if sess in ["S32000107", "S32000207", "S32003107", ]:
        #     continue
        (Path(args.out_dir) / f"{sess}" ).mkdir(exist_ok=True, parents=True)
        print(f"Diarizing Session: {sess}", flush=True)
        diar_result = diarize_session(
            sess_name=sess,
            rttm_file=args.rttm_file,
            in_wav=in_wav,
            out_dir=Path(args.out_dir) / f"{sess}",
            model = model,
            scp=args.in_wav_scp,
        )

    preds = []
    targets = []
    out_dir = Path(args.out_dir)
    for d in os.listdir(out_dir):
        npz_path = Path(out_dir) / d / "preds_targets.npz"
        if not npz_path.is_file():
            continue

        data = np.load(npz_path)
        pred = np.asarray(data["preds"]).reshape(-1)
        target = np.asarray(data["targets"]).reshape(-1)

        preds.append(pred)
        targets.append(target)

    preds = np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)
    targets = np.concatenate(targets, axis=0) if targets else np.array([], dtype=np.int64)

    num_correct, num_total, num_correct_ov, num_total_ov, total_active = compute_correct_predictions_numpy(targets, preds)
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    accuracy_ov = num_correct_ov / num_total_ov if num_total_ov > 0 else 0.0



    cm, f1_macro, f1_weighted, f1_per_class, precision_macro, precision_per_class, recall_macro, recall_per_class, num_classes, all_preds, all_targets = compute_metrics(targets, preds, numpy=True)
    plot_cms(cm, num_classes, out_dir)
    # save predictions, targets and metrics
    np.savez_compressed(
        (out_dir / "preds_targets.npz").as_posix(),
        preds=all_preds,
        targets=all_targets,
    )
    summary_path = out_dir / "metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy: {accuracy:.6f}\n")
        f.write(f"accuracy_ov: {accuracy_ov:.6f}\n")
        f.write(f"Total OV-Time: {num_total_ov / num_total:.6f}\n")
        f.write(f"Total active Time: {total_active / num_total:.6f}\n")
        f.write(f"f1_macro: {f1_macro:.6f}\n")
        f.write(f"f1_weighted: {f1_weighted:.6f}\n")
        f.write(f"precision_macro: {precision_macro:.6f}\n")
        f.write(f"recall_macro: {recall_macro:.6f}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write("\nclass-wise metrics:\n")
        for c in range(num_classes):
            f1_c = float(f1_per_class[c]) if c < len(f1_per_class) else float("nan")
            p_c = float(precision_per_class[c]) if c < len(precision_per_class) else float("nan")
            r_c = float(recall_per_class[c]) if c < len(recall_per_class) else float("nan")
            f.write(f"class {c}: f1={f1_c:.6f}, precision={p_c:.6f}, recall={r_c:.6f}\n")

