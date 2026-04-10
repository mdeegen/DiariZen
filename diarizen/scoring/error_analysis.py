import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
import matplotlib.pyplot as plt
import paderbox as pb


def rttm_to_annotation(rttm_path):
    """
    Lädt eine RTTM-Datei und gibt ein dict[file_id] -> Annotation zurück.
    """
    annotations = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            _, file_id, _, start, dur, _, _, speaker, *_ = parts
            # if file_id != "IS1009a":
            #     continue
            start, dur = float(start), float(dur)
            end = start + dur
            if file_id not in annotations:
                annotations[file_id] = Annotation(uri=file_id)
            annotations[file_id][Segment(start, end)] = speaker
            # print(file_id)
    return annotations

def annotation_to_frame_matrix(ann, frame_hop=0.01, duration=None):
    """
    Konvertiert alle Annotationen in Frame-Matrizen.
    Gibt dict[file_id] -> (matrix, speaker_list) zurück.
    """
    speakers = list(sorted(ann.labels()))
    if duration is None:
        duration = max(seg.end for seg in ann.itersegments())

    num_frames = int(np.ceil(duration / frame_hop))
    act = np.zeros((num_frames, len(speakers)), dtype=int)

    for i, spk in enumerate(speakers):
        for segment in ann.label_timeline(spk).support():
            start_f = max(0, int(np.floor(segment.start / frame_hop)))
            end_f = min(num_frames, int(np.ceil(segment.end / frame_hop)))
            act[start_f:end_f, i] = 1

    return act, speakers

def frame_error_mask(ref_mat, hyp_mat, ov=False, single=False, silence=False):
    n = min(ref_mat.shape[0], hyp_mat.shape[0])
    ref = ref_mat[:n].astype(bool)
    hyp = hyp_mat[:n].astype(bool)

    miss_per_speaker = ref & ~hyp
    fa_per_speaker = ~ref & hyp
    correct_per_speaker = ref == hyp

    both_active = ref.any(axis=1) & hyp.any(axis=1)
    confusion = both_active & np.any(ref != hyp, axis=1)

    miss_any = miss_per_speaker.any(axis=1)
    fa_any = fa_per_speaker.any(axis=1)

    global_error = np.full(n, "correct", dtype=object)
    global_error[miss_any & ~fa_any] = "miss"
    global_error[~miss_any & fa_any] = "fa"
    global_error[~miss_any & ~fa_any & confusion] = "confusion"
    global_error[(miss_any.astype(int) + fa_any.astype(int) + confusion.astype(int)) > 1] = "mixed"
    if ov:
        global_error[ref_mat.sum(axis=1) <= 1] = "correct"  # dont count errors in non overlap region
    elif single:
        global_error[ref_mat.sum(axis=1) != 1] = "correct"  # dont count errors in overlap region
    elif silence:
        global_error[ref_mat.sum(axis=1) >= 1] = "correct"  # dont count errors in silence region
    return {
        "global": global_error,
        "miss_per_speaker": miss_per_speaker,
        "fa_per_speaker": fa_per_speaker,
        "correct_per_speaker": correct_per_speaker,
    }

def _binary_dilate_1d(x, k):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.int32)
    return np.convolve(x.astype(np.int32), kernel, mode="same") > 0


def _binary_erode_1d(x, k):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.int32)
    return np.convolve(x.astype(np.int32), kernel, mode="same") == k


def smooth_error_mask(error_mask, close_k=9, open_k=5):
    # Closing: kleine Lücken schließen
    y = _binary_dilate_1d(error_mask, close_k)
    y = _binary_erode_1d(y, close_k)

    # Opening: kurze Peaks entfernen
    y = _binary_erode_1d(y, open_k)
    y = _binary_dilate_1d(y, open_k)
    return y


def extract_error_clusters(error_mask, frame_hop=0.01, close_k=9, open_k=5, min_len=20):
    smoothed = smooth_error_mask(error_mask, close_k=close_k, open_k=open_k)

    edges = np.diff(np.r_[False, smoothed, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    clusters = []
    for s, e in zip(starts, ends):
        length = e - s
        if length < min_len:
            continue
        clusters.append(
            {
                "start_frame": int(s),
                "end_frame": int(e - 1),
                "length_frames": int(length),
                "start_s": float(s * frame_hop),
                "end_s": float(e * frame_hop),
                "duration_s": float(length * frame_hop),
            }
        )

    return smoothed, clusters


def _align_hyp_to_ref(ref_ann, hyp_ann, ref_mat, ref_spk, frame_hop=0.01, metric=None):
    """Aligniert ein Hypothesen-Annotation-Objekt auf die Referenz-Sprecherachsen."""
    if metric is None:
        metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    hyp_mat, hyp_spk = annotation_to_frame_matrix(hyp_ann, frame_hop=frame_hop)
    mapping = metric.optimal_mapping(ref_ann, hyp_ann)

    n = min(ref_mat.shape[0], hyp_mat.shape[0])
    aligned = np.zeros((n, ref_mat.shape[1]), dtype=ref_mat.dtype)

    for hyp_label, ref_label in mapping.items():
        if hyp_label not in hyp_spk or ref_label not in ref_spk:
            continue
        h_idx = hyp_spk.index(hyp_label)
        r_idx = ref_spk.index(ref_label)
        aligned[:, r_idx] = hyp_mat[:n, h_idx]

    return aligned, n


def _summarize_mask(mask, frame_hop, close_k=9, open_k=5, min_len=20):
    """Erzeugt kompakte Kennzahlen und Cluster fuer eine boolsche Fehlermaske."""
    smoothed, clusters = extract_error_clusters(
        mask,
        frame_hop=frame_hop,
        close_k=close_k,
        open_k=open_k,
        min_len=min_len,
    )

    durations = [c["duration_s"] for c in clusters]
    return {
        "frames": int(mask.sum()),
        "ratio": float(mask.mean()) if len(mask) else 0.0,
        "smoothed_frames": int(smoothed.sum()),
        "num_clusters": len(clusters),
        "mean_cluster_s": float(np.mean(durations)) if durations else 0.0,
        "max_cluster_s": float(np.max(durations)) if durations else 0.0,
        "clusters": clusters,
    }


def analyze_spatial_spectral_error_differences(
    ref_ann,
    hyp_spatial_ann,
    hyp_spectral_ann,
    frame_hop=0.01,
    close_k=9,
    open_k=5,
    min_len=20,
    ov=False,
    single=False,
    silence=False,
    metric=None,
):
    """
    Zeitliche Analyse fuer Unterschiede zwischen Spatial- und Spectral-Fehlern.

    Rueckgabe enthaelt:
      - Spatial-only / Spectral-only / Both-error Masken als Kennzahlen und Cluster
      - Unterschiedliche Fehlertypen in Bereichen, in denen beide Systeme falsch sind
      - Paarzaehlungen der Fehlertypen (z.B. spatial=miss vs spectral=fa)
    """
    if metric is None:
        metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)

    ref_mat, ref_spk = annotation_to_frame_matrix(ref_ann, frame_hop=frame_hop)
    aligned_spatial, n_spatial = _align_hyp_to_ref(
        ref_ann,
        hyp_spatial_ann,
        ref_mat,
        ref_spk,
        frame_hop=frame_hop,
        metric=metric,
    )
    aligned_spectral, n_spectral = _align_hyp_to_ref(
        ref_ann,
        hyp_spectral_ann,
        ref_mat,
        ref_spk,
        frame_hop=frame_hop,
        metric=metric,
    )

    n = min(ref_mat.shape[0], n_spatial, n_spectral)
    ref_cut = ref_mat[:n]
    aligned_spatial = aligned_spatial[:n]
    aligned_spectral = aligned_spectral[:n]

    err_spatial = frame_error_mask(ref_cut, aligned_spatial, ov=ov, single=single, silence=silence)
    err_spectral = frame_error_mask(ref_cut, aligned_spectral, ov=ov, single=single, silence=silence)

    glob_spatial = err_spatial["global"]
    glob_spectral = err_spectral["global"]

    spatial_error = glob_spatial != "correct"
    spectral_error = glob_spectral != "correct"

    spatial_only = spatial_error & ~spectral_error
    spectral_only = spectral_error & ~spatial_error
    both_error = spatial_error & spectral_error
    both_diff_type = both_error & (glob_spatial != glob_spectral)

    error_types = ["miss", "fa", "confusion", "mixed"]
    spatial_only_type_counts = {
        t: int(np.sum((glob_spatial == t) & spatial_only)) for t in error_types
    }
    spectral_only_type_counts = {
        t: int(np.sum((glob_spectral == t) & spectral_only)) for t in error_types
    }

    pair_counts = {}
    for t_spatial in error_types:
        for t_spectral in error_types:
            key = f"{t_spatial}|{t_spectral}"
            pair_counts[key] = int(
                np.sum((glob_spatial == t_spatial) & (glob_spectral == t_spectral) & both_error)
            )

    return {
        "n_frames": int(n),
        "frame_hop": float(frame_hop),
        "spatial_only": _summarize_mask(
            spatial_only,
            frame_hop,
            close_k=close_k,
            open_k=open_k,
            min_len=min_len,
        ),
        "spectral_only": _summarize_mask(
            spectral_only,
            frame_hop,
            close_k=close_k,
            open_k=open_k,
            min_len=min_len,
        ),
        "both_error": _summarize_mask(
            both_error,
            frame_hop,
            close_k=close_k,
            open_k=open_k,
            min_len=min_len,
        ),
        "both_diff_type": _summarize_mask(
            both_diff_type,
            frame_hop,
            close_k=close_k,
            open_k=open_k,
            min_len=min_len,
        ),
        "spatial_only_type_counts": spatial_only_type_counts,
        "spectral_only_type_counts": spectral_only_type_counts,
        "both_error_type_pairs": pair_counts,
    }


def aggregate_spatial_spectral_diff(results):
    """Aggregiert die pro-Datei-Differenzanalyse fuer einen Datensatz."""
    file_results = [
        r["spatial_vs_spectral_diff"]
        for k, r in results.items()
        if k != "total" and "spatial_vs_spectral_diff" in r
    ]

    if not file_results:
        return {}

    out = {
        "n_files": len(file_results),
        "n_frames": int(sum(r["n_frames"] for r in file_results)),
        "spatial_only_frames": int(sum(r["spatial_only"]["frames"] for r in file_results)),
        "spectral_only_frames": int(sum(r["spectral_only"]["frames"] for r in file_results)),
        "both_error_frames": int(sum(r["both_error"]["frames"] for r in file_results)),
        "both_diff_type_frames": int(sum(r["both_diff_type"]["frames"] for r in file_results)),
        "spatial_only_clusters": int(sum(r["spatial_only"]["num_clusters"] for r in file_results)),
        "spectral_only_clusters": int(sum(r["spectral_only"]["num_clusters"] for r in file_results)),
        "both_error_clusters": int(sum(r["both_error"]["num_clusters"] for r in file_results)),
        "both_diff_type_clusters": int(sum(r["both_diff_type"]["num_clusters"] for r in file_results)),
        "spatial_only_type_counts": {k: 0 for k in ["miss", "fa", "confusion", "mixed"]},
        "spectral_only_type_counts": {k: 0 for k in ["miss", "fa", "confusion", "mixed"]},
        "both_error_type_pairs": {
            f"{a}|{b}": 0
            for a in ["miss", "fa", "confusion", "mixed"]
            for b in ["miss", "fa", "confusion", "mixed"]
        },
    }

    for r in file_results:
        for k, v in r["spatial_only_type_counts"].items():
            out["spatial_only_type_counts"][k] += int(v)
        for k, v in r["spectral_only_type_counts"].items():
            out["spectral_only_type_counts"][k] += int(v)
        for k, v in r["both_error_type_pairs"].items():
            out["both_error_type_pairs"][k] += int(v)

    n = max(1, out["n_frames"])
    out["spatial_only_ratio"] = round(out["spatial_only_frames"] / n, 6)
    out["spectral_only_ratio"] = round(out["spectral_only_frames"] / n, 6)
    out["both_error_ratio"] = round(out["both_error_frames"] / n, 6)
    out["both_diff_type_ratio"] = round(out["both_diff_type_frames"] / n, 6)
    out["spatial_advantage_ratio"] = round(
        (out["spectral_only_frames"] - out["spatial_only_frames"]) / n,
        6,
    )
    return out


der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)


frame_hop = 0.01
ov = True
single = False
silence = False
for dset in ["AMI", "NOTSOFAR1", "AliMeeting", "AISHELL4"]:
    spectral_dir = f"/home/deegen/n3/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/spk_count_ref/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc_debug/{dset}"
    # TODO: N2 version testen? die ist minimal schelchter gewesen auf test?
    spatial_dir = f"/home/deegen/n3/merlin/recipes/diar_gcc/exp/gcpsd_encoder/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/{dset}"
    combined_dir = f"/home/deegen/n3/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp_old/spk_count_linear_noisy_to_gcpsd_encoder_ffn_film_all_layers_finetune/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc_orig_debug/{dset}"

    # experiments = {
    #     "spectral" : spectral_dir,
    #     "spatial" : spatial_dir,
    #     "combined" : combined_dir,
    # }
    #
    # for exp, exp_dir in experiments.items():
    #     if exp == "spectral":
    ref_anns = rttm_to_annotation(f"{spectral_dir}/referenz.rttm")
    hyp_spectral_anns = rttm_to_annotation(f"{spectral_dir}/all_hyp.rttm")
    hyp_spatial_anns = rttm_to_annotation(f"{spatial_dir}/all_hyp.rttm")
    hyp_comb_anns = rttm_to_annotation(f"{combined_dir}/all_hyp.rttm")

    results = {}

    for file_id in ref_anns.keys():

        if file_id not in hyp_spectral_anns or file_id not in hyp_spatial_anns or file_id not in hyp_comb_anns:
            continue

        ref_ann = ref_anns[file_id]
        hyp_spectral = hyp_spectral_anns[file_id]
        hyp_spatial = hyp_spatial_anns[file_id]
        hyp_comb = hyp_comb_anns[file_id]

        # DER
        der_spec = der_metric(ref_ann, hyp_spectral)
        der_spat = der_metric(ref_ann, hyp_spatial)
        der_comb = der_metric(ref_ann, hyp_comb)
        print(f"DER SPECTRAL: {der_spec:.2%}", flush=True)
        print(f"DER Spatial: {der_spat:.2%}", flush=True)
        print(f"DER Combined: {der_comb:.2%}", flush=True)

        # Frame masks
        ref_mat, ref_spk = annotation_to_frame_matrix(ref_ann, frame_hop=frame_hop)
        hyp_mat_spectral, hyp_spk_spectral = annotation_to_frame_matrix(hyp_spectral, frame_hop=frame_hop)
        hyp_mat_spatial, hyp_spk_spatial = annotation_to_frame_matrix(hyp_spatial, frame_hop=frame_hop)
        hyp_mat_comb, hyp_spk_comb = annotation_to_frame_matrix(hyp_comb, frame_hop=frame_hop)

        # Permutation Mapping
        mapping_spectral = der_metric.optimal_mapping(ref_ann, hyp_spectral)
        mapping_spatial = der_metric.optimal_mapping(ref_ann, hyp_spatial)
        mapping_comb = der_metric.optimal_mapping(ref_ann, hyp_comb)

        # Align
        aligned_hyp_spectral = np.zeros_like(ref_mat)
        aligned_hyp_spatial = np.zeros_like(ref_mat)
        aligned_hyp_comb = np.zeros_like(ref_mat)
        num_frames = min(ref_mat.shape[0], hyp_mat_spectral.shape[0], hyp_mat_spatial.shape[0], hyp_mat_comb.shape[0])
        # Spectral
        for hyp_label, ref_label in mapping_spectral.items():
            h_idx = hyp_spk_spectral.index(hyp_label)
            r_idx = ref_spk.index(ref_label)
            aligned_hyp_spectral[:num_frames, r_idx] = hyp_mat_spectral[:num_frames, h_idx]
        # Spatial
        for hyp_label, ref_label in mapping_spatial.items():
            h_idx = hyp_spk_spatial.index(hyp_label)
            r_idx = ref_spk.index(ref_label)
            aligned_hyp_spatial[:num_frames, r_idx] = hyp_mat_spatial[:num_frames, h_idx]
        # Combined
        for hyp_label, ref_label in mapping_comb.items():
            h_idx = hyp_spk_comb.index(hyp_label)
            r_idx = ref_spk.index(ref_label)
            aligned_hyp_comb[:num_frames, r_idx] = hyp_mat_comb[:num_frames, h_idx]

        # Error analysis
        err_spec = frame_error_mask(ref_mat, aligned_hyp_spectral, ov=ov, single=single, silence=silence)
        err_spat = frame_error_mask(ref_mat, aligned_hyp_spatial, ov=ov, single=single, silence=silence)
        err_comb = frame_error_mask(ref_mat, aligned_hyp_comb, ov=ov, single=single, silence=silence)


        glob_spec = err_spec["global"].copy()
        glob_spat = err_spat["global"].copy()
        glob_comb = err_comb["global"].copy()

        # numerisch codieren
        glob_spec = np.where(glob_spec == "correct", 0, 1)
        glob_spat = np.where(glob_spat == "correct", 0, 2)
        glob_comb = np.where(glob_comb == "correct", 0, 4)

        combined = glob_spec + glob_spat + glob_comb

        results[file_id] = {
            "all_error": np.sum(combined == 7), # / np.sum(combined != 0),
            "spat_spec_error": np.sum(combined == 3), # / np.sum(combined != 0),
            "spat_comb_error": np.sum(combined == 6), # / np.sum(combined != 0),
            "spec_comb_error": np.sum(combined == 5), # / np.sum(combined != 0),
            "spec_only": np.sum(combined == 1), # / np.sum(combined != 0),
            "spat_only": np.sum(combined == 2), # / np.sum(combined != 0),
            "comb_only": np.sum(combined == 4), # / np.sum(combined != 0),
            "error_frames": np.sum(combined != 0),
            "total_frames": len(combined),
            "DER_spec": der_spec,
            "DER_spat": der_spat,
            "DER_comb": der_comb,
        }

        # results[file_id]["spatial_vs_spectral_diff"] = analyze_spatial_spectral_error_differences(
        #     ref_ann,
        #     hyp_spatial,
        #     hyp_spectral,
        #     frame_hop=frame_hop,
        #     close_k=9,
        #     open_k=5,
        #     min_len=20,
        #     ov=ov,
        #     single=single,
        #     silence=silence,
        #     metric=der_metric,
        # )

        # print(results[file_id], flush=True)
        #
        # ### Select which systems errors to look for
        # # error_mask = (combined == 1) | (combined == 5)
        # error_mask = (combined == 2) | (combined == 6)
        # # error_mask = (combined == 2) | (combined == 6)
        # smoothed_error_mask, clusters = extract_error_clusters(
        #     error_mask,
        #     frame_hop=frame_hop,
        #     close_k=9,  # ~90ms Lücken schließen bei 10ms Hop
        #     open_k=5,  # ~50ms kurze Peaks entfernen
        #     min_len=100,  # min. Clusterlänge ~1s
        # )
        # results[file_id]["num_clusters"] = len(clusters)
        # results[file_id]["clusters"] = clusters
        #
        # print(f"Cluster in {file_id}: {len(clusters)}", flush=True)
        # for c in clusters[:10]:
        #     print(c, flush=True)
        #
        # assert False



    total_all = sum(r["all_error"] for r in results.values())
    total_spat_spec = sum(r["spat_spec_error"] for r in results.values())
    total_spat_comb = sum(r["spat_comb_error"] for r in results.values())
    total_spec_comb = sum(r["spec_comb_error"] for r in results.values())
    total_spec = sum(r["spec_only"] for r in results.values())
    total_spat = sum(r["spat_only"] for r in results.values())
    total_comb = sum(r["comb_only"] for r in results.values())
    total_frames = sum(r["total_frames"] for r in results.values())
    error_frames = sum(r["error_frames"] for r in results.values())

    print("=== GLOBAL ===")
    print("All error (alle drei):", total_all)
    print("Spectral + Spatial:", total_spat_spec)
    print("Spatial + Combined:", total_spat_comb)
    print("Spectral + Combined:", total_spec_comb)
    print("Only spectral:", total_spec)
    print("Only spatial:", total_spat)
    print("Only combined:", total_comb)
    print("Total frames:", total_frames)
    print("Total error frames:", error_frames)

    results["total"] = {
        "all_error": round(total_all / error_frames, 4),
        "spat_spec_error": round(total_spat_spec / error_frames, 4),
        "spat_comb_error": round(total_spat_comb / error_frames, 4),
        "spec_comb_error": round(total_spec_comb / error_frames, 4),
        "spec_only": round(total_spec / error_frames, 4),
        "spat_only": round(total_spat / error_frames, 4),
        "comb_only": round(total_comb / error_frames, 4),
        "total_frames": total_frames,
        "error_frames": error_frames,
    }

    results["spatial_vs_spectral_diff_total"] = aggregate_spatial_spectral_diff(results)
    if results["spatial_vs_spectral_diff_total"]:
        diff_total = results["spatial_vs_spectral_diff_total"]
        print("=== SPATIAL VS SPECTRAL DIFF ===")
        print("Spatial-only ratio:", diff_total["spatial_only_ratio"])
        print("Spectral-only ratio:", diff_total["spectral_only_ratio"])
        print("Both-error ratio:", diff_total["both_error_ratio"])
        print("Both-diff-type ratio:", diff_total["both_diff_type_ratio"])
        print("Spatial advantage ratio:", diff_total["spatial_advantage_ratio"])

    pb.io.dump_json(results, f"/home/deegen/forschung/DiariZen/error_analysis/average/error_counts_ov_{dset}.json")

        # ALLE
    # Cluster in EN2002a: 161
    # {'start_frame': 4, 'end_frame': 174, 'length_frames': 171, 'start_s': 0.04, 'end_s': 1.75, 'duration_s': 1.71} #  audio gar nix da eig
    # {'start_frame': 321, 'end_frame': 549, 'length_frames': 229, 'start_s': 3.21, 'end_s': 5.5, 'duration_s': 2.29}
    # {'start_frame': 681, 'end_frame': 1077, 'length_frames': 397, 'start_s': 6.8100000000000005, 'end_s': 10.78,   Lachen und haha
    #  'duration_s': 3.97}
    # {'start_frame': 1126, 'end_frame': 1233, 'length_frames': 108, 'start_s': 11.26, 'end_s': 12.34, 'duration_s': 1.08}
    # {'start_frame': 2147, 'end_frame': 2418, 'length_frames': 272, 'start_s': 21.47, 'end_s': 24.19, 'duration_s': 2.72}
    # {'start_frame': 2522, 'end_frame': 2642, 'length_frames': 121, 'start_s': 25.22, 'end_s': 26.43, 'duration_s': 1.21}
    # {'start_frame': 2785, 'end_frame': 3075, 'length_frames': 291, 'start_s': 27.85, 'end_s': 30.76, 'duration_s': 2.91}
    # {'start_frame': 3421, 'end_frame': 3804, 'length_frames': 384, 'start_s': 34.21, 'end_s': 38.050000000000004,
    #  'duration_s': 3.84}
    # {'start_frame': 3916, 'end_frame': 4425, 'length_frames': 510, 'start_s': 39.160000000000004, 'end_s': 44.26,   lachen
    #  'duration_s': 5.1000000000000005}
    # {'start_frame': 4945, 'end_frame': 5046, 'length_frames': 102, 'start_s': 49.45, 'end_s': 50.47, 'duration_s': 1.02}

    # spatial :
    # Cluster in EN2002a: 15
    # {'start_frame': 40036, 'end_frame': 40148, 'length_frames': 113, 'start_s': 400.36, 'end_s': 401.49,    #  'duration_s': 1.1300000000000001}
    # {'start_frame': 69912, 'end_frame': 70016, 'length_frames': 105, 'start_s': 699.12, 'end_s': 700.17,    #  'duration_s': 1.05}
    # {'start_frame': 83094, 'end_frame': 83263, 'length_frames': 170, 'start_s': 830.94, 'end_s': 832.64,    #  'duration_s': 1.7}
    # {'start_frame': 97814, 'end_frame': 97917, 'length_frames': 104, 'start_s': 978.14, 'end_s': 979.180000000001,    #  'duration_s': 1.04}
    # {'start_frame': 101766, 'end_frame': 101897, 'length_frames': 132, 'start_s': 1017.66, 'end_s': 1018.98,    #  'duration_s': 1.32}
    # {'start_frame': 109252, 'end_frame': 109442, 'length_frames': 191, 'start_s': 1092.52, 'end_s': 1094.43,    #  'duration_s': 1.9100000000000001}
    # {'start_frame': 127916, 'end_frame': 128038, 'length_frames': 123, 'start_s': 1279.16, 'end_s': 1280.39,    #  'duration_s': 1.23}
    # {'start_frame': 132560, 'end_frame': 132681, 'length_frames': 122, 'start_s': 1325.6000000000001, 'end_s': 1326.82,    #  'duration_s': 1.22}
    # {'start_frame': 144514, 'end_frame': 144620, 'length_frames': 107, 'start_s': 1445.14, 'end_s': 1446.21,    #  'duration_s': 1.07}
    # {'start_frame': 174416, 'end_frame': 174615, 'length_frames': 200, 'start_s': 1744.16, 'end_s': 1746.16,    #  'duration_s': 2.0}

    # Spectral
    # Cluster in EN2002a: 25
    # {'start_frame': 8707, 'end_frame': 8856, 'length_frames': 150, 'start_s': 87.07000000000001, 'end_s': 88.57000000000001, 'duration_s': 1.5}
    # {'start_frame': 13980, 'end_frame': 14087, 'length_frames': 108, 'start_s': 139.8, 'end_s': 140.88, 'duration_s': 1.08}
    # {'start_frame': 14112, 'end_frame': 14219, 'length_frames': 108, 'start_s': 141.12, 'end_s': 142.20000000000002, 'duration_s': 1.08}
    # {'start_frame': 28407, 'end_frame': 28510, 'length_frames': 104, 'start_s': 284.07, 'end_s': 285.11, 'duration_s': 1.04}
    # {'start_frame': 67532, 'end_frame': 67631, 'length_frames': 100, 'start_s': 675.32, 'end_s': 676.32, 'duration_s': 1.0}
    # {'start_frame': 72391, 'end_frame': 72506, 'length_frames': 116, 'start_s': 723.91, 'end_s': 725.07, 'duration_s': 1.16}
    # {'start_frame': 80591, 'end_frame': 80773, 'length_frames': 183, 'start_s': 805.91, 'end_s': 807.74, 'duration_s': 1.83}
    # {'start_frame': 94268, 'end_frame': 94373, 'length_frames': 106, 'start_s': 942.6800000000001, 'end_s': 943.74, 'duration_s': 1.06}
    # {'start_frame': 105246, 'end_frame': 105390, 'length_frames': 145, 'start_s': 1052.46, 'end_s': 1053.91, 'duration_s': 1.45}
    # {'start_frame': 105682, 'end_frame': 105793, 'length_frames': 112, 'start_s': 1056.82, 'end_s': 1057.94, 'duration_s': 1.12}