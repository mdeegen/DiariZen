import os
import numpy as np
import paderbox as pb
from collections import defaultdict

from padercontrib.speech_separation.stitcher import solve_permutation_hungarian


def solve_permutation(activities, ref_activities):
    spks = np.maximum(len(activities.keys()), len(ref_activities.keys()))
    overlaps = np.zeros((spks, spks))
    for i, (spk, act) in enumerate(activities.items()):
        for j, (ref_spk, ref_act) in enumerate(ref_activities.items()):
            ref_act.shape = act.shape
            overlaps[i, j] = np.sum(act & ref_act)

    permutations = solve_permutation_hungarian(overlaps, minimize=False)
    # np.arange (spks), permutations
    return permutations

def load_rttm(rttm_path):
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue  # überspringe leere Zeilen
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue  # nur Zeilen mit Typ SPEAKER berücksichtigen

            # RTTM-Format: SPEAKER <file-id> <channel> <start-time> <duration> ...
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            speaker = parts[7]

            segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker
            })
    return segments


def get_activity(hyp1, hyp2, sr=16000):
    """
    Load two RTTM files and return their content as lists of dictionaries.
    Each dictionary represents a segment with keys like 'start', 'end', 'label'.
    """
    spk_act1 = {}
    spk_act2 = {}
    # for i, hyp in enumerate([hyp1, hyp2]):
    seg_list = load_rttm(hyp1)
    seg_list2 = load_rttm(hyp2)
    spk_intervals = defaultdict(list)
    spk_intervals2 = defaultdict(list)

    duration = int(np.ceil(max(seg["end"] for seg in seg_list) * sr))
    duration2 = int(np.ceil(max(seg["end"] for seg in seg_list2) * sr))
    duration = max(duration, duration2)

    for seg in seg_list:
        start_idx = int(np.floor(seg["start"] * sr))
        end_idx = int(np.ceil(seg["end"] * sr))
        spk_intervals[seg["speaker"]].append((start_idx, end_idx))
    for seg in seg_list2:
        start_idx = int(np.floor(seg["start"] * sr))
        end_idx = int(np.ceil(seg["end"] * sr))
        spk_intervals2[seg["speaker"]].append((start_idx, end_idx))
    for spk in spk_intervals:
        spk_act1[spk] = pb.array.interval.ArrayInterval.from_pairs(spk_intervals[spk], shape=duration)
    for spk in spk_intervals2:
        spk_act2[spk] = pb.array.interval.ArrayInterval.from_pairs(spk_intervals2[spk], shape=duration)
        # print(f"Speaker {spk} has {len(spk_act1[spk])} segments with duration {duration} at {sr}Hz")


    return spk_act1, spk_act2


def combine_act(hyp_gcc, hyp_wavlm, outdir, sr=16000):
    """
    loads the hypothesis from the two rttms files and combines the estimated activity with locial or between the two hypothesis
    """
    os.makedirs(outdir, exist_ok=True)
    session = os.path.basename(hyp_gcc).replace(".rttm", "")
    spk_act_gcc, spk_act_wavlm = get_activity(hyp_gcc, hyp_wavlm)
    # print(f"Loaded {len(spk_act_gcc)} segments from {spk_act_gcc} and {len(spk_act_wavlm)} segments from {spk_act_wavlm}")
    if len(spk_act_gcc.keys()) > len(spk_act_wavlm.keys()):
        min_activity_spk = min(spk_act_gcc, key=lambda spk: spk_act_gcc[spk].sum())
        print(f"Removing speaker {min_activity_spk} from spk_act_gcc due to minimal activity")
        del spk_act_gcc[min_activity_spk]
    permutations = solve_permutation(spk_act_gcc, spk_act_wavlm)
    print("Perm:", permutations, "WAV:", spk_act_wavlm.keys(), "GCC: ", spk_act_gcc.keys())
    spk_mapping = {spk: list(spk_act_wavlm.keys())[permutations[i]] for i, spk in enumerate(spk_act_gcc.keys())}
    print(f"Mapping of speakers: {spk_mapping}")
    comb_act = {}

    # #     for (start, end) in spk_act_gcc[spk].normalized_intervals():
    #  TODO:
    # if interval bei gcc aber nicht bei wavlm, dann bei adneren sprechern von wavlm gucken und wenn finden dann zu richtigem sprecher schieben
    # if intervall bei wavlm

    for spk in spk_act_gcc:
        combined_spk_act = spk_act_gcc[spk] | spk_act_wavlm[spk_mapping[spk]]
        assert combined_spk_act.sum() >= spk_act_gcc[spk].sum()
        comb_act[f"speaker{spk}"] = spk_act_gcc[spk] | spk_act_wavlm[spk_mapping[spk]]
    pb.array.interval.to_rttm({session: comb_act}, outdir + f"/{session}.rttm")


    return









if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Combine two activity hypotheses")
    # parser.add_argument("hyp1", type=str, help="Path to the first hypothesis file")
    # parser.add_argument("hyp2", type=str, help="Path to the second hypothesis file")
    # args = parser.parse_args()
    # hyp1 = args.hyp1
    # hyp2 = args.hyp2
    eval_dir = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_gcc/exp/initial/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc"
    eval_dir2 = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/baseline_sc/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc"
    outdir = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/combined/test_marc"
    for dataset in os.listdir(eval_dir):
        for session in os.listdir(os.path.join(eval_dir, dataset)):
            if not session.endswith("rttm"):
                continue
            hyp1 = os.path.join(eval_dir, dataset, f"{session}")
            hyp2 = os.path.join(eval_dir2, dataset, f"{session}")
            if os.path.exists(hyp1) and os.path.exists(hyp2):
                print(f"Combining {hyp1} and {hyp2}")
                # combine rttms and write in outdir
                combine_act(hyp1, hyp2, outdir + f"/{dataset}/")
            else:
                print(f"Missing files for {dataset}/{session}: {hyp1}, {hyp2}")
