import numpy as np
import meeteval
import paderbox as pb
from collections import defaultdict
import os
import argparse

def count_overlap(rttm, out_dir=None, dset=None):
    print ("Reading RTTM file:", rttm)

    speakers = defaultdict(lambda: defaultdict(list))
    max_end = defaultdict(int)
    with open(rttm, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 10 or parts[0] != "SPEAKER":
                continue
            rec = parts[1]
            spk = parts[7]
            start = float(parts[3])
            dur = float(parts[4])
            end = start + dur

            ### Seconds to ssamples
            sr = 16000
            start = int(start * sr)
            end = int(end * sr)
            speakers[rec][spk].append((start, end))
            max_end[rec] = max(max_end[rec], end)


    dur = []
    ov = []
    ov_more = []
    extreme_ov = []
    for rec in speakers:
        num_spk = len(speakers[rec])
        act_array = []

        for spk in speakers[rec]:
            act = pb.array.interval.ArrayInterval.from_pairs(speakers[rec][spk], shape=max_end[rec])
            arr = np.array(act)
            act_array.append(arr)
        act_array = np.array(act_array, dtype=np.int8)
        # print("Number of speakers:", num_spk)
        # print(act_array.shape)
        # print(act_array[:,250:270])
        summed_act = np.sum(act_array, axis=0)
        # print(summed_act.shape)
        # print(summed_act[250:270])
        num_ov = np.sum(summed_act > 1)
        more_ov = np.sum(summed_act > 2)
        extrem_ov = np.sum(summed_act > 3)
        spk_dur = np.sum(summed_act > 0)
        dur.append(spk_dur)
        ov.append(num_ov)
        ov_more.append(more_ov)
        extreme_ov.append(extrem_ov)
        # print(spk_dur, num_ov, more_ov)

    spk_dur = np.sum(dur)
    num_ov = np.sum(ov)
    more_ov = np.sum(ov_more)
    extrem_ov = np.sum(extreme_ov)

    ov_ratio = num_ov / spk_dur
    print("Overlap ratio:", ov_ratio)
    more_ov_ratio = more_ov / spk_dur
    print("More than 2 speakers ratio:", more_ov_ratio)
    extreme_ov_ratio = extrem_ov / spk_dur
    print("More than 3 speakers ratio:", extreme_ov_ratio)

    if out_dir is not None:
        pb.io.dump_json({
            "spk_dur": spk_dur,
            "num_ov": num_ov,
            "ov_ratio": ov_ratio,
            "more_ov": more_ov,
            "3 or more speakers": more_ov_ratio,
            "extreme_ov": extrem_ov,
            "4 or more speakers": extreme_ov_ratio,

        }, out_dir)
        print("Results saved to:", out_dir)


def count_time(rttm, out_dir=None, dset=None):
    print("Reading RTTM file:", rttm)

    speakers = defaultdict(lambda: defaultdict(list))
    max_end = defaultdict(float)
    with open(rttm, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 10 or parts[0] != "SPEAKER":
                continue

            if dset =="NOTSOFAR1":
                if not parts[1].startswith("S3"):
                    continue

            rec = parts[1]
            spk = parts[7]
            start = float(parts[3])
            dur = float(parts[4])
            end = start + dur

            speakers[rec][spk].append((start, end))
            max_end[rec] = max(max_end[rec], end)

    num_spks = []
    for rec in speakers:
        num_spk = len(speakers[rec])
        num_spks.append(num_spk)

    dur = sum(max_end.values()) / (60*60 )   # in hours

    if out_dir is not None:
        pb.io.dump_json({
            "num_spk": num_spks,
            "min_max": (min(num_spks), max(num_spks)),
            "duration": dur,
        }, out_dir)
        print("Results saved to:", out_dir)
    assert False


def main(rttm, out_dir=None, dset=None):
    count_overlap(rttm, out_dir + "/overlap_test.json", dset)
    count_time(rttm, out_dir + "/time_test.json", dset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rttm", type=str, required=True, help="Path to the RTTM file")
    # parser.add_argument("-i", type=str, required=True, help="Path to the out file")
    parser.add_argument("-o", type=str, required=True, help="Path to the in audio file")
    parser.add_argument("-dset", type=str, required=True, help="Dataset split")
    args = parser.parse_args()
    rttm = args.rttm
    out_dir = args.o
    dset = args.dset

    main(rttm, out_dir, dset=dset)