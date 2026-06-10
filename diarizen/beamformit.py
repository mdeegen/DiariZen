#!/usr/bin/env python3

import subprocess
from pathlib import Path


def get_mic_selection(rec):
    if rec.startswith(("mmcs")):
        mics = [0, 2, 3, 4]  # for NSF
    else:
        mics = [0, 2, 4, 6]  # default
    return mics

in_dir = Path("/scratch/hpc-prf-nt2/db/mmcsg/mmcsg_16")
out_dir = Path("/scratch/hpc-prf-nt2/db/AMI_AIS_ALI_NSF_CHiME7/split/mmcsg")
out_dir.mkdir(parents=True, exist_ok=True)



for dset in ["train", "dev", "eval"]: # "dev", "train",
    input_dir = in_dir / dset
    output_dir = out_dir / dset
    sessions =  []
    output_dir.mkdir(parents=True, exist_ok=True)
    channels_file_path = f"/scratch/hpc-prf-nt2/db/mmcsg2/channels_file_{dset}"
    with open(channels_file_path, "w") as f:
        for flac_file in sorted(input_dir.glob("*.wav")):
            base = flac_file.stem
            if "P" in base:     #base.startswith("S0") or base.startswith("S1") or base.startswith("S2"):
                continue
            if "." in base:
                base = base.split(".")[0]
            if base in sessions:
                continue
            sessions.append(base)
            print(f"Processing {base}...")

            # get number of channels
            cmd = ["soxi", "-c", str(flac_file)]
            n_channels = int(subprocess.check_output(cmd).decode().strip())

            channel_files = []
            print("BASE", base, flush=True)
            channels = get_mic_selection(base)     # get_mic_selection(base)
            for ch in channels:
                assert ch <= n_channels, (n_channels, ch, "Ch cant be larger than num channels")
                ch = ch + 1 # sox startet bei 1 und numpy bei 0
                out_file = output_dir / f"{base}.CH{ch}.wav" #  / f"{base}"

                subprocess.run([
                    "sox",
                    str(flac_file),
                    str(out_file),
                    "remix",
                    str(ch)
                ], check=True)

                channel_files.append(out_file.name)

            # write channels file entry
            line = base + " " + " ".join(channel_files) + "\n"
            f.write(line)
    print(f"FINISHED {dset} ")

print(f"\nDone! Channels file written to: {channels_file_path}")