import os
import numpy as np
import soundfile as sf
from pathlib import Path

import argparse
import toml
import pdb
from kaldiio import WriteHelper
from DiariZen.recipes.diar_ssl_mc.dataset import DiarizationDataset


def precompute_gccs(config, scp_file, uem_file, rttm_file):

    gcc_dir = Path("/mnt/matylda5/qdeegen/data/AMI_AIS_ALI_NSF_CHiME7/data/gccs") / "standard_gcc"
    os.makedirs(gcc_dir, exist_ok=True)
    config["train_dataset"]["args"]["scp_file"] = str(scp_file)
    config["train_dataset"]["args"]["uem_file"] = str(uem_file)
    config["train_dataset"]["args"]["rttm_file"] = str(rttm_file)

    dataset = DiarizationDataset(
        model_num_frames=1000,
        model_rf_duration=2.0,
        model_rf_step=0.25,
        # scp_file=scp_file,      #"data_mc/train/wav.scp",
        # rttm_file = rttm_file,  #  "data_mc/train/rttm",
        # uem_file = uem_file,    #  "data_mc/train/all.uem",
        **config["train_dataset"]["args"],
    )
    out_dir = gcc_dir
    pdb.set_trace()

    # for i in range dlp mpi parrallelisieren? oder geht das nicht mit dominiks idee zsm
    for i in range(len(dataset)):
        _, _, session, path, _, gcc_features = dataset[i]
        chunk_start = dataset.chunk_indices[i][2]
        chunk_end = dataset.chunk_indices[i][3]
        file_stem = Path(path).stem
        pdb.set_trace()

        ### numpy save
        out_path = out_dir / f'{file_stem}_{chunk_start}_{chunk_end}.npy'
        np.save(out_path, gcc_features)

        # ### kaldi ark save?
        # segment_id = f'{file_stem}_{chunk_start}_{chunk_end}'
        # with WriteHelper('ark,scp:file.ark,file.scp') as writer:
        #     writer(segment_id, gcc_features)

    # from kaldiio import ReadHelper
    # with ReadHelper('scp:file.scp') as reader:
    #     for key, numpy_array in reader:
    return

# store in exp folder und mach sym link in data folder wenn nicht rechnen willst

# skript das scp zu save gcc zb in ark scp
# not kaldi.und dann großes scp in batches unterteilen und viele jobs mit mini scps starten


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config toml file")
    parser.add_argument("--scp_file", type=str, required=True, help="Path to scp file")
    parser.add_argument("--uem_file", type=str, required=True, help="Path to uem file")
    parser.add_argument("--rttm_file", type=str, required=True, help="Path to rttm file")
    args = parser.parse_args()


    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    scp_file = Path(args.scp_file)
    uem_file = Path(args.uem_file)
    rttm_file = Path(args.rttm_file)

    precompute_gccs(config, scp_file, uem_file, rttm_file)