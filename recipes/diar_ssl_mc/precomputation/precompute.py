import os
import numpy as np
import soundfile as sf
from pathlib import Path

import argparse
import toml
import h5py
import pdb
from tqdm import tqdm
from paderbox.io import dump_json
from kaldiio import WriteHelper
from diarizen.dataset_prepare import DiarizationDataset
import math
import dlp_mpi

def num_frames(L, n_fft, hop):
    return math.floor((L - n_fft) / hop) + 1


def pad_data(data):
    L = data.shape[-1]
    n_fft_wavlm = 400
    hop = 320
    N_ref = num_frames(L, n_fft_wavlm, hop)

    n_fft_big = 4096
    N_big = num_frames(L, n_fft_big, hop)

    L_target = (N_ref - 1) * hop + n_fft_big
    pad = max(0, L_target - L)

    last_vals = data[:, -1:]  # Shape (C,1)
    pad_block = np.repeat(last_vals, pad, axis=1)  # Shape (C,pad)
    return np.concatenate([data, pad_block], axis=1)

def precompute_gccs(config, scp_file, uem_file, out_dir, dataset, f_max_gcc, modelbased=False, apply_ifft=False):
    print("PYTHON: Precomputing GCCS")
    os.makedirs(out_dir, exist_ok=True)
    # config[f"train_dataset"]["args"]["frame_shift_gcc"]= gcc shift-...
    if "train" in str(dataset):
        config[f"train_dataset"]["args"]["scp_file"] = str(scp_file)
        config["train_dataset"]["args"]["uem_file"] = str(uem_file)
        if "rttm_file" in config["train_dataset"]["args"]:
            del config["train_dataset"]["args"]["rttm_file"]
        dataset = DiarizationDataset(
            model_num_frames=399,
            model_rf_duration=0.025,
            model_rf_step=0.02,
            verbose=True,
            f_max_gcc=f_max_gcc,
            **config["train_dataset"]["args"],
            modelbased=modelbased,
            apply_ifft=apply_ifft,
        )
    elif "dev" in str(dataset):
        config[f"validate_dataset"]["args"]["scp_file"] = str(scp_file)
        config["validate_dataset"]["args"]["uem_file"] = str(uem_file)
        if "rttm_file" in config["validate_dataset"]["args"]:
            del config["validate_dataset"]["args"]["rttm_file"]
        dataset = DiarizationDataset(
            model_num_frames=399,
            model_rf_duration=0.025,
            model_rf_step=0.02,
            verbose=True,
            f_max_gcc=f_max_gcc,
            modelbased=modelbased,
            apply_ifft=apply_ifft,
            **config["validate_dataset"]["args"],
        )
    worker_id = os.environ.get("SGE_TASK_ID", "0")
    if modelbased:
        h5_filename = out_dir / f"num_spk{worker_id}.h5"
        json_filename = out_dir / f"num_spk_index{worker_id}.json"
    else:
        h5_filename = out_dir / f"gcc_features{worker_id}.h5"
        json_filename = out_dir / f"gcc_features_index{worker_id}.json"
    if h5_filename.exists() or json_filename.exists():
        print(f"Output file {h5_filename} or {json_filename} already exists. Skipping computation.")
        # raise FileExistsError(f"Output file {h5_filename} or {json_filename} already exists.")
    index = {}
    print(h5_filename, flush=True)
    with h5py.File(h5_filename, 'w') as f:
        for i in tqdm(range(len(dataset))):
            try:
                # if i < 100:
                #     continue
                # if i > 125:
                #     break
                _, _, session, path, _, gcc_features, fmax = dataset[i]

                
                # hits = dataset.num_correct
                # total = dataset.num_total
                # acc = hits / total if total > 0 else 0.0
                # print(f"Accuracy: {acc:.4f} ({hits}/{total})")
                # under = dataset.under / total if total > 0 else 0.0
                # over = dataset.over / total if total > 0 else 0.0
                # print(f"Underestimates: {under}, Overestimates: {over}")

                chunk_start = dataset.chunk_indices[i][2]
                chunk_end = dataset.chunk_indices[i][3]
                file_stem = Path(path).stem
                segment_id = f'{file_stem}_{chunk_start}_{chunk_end}'
                if not isinstance(gcc_features, np.ndarray):
                    gcc_features = gcc_features.cpu().numpy()
                if gcc_features is None or gcc_features.size == 0:
                    print(f"[Warning] Empty gcc_features for {segment_id}")
                    continue
                grp = f.require_group(file_stem)            # sturctue file_name: file_name_chunk_start_chunk_end...
                if segment_id in grp:
                    continue  # Avoid overwriting (safe parallel use)
                grp.create_dataset(segment_id, data=gcc_features, compression="gzip")
                # grp.attrs['utt_id'] = 'sample_001'
                if file_stem not in index:
                    index[file_stem] = str(h5_filename)

            except Exception as e:
                print(f"[Worker {worker_id}] Failed to write segment {segment_id}: {e}")
                print(gcc_features.shape, "SUM", np.sum(gcc_features), grp)
        ## accuracy
        print(f"Saved GCC features to {h5_filename} and index to {json_filename}", flush=True)
        # hits = dataset.num_correct
        # total = dataset.num_total
        # acc = hits / total if total > 0 else 0.0
        # print(f"Accuracy: {acc:.4f} ({hits}/{total})")
        # under = dataset.under / total if total > 0 else 0.0
        # over = dataset.over / total if total > 0 else 0.0
        # print(f"Underestimates: {under}, Overestimates: {over}")
        # acc_dict = {"accuracy": acc, "hits": hits, "total": total, "underestimates": under, "overestimates": over}

        # # If file exists, increment index until a unique filename is found
        # acc_path = out_dir / f"acc_{worker_id}.json"
        # idx = 1
        # while acc_path.exists():
        #     acc_path = out_dir / f"acc_{worker_id}_{idx}.json"
        #     idx += 1
        # dump_json(acc_dict, acc_path)
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
    # parser.add_argument("--modelbased", required=True, help="modelbased")
    # parser.add_argument("--f_max_gcc", type=lambda x: None if x == "None" else float(x), required=True,
    #                     help="which freq to filter, None for no upper limit (pass 'None')")
    args = parser.parse_args()


    config_path = Path(args.config).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    scp_file = Path(args.scp_file)
    out_dir = Path(args.out_dir)
    uem_file = Path(args.uem_file)
    dataset = Path(args.dataset)
    # modelbased = args.modelbased.lower() == "true"

    if isinstance(args.f_max_gcc, str):
        if args.f_max_gcc.lower() == "none":
            f_max_gcc = None
        elif args.f_max_gcc.replace('.', '', 1).isdigit() or args.f_max_gcc.isdigit():
            f_max_gcc = float(args.f_max_gcc)
        else:
            assert False, f"Invalid value for f_max_gcc: {args.f_max_gcc}"
    else:
        f_max_gcc = None if args.f_max_gcc is None else float(args.f_max_gcc)


    # scp_file = Path("/mnt/scratch/tmp/qdeegen/split_dir/train/wav.scp.")
    # out_dir = Path("/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size2048_shift316/train")
    # uem_file = Path("/mnt/scratch/tmp/qdeegen/split_dir/train/all.uem.0000")
    # dataset = Path("train")
    # config_path = Path("/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/conf/gcc_encoder.toml").expanduser().absolute()
    # config = toml.load(config_path.as_posix())

    precompute_gccs(config, scp_file, uem_file, out_dir, dataset, f_max_gcc)
