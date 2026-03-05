# Licensed under the MIT license.
# Copyright 2022 Brno University of Technology (author: Federico Landini, landini@fit.vut.cz)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
from pathlib import Path

import torch
import torch.nn as nn

import copy

from typing import List, Dict


def average_checkpoints(
    model: nn.Module,
    checkpoint_list: str,
) -> nn.Module:
    states_dict_list = []
    for ckpt_data in checkpoint_list:
        ckpt_path = ckpt_data['bin_path']
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        copy_model.load_state_dict(checkpoint)
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list, torch.device('cpu'))
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model

def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)
    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state

def load_metric_summary(metric_file, ckpt_path):
    with open(metric_file, "r") as f:
        lines = f.readlines()
    out_lst = []
    for line in lines:
        assert "Validation Loss/DER" in line
        epoch = line.split()[4].split(':')[0]
        Loss, DER = line.split()[-3], line.split()[-1]
        bin_path = f"epoch_{str(epoch).zfill(4)}/pytorch_model.bin"
        out_lst.append({
            'epoch': int(epoch),
            'bin_path': ckpt_path / bin_path,
            'Loss': float(Loss),
            'DER': float(DER)
        })
    return out_lst

def partly_load(module, prefix, ckpt):
    sub_state_dict = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    # print(ckpt.items(), prefix)
    print(f"Partly loading {len(sub_state_dict)} keys with prefix '{prefix}'")
    if sub_state_dict:
        module.load_state_dict(sub_state_dict, strict=True)

def add_prefix_to_state_dict(state_dict, prefix):
    return {f"{prefix}{k}": v for k, v in state_dict.items()}

def fix_keys(pretrained_state, old_prefix, new_prefix):
    new_state = {}
    for key, value in pretrained_state.items():
        if key.startswith(f"{old_prefix}."):
            # proj -> proj_gcc
            new_key = key.replace(f"{old_prefix}", f"{new_prefix}", 1)
        else:
            new_key = key
        new_state[new_key] = value
    return new_state

def average_ckpt(ckpt_dir, model, val_metric='Loss', avg_ckpt_num=5, val_mode="best", load_wavlm_only=False,
                 load_encoder=None, load_spk_counting=False, load_der_encoder=False, dont_load_wavlm=False):

    # check if dir exists otherwise switch prefix to noctua2
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = ckpt_dir.replace("/mnt/matylda3/ihan/project/diarization/huggingface_hub/", "/scratch/hpc-prf-nt2/deegen/merlin/wavlm_pruned/")
        ckpt_dir = ckpt_dir.replace("/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/", "/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/")

    if os.path.isfile(ckpt_dir):
        print(f"No model averaging | Fine-tune model from: {ckpt_dir}")
        ckpt_loaded = torch.load(ckpt_dir, map_location=torch.device('cpu'))
        # model.load_state_dict(ckpt_loaded, strict=False)
        if load_wavlm_only:
            print("Loading WavLM only...")
            partly_load(model.wavlm_model, "wavlm_model.", ckpt_loaded)
            partly_load(model.weight_sum, "weight_sum.", ckpt_loaded)
            partly_load(model.proj, "proj.", ckpt_loaded)
            partly_load(model.lnorm, "lnorm.", ckpt_loaded)
        else: # if not dont_load_wavlm:
            missing, unexpected = model.load_state_dict(ckpt_loaded, strict=False)
            print("Missing Keys at model.load :", missing)
            print("Unexpected Keys at model.load :", unexpected)
        if load_encoder is not None:
            print("Loading encoder from:", load_encoder)
            enc_ckpt_loaded = torch.load(load_encoder, map_location=torch.device('cpu'))
            new_state = fix_keys(enc_ckpt_loaded)
            partly_load(model.gcc_encoder, "gcc_encoder.", new_state)
            partly_load(model.proj_gcc, "proj_gcc.", new_state)
            partly_load(model.lnorm_gcc, "lnorm_gcc.", new_state)
        if load_spk_counting:
            print("Loading speaker counting head...")
            ckpt_loaded_spk = torch.load(load_spk_counting, map_location=torch.device('cpu'))
            # model.spk_counting.load_state_dict(ckpt_loaded_spk, strict=True)
            partly_load(model.spk_counting, "gcc_encoder.", ckpt_loaded_spk)
        if load_der_encoder:
            print("Loading Encoder system for DER...")
            ckpt_loaded_spk = torch.load(load_der_encoder, map_location=torch.device('cpu'))
            if hasattr(model, 'gcc_encoder'):
                model.gcc_encoder.load_state_dict(ckpt_loaded_spk, strict=True)
            else:
                model.encoder.load_state_dict(ckpt_loaded_spk, strict=True)


        print("Model loaded successfully.")
        return model

    if 'checkpoints/epoch' in ckpt_dir:
        print(f"No model averaging | Fine-tune model from certain epoch: {ckpt_dir.split('/')[-1]}")
        ckpt_loaded = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model.load_state_dict(ckpt_loaded)
        return model

    assert val_metric == "Loss" and val_mode == "best"
    print(f'averaging best {avg_ckpt_num} checkpoints to the converged moment...')

    ckpt_dir = Path(ckpt_dir).expanduser().absolute()
    ckpt_path = ckpt_dir / 'checkpoints'
    val_metric_path = ckpt_dir / 'val_metric_summary.lst'

    val_metric_lst = load_metric_summary(val_metric_path, ckpt_path)
    val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[val_metric])
    best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
    val_metric_lst_out = val_metric_lst[
            best_val_metric_idx - avg_ckpt_num + 1 :
            best_val_metric_idx + 1
    ]

    return average_checkpoints(model, val_metric_lst_out)

def average_ckpt_old(ckpt_dir, model, val_metric='Loss', avg_ckpt_num=5, val_mode="prev"):
    if 'checkpoints/epoch_' in ckpt_dir:
        print(f"No model averaging | Fine-tune model from: {ckpt_dir.split('/')[-1]}")
        ckpt_loaded = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        # model.load_state_dict(ckpt_loaded)
        missing, unexpected = model.load_state_dict(ckpt_loaded, strict=False)
        print("Missing Keys:", missing)
        print("Unexpected Keys:", unexpected)

        return model

    assert val_metric == "Loss" and val_mode == "prev"
    print(f'averaging previous {avg_ckpt_num} checkpoints to the converged moment...')

    ckpt_dir = Path(ckpt_dir).expanduser().absolute()
    ckpt_path = ckpt_dir / 'checkpoints'
    val_metric_path = ckpt_dir / 'val_metric_summary.lst'

    val_metric_lst = load_metric_summary(val_metric_path, ckpt_path)
    val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[val_metric])
    best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
    val_metric_lst_out = val_metric_lst[
            best_val_metric_idx - avg_ckpt_num + 1 :
            best_val_metric_idx + 1
    ]
    
    return average_checkpoints(model, val_metric_lst_out)
