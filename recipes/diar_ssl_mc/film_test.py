import numpy as np
import torch

from diarizen.utils import instantiate
from diarizen.ckpt_utils import average_ckpt
from torch.utils.data import DataLoader
from dataset import _collate_fn
from functools import partial


def main():
    print("START")
    args = {
        "ffn": 1024,
        "film_dim": 11,
        "film": True,
        "film_layers": True,  # conf mit layers amchen
        "num_layer_aux": 5,
        "max_num_spk": 4,
        "sin_cos": True,
        "attention_in_aux": 216,  # 2*513 + 0 0 für /4
        "wavlm_src": "wavlm_base_s80_md",
        "wavlm_layer_num": 13,
        "wavlm_feat_dim": 768,
        "attention_in": 256,
        "ffn_hidden": 1024,
        "num_head": 4,
        "num_layer": 4,
        "dropout": 0.1,
        "chunk_size": 8,
        "use_posi": False,
        "output_activate_function": False,
        "max_speakers_per_chunk": 4,
        "selected_channel": 0,
    }

    # ckpt_dir = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/spk_count_linear_noisy_to_gcpsd_encoder_ffn_film_all_layers/checkpoints/best/pytorch_model.bin"
    # model = instantiate("diarizen.models.eend.model_encoder_gcpsd_film.Model", args=args)
    ckpt_dir = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/spk_count_linear_noisy_to_gcpsd_encoder_ffn_film_all_layers_finetune/checkpoints/best/pytorch_model.bin"
    model = instantiate("diarizen.models.eend.model_encoder_gcpsd_film_finetune.Model", args=args)
    # ckpt_dir = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/spk_count_film/checkpoints/best/pytorch_model.bin"
    # model = instantiate("diarizen.models.eend.model_spk_count_linear.Model", args=args)

    model = average_ckpt(ckpt_dir, model, avg_ckpt_num=1,)
    model.eval()

    train_dataset_args = {
        "load_gcc_dir": "/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcpsd_1024_new/",
        "gcpsd": True,
        "subset": "train",
        "scp_file": "data_no_chime/train/wav.scp",
        "rttm_file": "data_no_chime/train/rttm",
        "uem_file": "data_no_chime/train/all.uem",
        "chunk_size": 8,
        "chunk_shift": 6,
        "sample_rate": 16000,
        "channel_mode": "multichannel",
        "num_spk": True,
        "model_num_frames": 399,
        "model_rf_duration": 0.025,
        "model_rf_step": 0.02,
    }
    train_dataloader_args = {
        "batch_size": 16,
        "num_workers": 1,
        "drop_last": True,
        "pin_memory": True,
        "prefetch_factor": 20,
    }

    print("DATASTART")
    train_dataset = instantiate("dataset.DiarizationDataset", args=train_dataset_args)

    collate_fn_partial = partial(
        _collate_fn,
        max_speakers_per_chunk= 4,
        noisy_labels= False,
        noise_prob= 0.2,
        gcpsd= True,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, collate_fn=collate_fn_partial, shuffle=True, **train_dataloader_args
    )
    activations = []
    print("data laoded")
    def save_gamma_beta(name):
        # returns the hook function (basically just ovehead for different hooks)
        def hook(module, input, output):
            activation = {}
            # output ist das, was deine FiLM forward zurückgibt (gamma*x + beta)
            cond = input[1]  # das zweite Input ist cond
            with torch.no_grad():
                gamma_beta = module.mlp(cond)
                gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
                print("FILM LAYER:", name, gamma.shape, beta.shape)
                print(name, gamma.mean().item(), beta.mean().item())
                print("VAR", gamma.var().item(), beta.var().item())
                activation[name] = {
                    "gamma": gamma.detach().cpu(),
                    "beta": beta.detach().cpu()
                }
                activations.append(activation)
        return hook
    # Hook registrieren

    for i, layer in enumerate(model.conformer.film_layers):
        layer.register_forward_hook(save_gamma_beta(f"film_{i}"))


    model.film.register_forward_hook(save_gamma_beta("film1"))
    with torch.no_grad():
        batch = next(iter(train_dataloader))
        xs, target, gccs, num_spk = batch['xs'], batch['ts'], batch["gccs"], batch["num_spks"]

        output = model(xs, gccs)
    import pdb
    pdb.set_trace()

    # print(activations)

    return


if __name__ == "__main__":
    main()