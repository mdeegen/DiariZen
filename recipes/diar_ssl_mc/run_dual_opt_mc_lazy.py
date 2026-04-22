# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
from pathlib import Path
import itertools
import toml

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from diarizen.logger import init_logging_logger
from diarizen.utils import instantiate
from diarizen.ckpt_utils import average_ckpt
from torch.utils.data import Dataset, IterableDataset

from dataset_lazy import _collate_fn, IterableWrapper
from dataset import _collate_fn as _collate_fn_non_lazy
from functools import partial



def run(config, resume):
    logger = init_logging_logger(config)
    # log config file into output
    logger.info(f"Configuration file: {config}")
    # # TODO: check find_unused_parameters is neccessary or not
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["trainer"]["args"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
        # split_batches=False,
        # dispatch_batches=None,
        # dispatch_batches = False,
        # use_distributed_sampler=False,
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info
    
    if config["finetune"]["finetune"]:
        load_encoder = config["finetune"].get("load_encoder", None)  #  TODO: laod encoder kaputt? load_gcc mismatcht load_encoder!!!!!!!!!
        accelerator.print('fine-tuning...')
        load_wav = config["finetune"].get("load_wavlm_only", False)
        chkp_dir_clean_labels = config["finetune"].get("chkp_dir_clean_labels", None)
        load_der_encoder = config["finetune"].get("load_der_encoder", False)
        # dont_load_wavlm = config["finetune"].get("dont_load_wavlm", False)

        if chkp_dir_clean_labels is not None:
            accelerator.print(f'Fine-tuning from a model trained with labels: {chkp_dir_clean_labels}')
            model = average_ckpt(chkp_dir_clean_labels, model, avg_ckpt_num=config["finetune"]["avg_ckpt_num"],
                                 load_wavlm_only=load_wav, load_encoder=load_encoder, load_spk_counting=config["finetune"]["load_spk_counting"],
                                 load_der_encoder=load_der_encoder, ) #dont_load_wavlm=dont_load_wavlm)
        else:
            model = average_ckpt(config["finetune"]["ckpt_dir"], model, avg_ckpt_num=config["finetune"]["avg_ckpt_num"],
                                 load_wavlm_only=load_wav, load_encoder=load_encoder, load_spk_counting=config["finetune"].get("load_spk_counting", None),
                                 load_der_encoder=load_der_encoder,) # dont_load_wavlm=dont_load_wavlm)

    # optimizer_small = instantiate(
    #     config["optimizer_small"]["path"],
    #     args={"params": model.pretrained_parameters()}
    #     | config["optimizer_small"]["args"]
    #     | {"lr": config["optimizer_small"]["args"]["lr"]},
    # )
    # optimizer_big = instantiate(
    #     config["optimizer_big"]["path"],
    #     args={"params": model.channel_fusion_parameters()}
    #     | config["optimizer_big"]["args"]
    #     | {"lr": config["optimizer_big"]["args"]["lr"]},
    # # )
    freeze_wavlm = config["finetune"].get("freeze_wavlm", False)

    if freeze_wavlm:
        dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        optimizer_small = instantiate(
            config["optimizer_small"]["path"],
            args={"params": [dummy_param]}
                 | config["optimizer_small"]["args"]
                 | {"lr": config["optimizer_small"]["args"]["lr"]},
        )
    else:
        optimizer_small = instantiate(
            config["optimizer_small"]["path"],
            args={"params": model.wavlm_model.parameters()}
            | config["optimizer_small"]["args"]
            | {"lr": config["optimizer_small"]["args"]["lr"]},
        )
    optimizer_big = instantiate(
        config["optimizer_big"]["path"],
        args={"params": model.non_wavlm_parameters()}
        | config["optimizer_big"]["args"]
        | {"lr": config["optimizer_big"]["args"]["lr"]},
    )
    
    (model, optimizer_small, optimizer_big) = accelerator.prepare(model, optimizer_small, optimizer_big)

    # pass model receptive field info to dataset
    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step

    validate_dataset_config = config["validate_dataset"]["args"]
    validate_dataset_config["model_num_frames"] = model_num_frames
    validate_dataset_config["model_rf_duration"] = model_rf_duration
    validate_dataset_config["model_rf_step"] = model_rf_step

    collate_fn_partial = partial(
        _collate_fn,
        # acc = accelerator,
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
        # noisy_labels=config["trainer"]["args"].get("noisy_labels", False),
        # noise_prob=config["trainer"]["args"].get("noise_prob", 0.2),
        gcpsd=config["meta"].get("gcpsd", False),
        only_waveform=config["train_dataset"]["args"].get("only_wav", False),
        bf=config["train_dataset"]["args"].get("beamformit", False),
    )
    _collate_fn_non_lazy_partial = partial(
        _collate_fn_non_lazy,
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
        # noisy_labels=config["trainer"]["args"].get("noisy_labels", False),
        # noise_prob=config["trainer"]["args"].get("noise_prob", 0.2),
        gcpsd=config["meta"].get("gcpsd", False),
        only_waveform=config["train_dataset"]["args"].get("only_wav", False)
    )

    # accelerator.state.use_distributed_sampler = False
    if "train" in args.mode:
        if "lazy" in config["train_dataset"]["path"]:
            train_dataset_config["acc"] = accelerator
            train_dataset_config["num_workers"] = config["train_dataset"]["dataloader"]["num_workers"]
            train_dataset_config["batch_size"] = config["train_dataset"]["dataloader"].get("batch_size", 16),
            train_dataset_config["gradient_accumulation_steps"] = config["trainer"]["args"]["gradient_accumulation_steps"]
            train_dataset = instantiate(config["train_dataset"]["path"], args=train_dataset_config).lazy

            # import itertools
            # starting = 1999 * 16  # 2000 batches skipped
            # train_dataset2 = itertools.islice(train_dataset, starting, None)
            #
            # class IterableWrapper(IterableDataset):
            #     def __init__(self, dataset, len=None):
            #         self.dataset = dataset
            #         self.get_my_length = len
            #
            #     def __iter__(self):
            #         return iter(self.dataset)
            #
            #     def get_my_length(self):
            #         # if self._len is not None:
            #         return self.get_my_length
            #         # return len(self.dataset)
            #
            #     def __len__(self):
            #         return self.get_my_length
            #
            # train_dataset = IterableWrapper(train_dataset2, len=len(train_dataset) - starting) # world *
            # train_dataloader = SizedIterator(train_dataset2, max(0, len(train_dataset) - starting))
            # train_dataset = itertools.islice(train_dataset, 100, None)

            # train_dataset = IterableWrapper(train_dataset)
            # acc = accelerator
            # world = acc.num_processes
            # rank = acc.process_index
            #
            # lazy = train_dataset.lazy.filter(lambda _, i: i % world == rank)
            # train_dataset = apply_mappings(train_dataset, lazy)
            # print("Wrapper length:", len(train_dataset))
            train_dataloader = DataLoader(
                dataset=train_dataset, collate_fn=collate_fn_partial, shuffle=False,  **config["train_dataset"]["dataloader"]  # sampler=None,
            )

            # train_dataloader = accelerator.prepare(train_dataloader)
            # print(type(train_dataloader))

            # print("After prepare length:", len(train_dataset))
        else:
            train_dataset = instantiate(config["train_dataset"]["path"], args=train_dataset_config)
            train_dataloader = DataLoader(
                dataset=train_dataset, collate_fn=_collate_fn_non_lazy_partial, shuffle=False,
                **config["train_dataset"]["dataloader"]
            )
            train_dataloader = accelerator.prepare(train_dataloader)

    if "train" in args.mode or "validate" in args.mode:
        # TODO: dev doch nicht lokal shufflen
        if "lazy" in config["validate_dataset"]["path"]:
            validate_dataset_config["acc"] = accelerator
            validate_dataset_config["num_workers"] = config["validate_dataset"]["dataloader"]["num_workers"]
            validate_dataset_config["batch_size"] = config["validate_dataset"]["dataloader"].get("batch_size", 16),
            validate_dataset = instantiate(config["validate_dataset"]["path"], args=validate_dataset_config).lazy
            # validate_dataset= validate_dataset[200*16:]

            validate_dataloader = DataLoader(
                dataset=validate_dataset, collate_fn=collate_fn_partial, shuffle=False, **config["validate_dataset"]["dataloader"]   # sampler=None,
            )
        else:
            validate_dataset = instantiate(config["validate_dataset"]["path"], args=validate_dataset_config)
            # validate_dataset= validate_dataset[200*16:]

            validate_dataloader = DataLoader(
                dataset=validate_dataset, collate_fn=_collate_fn_non_lazy_partial, shuffle=False, **config["validate_dataset"]["dataloader"]   # sampler=None,
            )

            validate_dataloader = accelerator.prepare(validate_dataloader)
        # # validate_dataloader = itertools.islice(validate_dataloader, 105, None)

    # if config["meta"]["precompute_gcc"]:
    #     from diarizen.spatial_features.precompute import precompute_gccs
    #     precompute_gccs(config)

    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        accelerator=accelerator,
        config=config,
        resume=resume,
        model=model,
        optimizer_small=optimizer_small,
        optimizer_big=optimizer_big
    )

    for flag in args.mode:
        if flag == "train":
            try:
                trainer.train(train_dataloader, validate_dataloader)
            except Exception as e:
                print(f"Training failed due to {e}.", flush=True)
                raise e
        elif flag == "validate":
            trainer.validate(validate_dataloader)
        else:
            raise ValueError(f"Unknown mode: {flag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-ZEN based EEND framework")
    parser.add_argument(
        "-C",
        "--configuration",
        required=True,
        type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-M",
        "--mode",
        nargs="+",
        type=str,
        default=["train"],
        choices=["train", "validate"],
        help="Mode of the experiment.",
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )
    parser.add_argument(
        "-FT",
        "--finetune",
        action="store_true",
        help="Label of fine-tuning.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint path for fine-tuning.",
    )

    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    config["meta"]["exp_id"] = config_path.stem
    config["meta"]["config_path"] = config_path.as_posix()

    run(config, args.resume)
