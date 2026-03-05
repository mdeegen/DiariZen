# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import json
from pathlib import Path
import itertools
import toml
import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
from diarizen.logger import init_logging_logger
from diarizen.utils import instantiate
from diarizen.ckpt_utils import average_ckpt
from sklearn.utils.class_weight import compute_class_weight

from dataset import _collate_fn
from functools import partial

def run(config, resume):
    logger = init_logging_logger(config)
    # log config file into output
    logger.info(f"Configuration file: {config}")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["trainer"]["args"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info

    if config["finetune"]["finetune"]:
        load_encoder = config["finetune"].get("load_encoder", None)
        accelerator.print('fine-tuning...')
        model = average_ckpt(config["finetune"]["ckpt_dir"], model, avg_ckpt_num=config["finetune"]["avg_ckpt_num"],
                             load_wavlm_only=config["finetune"].get("load_wavlm_only", False), load_encoder=load_encoder)
    dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    optimizer_small = instantiate(
        config["optimizer_small"]["path"],
        args={"params": [dummy_param]}
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
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
        gcpsd=config["meta"].get("gcpsd", False),
    )

    sample_dir = config["meta"].get("sample_dir", False)
    if sample_dir:
        with open(sample_dir) as f:
            ov_labels = json.load(f)
    if "train" in args.mode:
        train_dataset = instantiate(config["train_dataset"]["path"], args=train_dataset_config)

        if sample_dir:
            print("ACTIVATED BALANCED SAMPLING")
            labels = ov_labels["train"]
            # class_counts = np.bincount(labels)
            # weights = 1.0 / class_counts
            # print(class_counts, weights)
            # sample_weights = [weights[l] for l in labels]

            classes = np.unique(labels)
            class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
            weights = np.zeros_like(class_weights, dtype=np.float64)
            for idx, c in enumerate(classes):
                weights[c] = class_weights[idx]
            print("Class weights: ", weights)
            sample_weights = [weights[l] for l in labels]

            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

            train_dataloader = DataLoader(dataset=train_dataset, collate_fn=collate_fn_partial, sampler=sampler,
                                          shuffle=(sampler is None), **config["train_dataset"]["dataloader"]
                                          )
        else:
            train_dataloader = DataLoader(
                dataset=train_dataset, collate_fn=collate_fn_partial, shuffle=True, **config["train_dataset"]["dataloader"]
            )

        debug = config["meta"].get("debug", False)
        if debug:
            train_dataloader = list(itertools.islice(train_dataloader, 50))
            train_dataloader = [
                {
                    k: (v.to(accelerator.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                for batch in train_dataloader
            ]
        else:
            train_dataloader = accelerator.prepare(train_dataloader)

    if "train" in args.mode or "validate" in args.mode:
        validate_dataset = instantiate(config["validate_dataset"]["path"], args=validate_dataset_config)
        # resample = config["validate_dataset"]["args"].get("resample", False)

        # if sample:
        #     labels = ov_labels["dev"]
        #     class_counts = np.bincount(labels)
        #     weights = 1.0 / class_counts
        #     sample_weights = [weights[l] for l in labels]
        #     sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        #
        #     validate_dataloader = DataLoader(dataset=validate_dataset, collate_fn=collate_fn_partial, sampler=sampler,
        #                                      shuffle=(sampler is None), **config["validate_dataset"]["dataloader"]
        #                                      )
        # else:
        # TODO: Balanced samling nicht in val und test?
        # validate_dataset= validate_dataset[200*16:]

        validate_dataloader = DataLoader(
            dataset=validate_dataset, collate_fn=collate_fn_partial, shuffle=False, **config["validate_dataset"]["dataloader"]
        )

        # debug = True
        if debug:
            validate_dataloader = list(itertools.islice(validate_dataloader, 200, 210))
            validate_dataloader = [
                {
                    k: (v.to(accelerator.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                for batch in validate_dataloader
            ]
        else:
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
            trainer.train(train_dataloader, validate_dataloader)
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
