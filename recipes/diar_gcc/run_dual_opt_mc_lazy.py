# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import numpy as np
import toml
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from dataset import _collate_fn as _collate_fn_non_lazy
from dataset_lazy import IterableWrapper, _collate_fn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from diarizen.ckpt_utils import average_ckpt
from diarizen.logger import init_logging_logger
from diarizen.utils import instantiate


def run(config, resume):
    logger = init_logging_logger(config)
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    # log config file into output
    logger.info(f"Configuration file: {config}")
    # # TODO: check find_unused_parameters is neccessary or not
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["trainer"]["args"][
            "gradient_accumulation_steps"
        ],
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info

    if config["finetune"]["finetune"]:
        load_encoder = config["finetune"].get("load_encoder", None)
        csd = config["finetune"].get("csd", False)
        load_wavlm_only = config["finetune"].get("load_wavlm_only", False)
        accelerator.print("fine-tuning...")
        model = average_ckpt(
            config["finetune"]["ckpt_dir"],
            model,
            avg_ckpt_num=config["finetune"]["avg_ckpt_num"],
            load_wavlm_only=load_wavlm_only,
            load_encoder=load_encoder,
            csd=csd,
        )
    dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    # frozen_wavlm = config["model"]["args"].get("wavlm_frozen", True)
    # if frozen_wavlm:
    optimizer_small = instantiate(
        config["optimizer_small"]["path"],
        args={"params": [dummy_param]}
        | config["optimizer_small"]["args"]
        | {"lr": config["optimizer_small"]["args"]["lr"]},
    )
    # else:
    #     optimizer_small = instantiate(
    #         config["optimizer_small"]["path"],
    #         args={"params": model.wavlm.parameters()}
    #             | config["optimizer_small"]["args"]
    #             | {"lr": config["optimizer_small"]["args"]["lr"]},
    #     )
    optimizer_big = instantiate(
        config["optimizer_big"]["path"],
        args={"params": model.non_wavlm_parameters()}
        | config["optimizer_big"]["args"]
        | {"lr": config["optimizer_big"]["args"]["lr"]},
    )

    (model, optimizer_small, optimizer_big) = accelerator.prepare(
        model, optimizer_small, optimizer_big
    )

    spk_count_loss = config["trainer"]["args"].get("spk_count_loss", False)
    # pass model receptive field info to dataset
    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step
    train_dataset_config["spk_count_loss"] = spk_count_loss

    validate_dataset_config = config["validate_dataset"]["args"]
    validate_dataset_config["model_num_frames"] = model_num_frames
    validate_dataset_config["model_rf_duration"] = model_rf_duration
    validate_dataset_config["model_rf_step"] = model_rf_step
    validate_dataset_config["spk_count_loss"] = spk_count_loss

    collate_fn_partial = partial(
        _collate_fn,
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
        gcpsd=config["meta"].get("gcpsd", False),
        only_waveform=spk_count_loss,
    )
    _collate_fn_non_lazy_partial = partial(
        _collate_fn_non_lazy,
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
        gcpsd=config["meta"].get("gcpsd", False),
    )
    # sample_dir = config["meta"].get("sample_dir", False)
    # if sample_dir:
    #     with open(sample_dir) as f:
    #         ov_labels = json.load(f)
    # accelerator.state.use_distributed_sampler = False
    if "train" in args.mode:
        train_dataset_config["acc"] = accelerator
        train_dataset_config["num_workers"] = config["train_dataset"]["dataloader"][
            "num_workers"
        ]
        train_dataset_config["batch_size"] = (
            config["train_dataset"]["dataloader"].get("batch_size", 16),
        )
        train_dataset_config["gradient_accumulation_steps"] = config["trainer"]["args"][
            "gradient_accumulation_steps"
        ]
        train_dataset = instantiate(
            config["train_dataset"]["path"], args=train_dataset_config
        ).lazy
        # if train_dataset.lazy is not None:
        #     train_dataset = train_dataset.lazy

        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=collate_fn_partial,
            shuffle=False,
            **config["train_dataset"]["dataloader"],  # sampler=None,
        )

        # train_dataloader = accelerator.prepare(train_dataloader)
        # print(type(train_dataloader))

        # print("After prepare length:", len(train_dataset))

    if "train" in args.mode or "validate" in args.mode:
        # TODO: dev doch nicht lokal shufflen? ist doch quasi egal für dev?
        if "lazy" in config["validate_dataset"]["path"]:
            validate_dataset_config["acc"] = accelerator
            validate_dataset_config["num_workers"] = config["validate_dataset"][
                "dataloader"
            ]["num_workers"]
            validate_dataset_config["batch_size"] = (
                config["validate_dataset"]["dataloader"].get("batch_size", 16),
            )
            validate_dataset = instantiate(
                config["validate_dataset"]["path"], args=validate_dataset_config
            ).lazy

            validate_dataloader = DataLoader(
                dataset=validate_dataset,
                collate_fn=collate_fn_partial,
                shuffle=False,
                **config["validate_dataset"]["dataloader"],  # sampler=None,
            )
        else:
            validate_dataset = instantiate(
                config["validate_dataset"]["path"], args=validate_dataset_config
            )
            # validate_dataset= validate_dataset[200*16:]

            validate_dataloader = DataLoader(
                dataset=validate_dataset,
                collate_fn=_collate_fn_non_lazy_partial,
                shuffle=False,
                **config["validate_dataset"]["dataloader"],  # sampler=None,
            )
            validate_dataloader = accelerator.prepare(validate_dataloader)

        # # For debugging DER ov etc in validation step
        # validate_dataloader = list(itertools.islice(validate_dataloader, 5))
    #
    # if config["meta"]["precompute_gcc"]:
    #     from diarizen.spatial_features.precompute import precompute_gccs
    #     precompute_gccs(config)

    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        accelerator=accelerator,
        config=config,
        resume=resume,
        model=model,
        optimizer_small=optimizer_small,
        optimizer_big=optimizer_big,
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
