# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
import argparse
# import itertools
# import json
from functools import partial
from pathlib import Path
import pickle
# import numpy as np
import toml
# import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from dataset import _collate_fn as _collate_fn_non_lazy

from dataset_lazy import IterableWrapper, _collate_fn
# from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader

# from diarizen.ckpt_utils import average_ckpt
from diarizen.logger import init_logging_logger
# from diarizen.models.unix_enc.hubconf import unix_enc_base
from diarizen.utils import instantiate


def run(config, resume):


    import torch
    from diarizen.models.unix_enc.unix_enc_model import UnixEncModel, UnixEncConfig, UnixEncPretrainingConfig
    # load_unixenc = "/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/unixenc/fairseq_models/cfg1/checkpoints/checkpoint_last.converted.pt"
    # ckpt = torch.load(load_unixenc, map_location="cpu")
    # if isinstance(ckpt, dict):
    #     print(ckpt.keys())
    # unix_encoder
    load_unixenc = "/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/unixenc/fairseq_models/cfg1/checkpoints/checkpoint_last.converted.pt"
    ckpt = torch.load(load_unixenc)
    # model = unix_enc_base()
    model_cfg = ckpt["model_cfg"]
    # model_cfg["conv_feature_layers"] = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"

    # model_cfg = UnixEncConfig(**model_cfg)
    from dataclasses import fields
    valid_keys = {f.name for f in fields(UnixEncConfig)}
    model_cfg = {
        k: v
        for k, v in ckpt["model_cfg"].items()
        if k in valid_keys
    }
    model_cfg = UnixEncConfig(**model_cfg)

    valid_keys = {f.name for f in fields(UnixEncPretrainingConfig)}
    task_cfg = {
        k: v
        for k, v in ckpt["task_cfg"].items()
        if k in valid_keys
    }
    task_cfg = UnixEncPretrainingConfig(**task_cfg)
    unixenc = UnixEncModel(
        cfg=model_cfg,
        task_cfg=task_cfg,
        dictionaries=ckpt["dictionaries_symbols"]
    )

    # "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
    unixenc.load_state_dict(ckpt["model_weight"], strict=True)
    model = unixenc
    # assert False
    # exp_dir = "/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_gcc/exp/test"
    # combined_id_list = []
    # total_read_length = 0
    # total_read_length_set = 0
    #
    # for rank in [0, 1, 2, 3]:
    #     stats_path = Path(exp_dir) / f"id_list_stats{rank}.txt"
    #     id_list_path = Path(exp_dir) / f"id_list{rank}.pkl"
    #
    #     with stats_path.open("r", encoding="utf-8") as fh:
    #         for line in fh:
    #             if line.startswith("LISTE len(id_list):"):
    #                 total_read_length += int(line.split(":", 1)[1].strip())
    #             if line.startswith("SET"):
    #                 total_read_length_set += int(line.split(":", 1)[1].strip())
    #
    #
    #     with id_list_path.open("rb") as fh:
    #         combined_id_list.extend(pickle.load(fh))
    #
    # print(
    #     "COMBINED:", len(combined_id_list),
    #     "\n SUM_READ_LENGTHS:", total_read_length,
    #     "\n MATCH:", len(combined_id_list) == total_read_length,
    #     "\n SET SHOULD BE LONG:", total_read_length_set,
    #     "\n SET ist lang:", len(set(combined_id_list)),
    #     flush=True,
    # )
    # assert False
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
    # TODO: ACHTUNG! DEVICE SPECIFIC TRUE
    set_seed(config["meta"]["seed"])#, device_specific=True)

    # model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step =  (399, 0.025, 0.02) # model.get_rf_info
    # print("model_num_frames, model_rf_duration, model_rf_step = ",  model.get_rf_info)


    spk_count_loss = config["trainer"]["args"].get("spk_count_loss", False)
    only_waveform = config["trainer"]["args"].get("only_waveform", False)
    # pass model receptive field info to dataset
    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step
    train_dataset_config["spk_count_loss"] = spk_count_loss
    train_dataset_config["only_waveform"] = only_waveform

    validate_dataset_config = config["validate_dataset"]["args"]
    validate_dataset_config["model_num_frames"] = model_num_frames
    validate_dataset_config["model_rf_duration"] = model_rf_duration
    validate_dataset_config["model_rf_step"] = model_rf_step
    validate_dataset_config["spk_count_loss"] = spk_count_loss
    validate_dataset_config["only_waveform"] = only_waveform

    collate_fn_partial = partial(
        _collate_fn,
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
        gcpsd=config["meta"].get("gcpsd", False),
        only_waveform=only_waveform,
        debug=True,
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

    # TODO: TRainer weg schmeißen, acc komm checken ob trainer was macht und dann hier dataloader iterieren!
    from tqdm.auto import tqdm
    dataloader_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc="Training debug",
        dynamic_ncols=True,
        bar_format="{l_bar}{r_bar}",
        colour="green",
        disable=not accelerator.is_local_main_process,
        position=0,
        leave=True,
    )
    id_list = []
    for batch_idx, batch in enumerate(dataloader_bar):
        id_list.extend(batch["ids"])
        continue

    print(
        "LISTE:", len(id_list), "SET:", len(set(id_list)), flush=True
    )
    total = len(id_list)
    unique = len(set(id_list))
    exp_dir = "/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_gcc/exp/test"
    rank = accelerator.process_index
    stats_path = Path(exp_dir) / f"id_list_stats{rank}.txt"
    duplicates_path = Path(exp_dir) / f"id_list_duplicates{rank}.txt"
    id_list_path = Path(exp_dir) / f"id_list{rank}.pkl"
    duplicates_list_path = Path(exp_dir) / f"dublicates_list{rank}.pkl"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as fh:
        fh.write(f"LISTE len(id_list): {total}\n")
        fh.write(f"SET len(set(id_list)): {unique}\n")
    logger.info(f"Saved id list stats to {stats_path.as_posix()}")

    duplicates = [id for id in id_list if id_list.count(id) > 1]
    unique_duplicates = sorted(duplicates)
    # print("DUPLICATES:", unique_duplicates, "COUNT:", len(unique_duplicates), flush=True)
    with duplicates_path.open("w", encoding="utf-8") as fh:
        fh.write(f"unique_duplicates: {unique_duplicates}\n")
        fh.write(f"COUNT:, len(unique_duplicates): {len(unique_duplicates)}\n")

    with id_list_path.open("wb") as fh:
        pickle.dump(id_list, fh)
    logger.info(f"Saved id list to {id_list_path.as_posix()}")
    with duplicates_list_path.open("wb") as fh:
        pickle.dump(duplicates, fh)
    logger.info(f"Saved id list to {duplicates_list_path.as_posix()}")

    # trainer = instantiate(config["trainer"]["path"], initialize=False)(
    #     accelerator=accelerator,
    #     config=config,
    #     resume=resume,
    #     model=None,
    #     optimizer_small=None,
    #     optimizer_big=None,
    #     debug_data=True,
    # )

    # for flag in args.mode:
    #     if flag == "train":
    #         try:
    #             trainer.train(train_dataloader, validate_dataloader)
    #         except Exception as e:
    #             print(f"Training failed due to {e}.", flush=True)
    #             raise e
    #     elif flag == "validate":
    #         trainer.validate(validate_dataloader)
    #     else:
    #         raise ValueError(f"Unknown mode: {flag}.")


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
