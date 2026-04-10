# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import math
import sys
import time
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

import pandas as pd
import toml
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torchinfo import summary
import shutil
from tqdm.auto import tqdm
import torch.profiler
from diarizen.logger import TensorboardLogger
from diarizen.optimization import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from diarizen.trainer_utils import TrainerState
from diarizen.utils import prepare_empty_dir, print_env

from diarizen.noam_updater import get_rate

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        accelerator: Accelerator,
        config,
        resume,
        model,
        optimizer_small,
        optimizer_big,
        aux_loss=False,
        spk_count_loss=False,
    ):
        """Create an instance of BaseTrainer for training, validation, and fine-tuning."""
        self.config = config
        self.resume = resume

        # GPU
        self.accelerator = accelerator
        self.rank = accelerator.device
        self.device = accelerator.device  # alias of rank
        # Setup directories
        self._initialize_exp_dirs_and_paths(config)

        # Store source code files
        if self.accelerator.is_local_main_process:
            self.track_files()

        # Model
        self.model = model
        self.optimizer_small = optimizer_small
        self.optimizer_big = optimizer_big

        # Unwrap_Model
        self.unwrap_model = self.accelerator.unwrap_model(self.model)

        # Trainer.train args
        self.trainer_config = config["trainer"]["args"]
        self.num_spk = self.trainer_config.get("num_spk", False)
        self.num_spk_and_gcc = self.trainer_config.get("num_spk_and_gcc", False)
        self.spk_prob = self.trainer_config.get("spk_prob", False)
        self.ov = self.trainer_config.get("ov", False)
        self.weighted_loss = self.trainer_config.get("weighted_loss", False)
        self.focal_loss = self.trainer_config.get("focal_loss", False)
        self.ov_loss = self.trainer_config.get("ov_loss", False)
        self.spk_count_loss = self.trainer_config.get("spk_count_loss", False)
        self.only_waveforms = self.trainer_config.get("only_waveforms", False)
        self.bce_loss = self.trainer_config.get("bce_loss", False)
        self.aux_loss = self.trainer_config.get("aux_loss", False)
        self.debug = self.trainer_config.get("debug", False)
        self.compute_second_der = self.trainer_config.get("compute_second_der", True)

        self.max_steps = self.trainer_config.get("max_steps", 0)
        self.max_epochs = self.trainer_config.get("max_epochs", sys.maxsize)
        self.max_grad_norm = self.trainer_config.get("max_grad_norm", 0)
        self.save_max_score = self.trainer_config.get("save_max_score", True)
        self.save_ckpt_interval = self.trainer_config.get("save_ckpt_interval", 1)
        self.max_patience = self.trainer_config.get("max_patience", 10)
        self.plot_norm = self.trainer_config.get("plot_norm", True)
        self.plot_lr = self.trainer_config.get("plot_lr", False)
        self.validation_interval = self.trainer_config.get("validation_interval", 1)
        self.max_num_checkpoints = self.trainer_config.get("max_num_checkpoints", 50)
        self.scheduler_name = self.trainer_config.get(
            "scheduler_name", "linear_schedule_with_warmup"
        )
        self.warmup_steps = self.trainer_config.get("warmup_steps", 0)
        self.warmup_steps_enc = self.trainer_config.get("warmup_steps_enc", 0)
        self.preheat_epochs = self.trainer_config.get("preheat_epochs", 0)
        self.warmup_ratio = self.trainer_config.get("warmup_ratio", 0.0)
        self.gradient_accumulation_steps = self.trainer_config.get(
            "gradient_accumulation_steps", 1
        )

        self.validation_before_training = self.trainer_config.get(
            "validation_before_training", False
        )

        self.lr_decay = self.trainer_config.get("lr_decay", False)
        self.lr_decay_patience = self.trainer_config.get("lr_decay_patience", 2)

        self.use_one_cycle_lr = self.trainer_config.get("use_one_cycle_lr", False)

        self.gradient_percentile = self.trainer_config.get("gradient_percentile", 10)
        self.gradient_history_size = self.trainer_config.get(
            "gradient_history_size", 1000
        )

        # wavlm
        self.freeze_wavlm = self.trainer_config.get("freeze_wavlm", False)
        self.save_path = self.trainer_config.get("save_path", None)
        # wavlm
        if self.freeze_wavlm:
            logger.info("Freeze WavLM...")
            self.unwrap_model.freeze_by_name("wavlm_model")
            self.unwrap_model.freeze_by_name("wavlm")

        # Dataset
        self.dataset_config = config["train_dataset"]["args"]
        self.chunk_size = self.dataset_config.get("chunk_size", 500)

        # Finetune
        self.finetune_config = config["finetune"]
        self.finetune = self.finetune_config["finetune"]
        self.init_epochs = self.finetune_config.get("init_epochs", " ")
        self.ckpt_path = self.finetune_config.get("ckpt_path", " ")

        if self.max_steps > 0:
            logger.info(
                f"`max_steps` is set to {self.max_steps}. Ignoring `max_epochs`."
            )

        if self.validation_interval < 1:
            logger.info(
                f"`validation_interval` is set to {self.validation_interval}. It must be >= 1."
            )

        # Trainer states
        self.state = TrainerState(save_max_score=self.save_max_score)
        self.accelerator.register_for_checkpointing(
            self.state
        )  # Register accelerate objects

        # Others
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        if self.accelerator.is_local_main_process:
            prepare_empty_dir(
                [
                    self.save_dir,
                    self.exp_dir,
                    self.checkpoints_dir,
                    self.tb_log_dir,
                ],
                resume=resume,
            )

        self.writer = TensorboardLogger(self.tb_log_dir.as_posix())
        self.writer.log_config(config)

        # import wandb
        # wandb.init(project="diarizen", dir=self.tb_log_dir.as_posix())
        # wandb.config.update(config)
        # self.writer = wandb

        with open(self.config_path.as_posix(), "w") as handle:
            toml.dump(config, handle)

        logger.info(f"Configuration file is saved to {self.config_path.as_posix()}.")

        logger.info(f"Environment information:\n{print_env()}")

        # Model summary
        logger.info(f"\n {summary(self.model, verbose=0)}")

    def track_files(self):
        out_dir = self.source_code_backup_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_dirs = [
            "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen",
            "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_gcc",
            "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc",
        ]
        patterns = ["*.py", "*.toml", "*.sh"]
        for folder in save_dirs:
            for pattern in patterns:
                for file in Path(folder).glob(pattern):
                    # Erzeuge einen eindeutigen Zielpfad, der den relativen Pfad zum Quellordner enthält
                    rel_path = file.relative_to(folder)
                    last_folder = Path(folder).name
                    dest_path = out_dir / last_folder / file.name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file, dest_path)
        logger.info(f"Copied files to {out_dir}")
        return

    def _run_early_stop_check(self, score: float):
        should_stop = False

        if self._check_improvement(score, save_max_score=self.save_max_score):
            self.state.best_score = score
            self.state.best_score_epoch = self.state.epochs_trained
            self._save_checkpoint(self.state.epochs_trained, is_best_epoch=True)
            self.state.patience = 0
            logger.info(f"Found new best score: {score:.4f}, saving checkpoint...")
        else:
            logger.info(
                f"Score did not improve from {self.state.best_score:.4f} at epoch {self.state.best_score_epoch}."
            )
            self.state.patience += 1
            logger.info(
                f"Early stopping counter: {self.state.patience} out of {self.max_patience}"
            )

            if self.state.patience >= self.max_patience:
                logger.info("Early stopping triggered, stopping training...")
                should_stop = True

        return should_stop

    @staticmethod
    def _get_time_now():
        return time.strftime("%Y_%m_%d--%H_%M_%S")

    def _initialize_exp_dirs_and_paths(self, config):
        """Initialize directories.

        Args:
            save_dir: the root directory to save all experiments.
            exp_id: the experiment id.

        Notes:
            - save_dir: /home/xhao/exp
            - exp_dir: /home/xhao/exp/fullsubnet_lr_0.1
            - checkpoints_dir: /home/xhao/exp/fullsubnet_lr_0.1/checkpoints
            - tb_log_dir: /home/xhao/exp/fullsubnet_lr_0.1/tb_log
            - src_source_code_dir: /home/xhao/diarizen
            - source_code_backup_dir: /home/xhao/exp/fullsubnet_lr_0.1/source_code__2023_01_07__17_19_57
            - config_path: /home/xhao/exp/fullsubnet_lr_0.1/config__2023_01_07__17_19_57.toml
        """
        self.save_dir = Path(config["meta"]["save_dir"]).expanduser().absolute()
        self.exp_dir = self.save_dir / config["meta"]["exp_id"]
        self.summary_path = Path(self.exp_dir) / "val_metric_summary.lst"
        self.summary_path_date = (
            Path(self.exp_dir) / f"val_metric_summary__{self._get_time_now()}.lst"
        )

        # delete old val metric files if exist and not resume, when resume continue file
        if self.accelerator.is_local_main_process:
            if self.summary_path.exists() and not self.resume:
                self.summary_path.rename(
                    self.summary_path.parent
                    / f"old_val_metric_summary__{self._get_time_now()}.lst"
                )  # unlink()

        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.tb_log_dir = self.exp_dir / "tb_log"

        # Each run will have a unique source code, config, and log file.
        time_now = self._get_time_now()
        self.source_code_dir = (
            Path(__file__).expanduser().absolute().parent.parent.parent
        )
        self.source_code_backup_dir = (
            self.exp_dir / "source_code" / f"source_code__{time_now}"
        )
        self.config_path = self.exp_dir / f"config__{time_now}.toml"

    def _find_latest_ckpt_path(self):
        """Find the latest checkpoint path."""
        # Pick up all checkpoints with the format `epoch_*`
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_" + ("[0-9]" * 4)))

        # Remove files that is not a checkpoint
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.is_dir()]

        if len(checkpoints) == 0:
            raise FileNotFoundError(
                f"No checkpoints found in {self.checkpoints_dir.as_posix()}."
            )

        # Pick up the latest checkpoint
        ckpt_path = checkpoints[-1]

        return ckpt_path

    def _load_checkpoint(self, ckpt_path):
        """load a checkpoint from the checkpints directory.

        Args:
            ckpt_path: "best", "latest", or a path to a checkpoint file
        """
        if ckpt_path == "best":
            ckpt_path = self.checkpoints_dir / "best"
        elif ckpt_path == "latest":
            ckpt_path = self._find_latest_ckpt_path()
        else:
            ckpt_path = Path(ckpt_path).expanduser().absolute()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path.as_posix()} not found.")

        self.accelerator.load_state(ckpt_path, map_location="cpu")

        # if ckpt_path.name.endswith("0000"):
        #     self.state.epochs_trained = 0
        logger.info(f"Checkpoint on epoch {self.state.epochs_trained} is loaded.")

    def _save_checkpoint(self, epoch, is_best_epoch):
        """Save checkpoint.

        Args:
            epoch: the current epoch.
            is_best_epoch: whether the current epoch is the best epoch.
        """
        # Save checkpoint
        if is_best_epoch:
            self.accelerator.save_state(
                self.checkpoints_dir / "best", safe_serialization=False
            )
        else:
            # Regular checkpoint
            ckpt_path = self.checkpoints_dir / f"epoch_{str(epoch).zfill(4)}"
            self.accelerator.save_state(ckpt_path.as_posix(), safe_serialization=False)

        # Find all regular checkpoints and only keep the latest `max_num_checkpoints` regular checkpoints
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*"))

        if epoch <= len(checkpoints):
            logger.warning(
                f"Current epoch is {epoch}, but found {len(checkpoints)} checkpoints. "
                f"This may be caused by you running the same experiment multiple times. "
                f"Recommend to run the experiment with a different `exp_id`."
            )

        if len(checkpoints) > self.max_num_checkpoints:
            logger.info(
                f"Found {len(checkpoints)} checkpoints, only keeping the latest {self.max_num_checkpoints} checkpoints."
            )
            for checkpoint_dir in checkpoints[: -self.max_num_checkpoints]:
                shutil.rmtree(checkpoint_dir.as_posix())
                logger.info(f"Checkpoint {checkpoint_dir.as_posix()} is removed.")

    @staticmethod
    def get_warmup_steps(warmup_steps, max_steps, warmup_ratio):
        if warmup_steps > 0:
            logger.info(f"warmup_steps={warmup_steps}. warmup_ratio will be ignored.")
            return warmup_steps
        else:
            return math.ceil(max_steps * warmup_ratio)

    def create_warmup_scheduler(self, optimizer, scheduler_name, max_steps: int):
        num_warmup_steps = self.get_warmup_steps(
            self.warmup_steps, max_steps, self.warmup_ratio
        )
        if scheduler_name == "constant_schedule_with_warmup":
            return get_constant_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps
            )
        elif scheduler_name == "linear_schedule_with_warmup":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_steps,
            )

    def create_warmup_scheduler_pretraining(
        self, optimizer, scheduler_name, max_steps: int
    ):
        num_warmup_steps = self.get_warmup_steps(
            self.warmup_steps_enc, max_steps, self.warmup_ratio
        )
        if scheduler_name == "constant_schedule_with_warmup":
            return get_constant_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps
            )
        elif scheduler_name == "linear_schedule_with_warmup":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_steps,
            )

    def create_warmup_scheduler_offset(
        self, optimizer, scheduler_name, max_steps, warumup_steps, step_offset=0
    ):
        num_warmup_steps = self.get_warmup_steps(
            warumup_steps, max_steps, self.warmup_ratio
        )

        def lr_lambda(current_step):
            adjusted_step = current_step - step_offset
            if adjusted_step < 0:
                return 0.0
            elif adjusted_step < num_warmup_steps:
                return float(adjusted_step) / float(max(1, num_warmup_steps))
            return 1.0  #  max(0.0, float(max_steps - current_step) / float(max(1, max_steps - num_warmup_steps)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def create_schedulers_pretraining(
        self, max_steps: int, max_steps_preheat, step_offset
    ):
        """Create schedulers.

        You can override this method to create your own schedulers. For example, in GAN training, you may want to
        create two schedulers for the generator and the discriminator.

        Args:
            max_steps: the maximum number of steps to train.
        """
        self.lr_scheduler_small = self.create_warmup_scheduler(
            optimizer=self.optimizer_small,
            scheduler_name=self.scheduler_name,
            max_steps=max_steps,
        )
        self.lr_scheduler_big = self.create_warmup_scheduler_offset(
            optimizer=self.optimizer_big,
            scheduler_name=self.scheduler_name,
            warumup_steps=self.warmup_steps,
            max_steps=max_steps,
            step_offset=step_offset,
        )
        # self.lr_scheduler_big = self.create_warmup_scheduler(
        #     optimizer=self.optimizer_big, scheduler_name=self.scheduler_name, max_steps=max_steps
        # )
        self.lr_scheduler_big_pretrain = self.create_warmup_scheduler_offset(
            optimizer=self.optimizer_big,
            scheduler_name=self.scheduler_name,
            warumup_steps=self.warmup_steps_enc,
            max_steps=max_steps_preheat,
            step_offset=0,
        )
        # self.lr_scheduler_big_pretrain = self.create_warmup_scheduler_pretraining(
        #     optimizer=self.optimizer_big, scheduler_name=self.scheduler_name, max_steps=max_steps
        # )
        # todo: encoder_warmup_steps, warm up steps setzen, prepare machen , und in loop passend wählen
        (
            self.lr_scheduler_small,
            self.lr_scheduler_big,
            self.lr_scheduler_big_pretrain,
        ) = self.accelerator.prepare(
            self.lr_scheduler_small,
            self.lr_scheduler_big,
            self.lr_scheduler_big_pretrain,
        )

    def create_schedulers(self, max_steps: int):
        """Create schedulers.

        You can override this method to create your own schedulers. For example, in GAN training, you may want to
        create two schedulers for the generator and the discriminator.

        Args:
            max_steps: the maximum number of steps to train.
        """
        self.lr_scheduler_small = self.create_warmup_scheduler(
            optimizer=self.optimizer_small,
            scheduler_name=self.scheduler_name,
            max_steps=max_steps,
        )
        self.lr_scheduler_big = self.create_warmup_scheduler(
            optimizer=self.optimizer_big,
            scheduler_name=self.scheduler_name,
            max_steps=max_steps,
        )
        self.lr_scheduler_small, self.lr_scheduler_big = self.accelerator.prepare(
            self.lr_scheduler_small, self.lr_scheduler_big
        )

    def set_models_to_train_mode(self):
        """Set models to train mode.

        You can override this method to set your own models to train mode. For example, in GAN training, you may want to
        set the generator and the discriminator to train mode.
        """
        self.model.train()

    def set_models_to_eval_mode(self):
        self.model.eval()

    def get_optimizer_lr(self, optimizer):
        return optimizer.state_dict()["param_groups"][0]["lr"]

    def lr_scheduler_step(self):
        """Step the lr scheduler.

        You can override this method to step your own lr scheduler. For example, in GAN training, you may want to
        step the lr scheduler of the generator and the discriminator.
        """
        self.lr_scheduler_small.step(self.state.steps_trained)
        self.lr_scheduler_big.step(self.state.steps_trained)

    def create_lr_decay_scheduler(self):
        self.lr_decay_scheduler_small = ReduceLROnPlateau(
            optimizer=self.optimizer_small,
            mode="min",
            factor=0.95,
            patience=self.lr_decay_patience,
            min_lr=1e-8,
        )
        self.lr_decay_scheduler_big = ReduceLROnPlateau(
            optimizer=self.optimizer_big,
            mode="min",
            factor=0.95,
            patience=self.lr_decay_patience,
            min_lr=1e-8,
        )
        self.lr_decay_scheduler_small, self.lr_decay_scheduler_big = (
            self.accelerator.prepare(
                self.lr_decay_scheduler_small, self.lr_decay_scheduler_big
            )
        )

    def create_lr_one_cycle_scheduler(self, max_steps):
        self.lr_one_cycle_scheduler_small = OneCycleLR(
            optimizer=self.optimizer_small,
            max_lr=self.get_optimizer_lr(self.optimizer_small),
            total_steps=max_steps,
        )
        self.lr_one_cycle_scheduler_big = OneCycleLR(
            optimizer=self.optimizer_big,
            max_lr=self.get_optimizer_lr(self.optimizer_big),
            total_steps=max_steps,
        )
        self.lr_one_cycle_scheduler_small, self.lr_one_cycle_scheduler_big = (
            self.accelerator.prepare(
                self.lr_one_cycle_scheduler_small, self.lr_one_cycle_scheduler_big
            )
        )

    def create_bar_desc(self, loss_dict, norm):
        bar_desc = ""
        for k, v in loss_dict.items():
            bar_desc += f"{k}: {(v):.4f}, "
        bar_desc += (
            f"norm: {norm:.4f}, " f"lr: {self.lr_scheduler.get_last_lr()[-1]:.10f}"
        )

        # plot norm
        if self.plot_norm:
            self.writer.add_scalar("Train_Step/norm", norm, self.state.steps_trained)

        if self.plot_lr:
            self.writer.add_scalar(
                "Train_Step/lr",
                self.lr_scheduler.get_last_lr()[-1],
                self.state.steps_trained,
            )

        return bar_desc

    def freeze_all_except_encoder(self):
        # Falls DDP benutzt wird, greife auf das Originalmodell zu
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        for p in model_ref.parameters():
            p.requires_grad = False
        for p in model_ref.gcc_encoder.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        for p in model_ref.parameters():
            p.requires_grad = True

    def train(self, train_dataloader: DataLoader, validation_dataloader):
        """Train loop entry point.

        Args:
            train_dataloader: the dataloader to train.
            validation_dataloades: the dataloader(s) to validate.

        Notes:
            You are responsible for calling ``.backward()``, ``.step()``, and ``.zero_grad()`` in your implementation
            of `training_step()`. Accelerate will automatically handle the gradient accumulation for you.
            It means that in gradient accumulation, the step() of optimizer and scheduler is called only when gradient_accumulation_steps is reached.

            The training step is implemented as follows:

            .. code-block:: python

                    self.optimizer.zero_grad()
                    loss = training_step(batch, batch_idx)
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                    return {
                        "loss": loss,
                    }
        """
        early_stop_mark = torch.zeros(1, device=self.device)

        # Setting up training control variables
        steps_per_epoch = len(train_dataloader)
        # steps_per_epoch = int(train_dataloader.dataset.get_my_length / train_dataloader.batch_size +1) # for sharded dataloader
        # print("STEPS PER EPOCH:", steps_per_epoch, flush=True)
        update_steps_per_epoch = steps_per_epoch // self.gradient_accumulation_steps
        update_steps_per_epoch = max(update_steps_per_epoch, 1)

        if self.max_steps > 0:
            max_steps = self.max_steps
            max_epochs = self.max_steps // update_steps_per_epoch + int(
                self.max_steps % update_steps_per_epoch > 0
            )
        else:
            max_steps = self.max_epochs * update_steps_per_epoch
            max_epochs = self.max_epochs

        logger.info("Training control variables:")
        logger.info(f"`steps_per_epoch`: {steps_per_epoch}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"`update_steps_per_epoch`: {update_steps_per_epoch}")
        logger.info(f"`max_steps`: {max_steps}")
        logger.info(f"`max_epochs`: {max_epochs}")

        # Generator learning rate scheduler
        if self.warmup_steps > 0:
            if self.preheat_epochs == 0:
                self.create_schedulers(max_steps=max_steps)
            else:
                step_offset = self.preheat_epochs * steps_per_epoch
                self.create_schedulers_pretraining(
                    max_steps=max_steps,
                    max_steps_preheat=step_offset,
                    step_offset=step_offset,
                )

        if self.use_one_cycle_lr:
            self.create_lr_one_cycle_scheduler(
                max_steps=max_steps * self.accelerator.num_processes
            )

        # Resume
        if self.resume:
            self._load_checkpoint(ckpt_path="latest")

        # # validation 0 epoch performance
        if self.validation_before_training:
            with torch.no_grad():
                logger.info("Validation on ZERO epoch...")
                score = self.validate(validation_dataloader)
                # score = self.validate(train_dataloader)
                # print("VALIDATION DONE", self.accelerator.process_index, flush=True)

            if self.accelerator.is_local_main_process:
                self._save_checkpoint(epoch=0, is_best_epoch=False)

        for epoch in range(self.state.epochs_trained + 1, max_epochs + 1):
            # print(f"epoch {epoch}",self.accelerator.process_index, flush=True)
            logger.info(f"{'=' * 9} Epoch {epoch} out of {max_epochs} {'=' * 9}")
            logger.info("Begin training...")

            self.num_correct = 0
            self.num_total = 0

            self.num_correct_ov = 0
            self.num_total_ov = 0

            self.set_models_to_train_mode()

            # if self.unwrap_model.wavlm_frozen:
            if hasattr(
                self.unwrap_model, "wavlm_frozen"
            ):  # and self.unwrap_model.wavlm_frozen:
                try:
                    self.unwrap_model.wavlm.eval()
                except AttributeError:
                    self.unwrap_model.wavlm_model.eval()

            if self.freeze_wavlm:
                self.unwrap_model.wavlm_model.eval()
            if epoch <= self.preheat_epochs:
                logger.info(">> Pretraining encoder only")
                self.freeze_all_except_encoder()
            else:
                logger.info(">> Full model training")
                self.unfreeze_all()

            training_epoch_output = []
            # print(f"waiting for all",self.accelerator.process_index, flush=True)
            # self.accelerator.wait_for_everyone()
            # print("all here", self.accelerator.process_index, flush=True)

            # # the iter number of progress bar increments by 1 by default whether gradient accumulation is used or not.
            # # but we update the description of the progress bar only when the gradients are synchronized across all processes.
            # # train_dataloader.__len__ = update_steps_per_epoch
            # # gather len von sharded dataloaders
            # print("lens simple:", len(train_dataloader), flush=True)
            # lens = self.accelerator.gather(torch.tensor(len(train_dataloader), device=self.device))
            # lengths_list = lens.cpu().tolist()  # Liste mit Längen pro Prozess
            # print("Lengths list:", lengths_list, flush=True)

            dataloader_bar = tqdm(
                train_dataloader,
                total=update_steps_per_epoch,
                desc="",
                dynamic_ncols=True,
                bar_format="{l_bar}{r_bar}",
                colour="green",
                disable=not self.accelerator.is_local_main_process,
                position=0,
                leave=True,
            )
            # import time
            self.id_list = []
            for batch_idx, batch in enumerate(dataloader_bar):
                # print(f"Training batch {batch_idx}", self.accelerator.process_index, flush=True)
                # print(f"Batch {batch_idx} IDs:", batch['ids'], flush=True)
                self.id_list.extend(batch["ids"])

                # print("LISTE:", self.id_list, "\n SET:", set(self.id_list), flush=True)
                # print("LISTE:", len(self.id_list), "SET:", len(set(self.id_list)), flush=True)
                for k, v in batch.items():
                    if torch.is_tensor(v) and v.device != self.accelerator.device:
                        batch[k] = v.to(self.accelerator.device)
                # if batch_idx == 0:
                #     print("DATALOADER SHARDED:", train_dataloader.__class__)
                # if self.debug:
                #     # CUDA warmup
                #     for _ in range(5):
                #         _ = self.training_step(batch, batch_idx)
                #
                #     torch.cuda.synchronize()
                #     start = time.time()
                #
                #     for _ in range(6):  # 10 Iterationen mitteln
                #         out = self.training_step(batch, batch_idx)
                #         torch.cuda.synchronize()
                #     end = time.time()
                #     print(f"Average iteration time: {(end - start) / 6:.4f} sec")
                #     if batch_idx >= 5:
                #         assert False
                #     continue

                # t0 = time.time()
                # data_loading_time = t0 - start if batch_idx > 0 else 0
                # t1 = time.time()
                # accumulate() will automatically skip synchronization if applicable loss is linearly scaled with the optimizer.grad
                # accumulate() will automatically divide the loss in backward by the number of gradient accumulation steps
                # However, it won't return this loss, so we need to manually divide the loss by the number of gradient accumulation steps.
                with self.accelerator.accumulate(self.model):
                    # You are responsible for calling `.backward()`, `.step()`, and `.zero_grad()` in your implementation
                    loss_dict = self.training_step(batch, batch_idx)
                    training_epoch_output.append(loss_dict)
                    # forward_time = time.time() - t1
                    if not self.accelerator.optimizer_step_was_skipped:
                        if self.warmup_steps > 0:

                            if self.state.epochs_trained < self.preheat_epochs:
                                # logger.info(">> Preheating encoder only, LEARNING RATE PREHEAT")
                                self.lr_scheduler_small.step(self.state.steps_trained)
                                self.lr_scheduler_big_pretrain.step(
                                    self.state.steps_trained
                                )
                            else:
                                # logger.info(">> WARMUP full model, learning rate should go up over 3000 steps")
                                self.lr_scheduler_small.step(self.state.steps_trained)
                                self.lr_scheduler_big.step(self.state.steps_trained)

                        if self.use_one_cycle_lr:
                            self.lr_one_cycle_scheduler_small.step()
                            self.lr_one_cycle_scheduler_big.step()
                    if batch_idx % 300 == 0 and batch_idx != 0:
                        self.accuracy = (
                            self.num_correct / self.num_total
                            if self.num_total > 0
                            else 0.0
                        )
                        self.accuracy_ov = (
                            self.num_correct_ov / self.num_total_ov
                            if self.num_total_ov > 0
                            else 0.0
                        )
                        self.training_epoch_end(training_epoch_output, stepwise=True)

                self.state.steps_trained += 1

                # print(f"Batch {batch_idx}: DataLoad {data_loading_time:.3f}s | Forward {forward_time:.3f}s")
                # start = time.time()

            # print("LISTE:", sorted(self.id_list), "\n SET:", sorted(set(self.id_list)), flush=True)
            print(
                "LISTE:", len(self.id_list), "SET:", len(set(self.id_list)), flush=True
            )

            # duplicates = [id for id in self.id_list if self.id_list.count(id) > 1]
            # unique_duplicates = sorted(duplicates)
            # print("DUPLICATES:", unique_duplicates, "COUNT:", len(unique_duplicates), flush=True)

            self.state.epochs_trained += 1
            self.accuracy = (
                self.num_correct / self.num_total if self.num_total > 0 else 0.0
            )
            print(f"Accuracy: {self.accuracy:.4f}")
            self.accuracy_ov = (
                self.num_correct_ov / self.num_total_ov
                if self.num_total_ov > 0
                else 0.0
            )
            print(f"Accuracy: {self.accuracy_ov:.4f}")
            self.training_epoch_end(training_epoch_output)

            # Should save, evaluate, and early stop?
            if (
                self.accelerator.is_local_main_process
                and epoch % self.save_ckpt_interval == 0
            ):
                self._save_checkpoint(epoch, is_best_epoch=False)

            if epoch % self.validation_interval == 0:
                with torch.no_grad():
                    logger.info("Training finished, begin validation...")

                    self.num_correct = 0
                    self.num_total = 0
                    self.num_correct_ov = 0
                    self.num_total_ov = 0

                    score = self.validate(validation_dataloader)

                    if self.accelerator.is_local_main_process:

                        if self.lr_decay:
                            self.lr_decay_scheduler_small.step(score)
                            self.lr_decay_scheduler_big.step(score)

                        should_stop = self._run_early_stop_check(score)
                        if should_stop:
                            early_stop_mark += 1

                    logger.info("Validation finished.")

            self.accelerator.wait_for_everyone()

            # Reduces the `early_stop_mark` data across all processes
            # If `early_stop_mark` is 1 in any process, then `reduce_early_stop_mark` will be 1 in all processes.
            reduced_early_stop_mark = self.accelerator.reduce(
                early_stop_mark, reduction="sum"
            )
            if self.save_path is not None:
                assert False, "End training since features have been saved"

            # If any process triggers early stopping, stop training
            if reduced_early_stop_mark != 0:
                break

    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model.

        Args:
            dataloaders: the dataloader(s) to validate.

        Returns:
            score: the metric score of the validation epoch.
        """
        logger.info("Begin validation...")

        self.set_models_to_eval_mode()

        validation_output = []
        import time

        self.all_preds = []
        self.all_targets = []
        # steps_per_epoch = int(len(dataloader.dataset) / dataloader.batch_size +1) # for sharded dataloader
        # print(f"rank {self.accelerator.process_index}:", steps_per_epoch)
        steps_per_epoch = int(len(dataloader))  # for sharded dataloader
        # print(f"rank {self.accelerator.process_index}:", steps_per_epoch)
        end = None
        # print(f"num batches for rank {self.accelerator.process_index}:", len(dataloader), flush=True)
        # TODO : einfach mal drüber iterieren und gucken wann die iteratoren leer sind, len() z#ählt nämlich nicht!!!
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                total=steps_per_epoch,
                desc="",
                bar_format="{l_bar}{r_bar}",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
            )
        ):
            # print(f"Validating batch {batch_idx}", self.accelerator.process_index, time.time()-start, flush=True)
            # continue
            for k, v in batch.items():
                if torch.is_tensor(v) and v.device != self.accelerator.device:
                    batch[k] = v.to(self.accelerator.device)
            # We recommend you directly calculate the metric score in the validation_step function and return the
            # metric score in the validation_step function, and then calculate the mean of the metric score
            # in the validation_epoch_end function.
            # if end is not None:
            #     print(f"Validation time: {end -start:.3f} sec", flush=True)
            # start = time.time()
            # if end is not None:
            #     print(f"Data loading time: {start - end:.3f} sec", flush=True)
            step_output = self.validation_step(batch, batch_idx)
            # print(self.rank, "first stop", flush=True)
            # mid = time.time()

            """
            {
                "metric_1": metric_1_score,
                "metric_2": metric_1_score,
                ...
            }
            """
            step_output_cpu = {}
            step_output_device = {}

            for k, v in step_output.items():
                if torch.is_tensor(v):
                    step_output_cpu[k] = v.cpu().item()  # CPU + float für gather_object
                    step_output_device[k] = v.to(
                        self.accelerator.device
                    )  # GPU/Device Tensor
                else:
                    step_output_cpu[k] = v  # Float bleibt Float
                    step_output_device[k] = torch.tensor(
                        v, device=self.accelerator.device, dtype=torch.float
                    )

            # print(step_output_cpu.items())
            gathered_step_output = self.accelerator.gather_for_metrics(
                step_output_device
            )  # , use_gather_object=True)
            # gathered_step_output = self.accelerator.gather(step_output_device)  # , use_gather_object=True)

            # print(type(gathered_step_output))
            # print(gathered_step_output)
            validation_output.append(gathered_step_output)
            # end = time.time()

            # step_output_device = {
            #     k: (v.to(self.accelerator.device).contiguous() if torch.is_tensor(v) else v)
            #     for k, v in step_output.items()
            # }
            # gathered_step_output = self.accelerator.gather_for_metrics(step_output_device)
            # validation_output.append(gathered_step_output)

            # gathered_step_output = self.accelerator.gather_for_metrics(step_output)
            # validation_output.append(gathered_step_output)

            # break

        logger.info("Validation inference finished, begin validation epoch end...")

        if hasattr(self, "num_total"):
            self.accuracy = (
                self.num_correct / self.num_total if self.num_total > 0 else 0.0
            )
            self.accuracy_ov = (
                self.num_correct_ov / self.num_total_ov
                if self.num_total_ov > 0
                else 0.0
            )
            print(f"Accuracy: {self.accuracy:.4f}")
        if len(self.all_preds) > 0:
            self.ex_pred = self.all_preds[0][8:12].cpu().numpy()
            self.ex_target = self.all_targets[0][8:12].cpu().numpy()

            self.all_preds = torch.cat(self.all_preds).numpy().reshape(-1)
            self.all_targets = torch.cat(self.all_targets).numpy().reshape(-1)
            self.num_classes = max(self.all_targets.max(), self.all_preds.max()) + 1
            self.cm = confusion_matrix(
                self.all_targets, self.all_preds, labels=list(range(self.num_classes))
            )
            # F1 Score

            self.f1_macro = f1_score(self.all_targets, self.all_preds, average="macro")
            self.f1_weighted = f1_score(
                self.all_targets, self.all_preds, average="weighted"
            )
            self.f1_per_class = f1_score(self.all_targets, self.all_preds, average=None)

            self.precision_macro = precision_score(
                self.all_targets, self.all_preds, average="macro"
            )
            self.precision_per_class = precision_score(
                self.all_targets, self.all_preds, average=None
            )

            self.recall_macro = recall_score(
                self.all_targets, self.all_preds, average="macro"
            )
            self.recall_per_class = recall_score(
                self.all_targets, self.all_preds, average=None
            )

        # print(f"Example predictions: {self.accelerator.process_index}")
        # print(self.rank, "2 stop", flush=True)
        self.accelerator.wait_for_everyone()
        # torch.distributed.barrier()
        # print(f"waited in validate: {self.accelerator.process_index}")

        # assert False
        # TODO: prefetch race?? hmm sund auf jeden fall ungleich lang nach test durch iterieren
        # TODO: mal ohne metriken probieren, zeiten printen
        if self.accelerator.is_local_main_process:
            # only the main process will run validation_epoch_end
            score = self.validation_epoch_end(validation_output)
        else:
            score = None

        # torch.distributed.barrier()
        # print(f"SCORE BERECHNET: {self.accelerator.process_index}")
        return score

    def _check_improvement(self, score, save_max_score=True):
        """Check if the current model got the best metric score"""
        if save_max_score:
            if score > self.state.best_score:
                return True
            else:
                return False
        else:
            if score < self.state.best_score:
                return True
            else:
                return False

    def training_step(self, batch, batch_idx):
        """Implement a training step.

        Implement your own training step here.
        The input batch is from a training dataloader and the output of this function should be a loss tensor.
        Here is the persuade code for training a model:

        .. code-block:: python
            :emphasize-lines: 7

            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    zero_grad()
                    loss = training_step(batch, batch_idx)
                    loss.backward()
                    optimizer.step()

                training_epoch_output.append(loss)
                training_epoch_end(training_epoch_output)

                save_checkpoint()

                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)


        Args:
            batch: a batch of data, which passed from a custom training dataloader.
            batch_idx: the index of the current batch.

        Returns:
            loss: the loss of the batch.
        """
        raise NotImplementedError

    def training_epoch_end(self, training_epoch_output, stepwise=False):
        """Implement the logic of the end of a training epoch. Please override this function if you want to do something.

        When the training epoch ends, this function will be called. The input is a list of the loss dict of each step
        in a training epoch. You may want to log the epoch-level training loss here.

        .. code-block:: python
            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    loss = training_step(batch, batch_idx)
                    training_epoch_output.append(loss)

                training_epoch_end(training_epoch_output)

                save_checkpoint()

                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)

        Args:
            training_epoch_output: the output of the training epoch. It may a list of the output of each batch.
        """
        loss_keys = training_epoch_output[0].keys()

        # Compute mean loss on all loss items on a epoch
        for key in loss_keys:
            loss_items = [step_out[key] for step_out in training_epoch_output]
            loss_mean = torch.mean(torch.tensor(loss_items))

            if self.accelerator.is_local_main_process:
                if stepwise:
                    logger.info(
                        f"Training Loss '{key}' on Train_Step {self.state.steps_trained}: {loss_mean}"
                    )
                    self.writer.add_scalar(
                        f"Train_Step/{key}", loss_mean, self.state.steps_trained
                    )
                    self.writer.add_scalar(
                        f"Train_Step/accuracy", self.accuracy, self.state.steps_trained
                    )
                    self.writer.add_scalar(
                        f"Train_Step/accuracy_OV",
                        self.accuracy_ov,
                        self.state.steps_trained,
                    )
                else:
                    logger.info(
                        f"Training Loss '{key}' on epoch {self.state.epochs_trained}: {loss_mean}"
                    )
                    self.writer.add_scalar(
                        f"Train_Epoch/{key}", loss_mean, self.state.epochs_trained
                    )
                    self.writer.add_scalar(
                        f"Train_Epoch/accuracy",
                        self.accuracy,
                        self.state.epochs_trained,
                    )
                    self.writer.add_scalar(
                        f"Train_Epoch/accuracy_OV",
                        self.accuracy_ov,
                        self.state.epochs_trained,
                    )
                    self.writer.add_scalar(
                        f"Train_Epoch/lr_small",
                        get_rate(self.optimizer_small),
                        self.state.epochs_trained,
                    )
                    self.writer.add_scalar(
                        f"Train_Epoch/lr_big",
                        get_rate(self.optimizer_big),
                        self.state.epochs_trained,
                    )

                    # self.writer.add_scalar(f"Train_metrics/F1_macro", self.f1_macro, self.state.epochs_trained)
                    # self.writer.add_scalar(f"Train_metrics/F1_weighted", self.f1_weighted,
                    #                        self.state.epochs_trained)
                    # for i, f1 in enumerate(self.f1_per_class):
                    #     print(f"F1 score for class {i}: {f1:.3f}", flush=True)
                    #     self.writer.add_scalar(f"Train_metrics/f1_class_{i}", f1, self.state.epochs_trained)

    def validation_step(self, batch, batch_idx):
        """Implement a validation step for validating a model on all processes.

        This function defines the validation step. The input batch is from a validation dataloader.
        Here is the persuade code for validating a model:

        .. code-block:: python
            :emphasize-lines: 4

            validation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = validation_step(batch, batch_idx)
                    validation_epoch_output.append(loss_or_data)

            score = validation_epoch_end(validation_epoch_output)
            return score

        Notes:
            **The validation step will be run on all processes.**

            About batch size:
            If your validation data have the same length, you may use a batch size larger than 1 to speed up the validation.
            For example, if you have 1000 samples in the validation set, and you have a batch size of 100, then you will
            have 10 batches in the validation set. However, if your data in the validation set has a different length, please
            use a batch size of 1. It still works for distributed validation. Otherwise, you will get an error.

            About distributed validation:
            The output of this function will be gathered across all processes. For example, if you have 4 processes, and
            you have a batch size of 1, then you will have 4 outputs from this function. The output of this function will
            be gathered across all processes. The first dimension of the result is num_processes multiplied by the first
            dimension of the input tensors. **Please make sure the first dimension of the input tensors is the batch size.**
            **The last dimension of the output will be padded to the length of the longest sample in the validation set.**
            It means that the output will be a tensor with the shape of [num_processes * batch_size, max_length]. If you
            calculate the metric score on the output, you should do a truncation to remove the padding. Otherwise, if you
            are using a metric that sensitive to the padding, you will get a wrong metric score. It is not easy to
            implement this truncation in the ``validation_epoch_end`` function. We recommend you directly calculate the metric
            score in the validation_step function. I guess the Accelerate team will implement a automatic truncation in the
            future. https://github.com/huggingface/accelerate/issues/226

        Args:
            batch: a batch of data.
            batch_idx: the index of the batch.
            dataloader_idx: the index of the dataloader.

        Returns:
            output: the output of the batch. It may enhanced audio signals.
        """
        raise NotImplementedError

    def validation_epoch_end(self, validation_epoch_output):
        """Validation epoch end.

        The input `validation_epoch_output` will be a list of list. For example, if you have two dataloaders, the `validation_epoch_output` will be:

        .. code-block:: python

            validation_epoch_output = [
                [dataloader_1_batch_1_output, dataloader_1_batch_2_output, ...],
                [dataloader_2_batch_1_output, dataloader_2_batch_2_output, ...],
                ...,
            ]


        The output of this function should be a metric score, which will be used to determine whether the current model is the best model.

        .. code-block:: python
            :emphasize-lines: 7

            validation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = validation_step(batch, batch_idx)
                    validation_epoch_output.append(loss_or_data)

            score = validation_epoch_end(validation_epoch_output)
            return score

        Args:
            validation_epoch_output: the output of the validation epoch. It is a list of list.

        Returns:
            score: the metric score of the validation epoch.
        """
        raise NotImplementedError
