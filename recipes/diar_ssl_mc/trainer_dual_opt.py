# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
import os
from pathlib import Path

import torch
import torch.profiler
import numpy as np
import seaborn as sns
from accelerate.logging import get_logger
from matplotlib import pyplot as plt

from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.loss import nll_loss

from diarizen.trainer_dual_opt import Trainer as BaseTrainer
from recipes.diar_gcc.trainer_dual_opt import get_der_ov, pit_bce_loss

logger = get_logger(__name__)

import numpy as np


def labels_to_rttm(labels, frame_shift=320, sample_rate=16000, session_ids=None, empty=None):
    """
    Convert (B, T, S) frame-level labels into RTTM format strings.

    Args:
        labels: np.ndarray of shape (B, T, S), binary {0,1}
        frame_shift: int, number of samples per frame
        sample_rate: int, e.g. 16000
        session_ids: list of str, optional names for each batch/session

    Returns:
        rttm_lines: list of str
    """
    B, T, S = labels.shape
    if session_ids is None:
        session_ids = [f"utt_{i}" for i in range(B)]

    rttm_lines = []
    skip_sess = []
    for b in range(B):
        session = session_ids[b]
        if empty is not None and session in empty:
            continue
        l = len(rttm_lines)
        for s in range(S):
            active = labels[b, :, s]

            changes = np.diff(np.pad(active, (1, 1)))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            # nicht für jeden sprecher !!! irgendeiner wird leer sien liste voll
            for start_f, end_f in zip(starts, ends):
                start_t = (start_f * frame_shift) / sample_rate
                dur = ((end_f - start_f) * frame_shift) / sample_rate
                # RTTM
                line = f"SPEAKER {session} 1 {start_t:.3f} {dur:.3f} <NA> <NA> {s} <NA> <NA>"
                rttm_lines.append(line)
        if len(rttm_lines) == l:
            skip_sess.append(session)
    return rttm_lines, skip_sess

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator.print(self.model)
        self.aux_weight = 0.05  # 1e-2 bis 1e-1
        self.num_correct = 0
        self.num_total = 0
        self.num_correct_ov = 0
        self.num_total_ov = 0
        self.frame_deltas = []
        # auto GN
        self.grad_history = []

    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def auto_clip_grad_norm_(self, model):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        if len(self.grad_history) > self.gradient_history_size:
            self.grad_history.pop(0)
        clip_value = np.percentile(self.grad_history, self.gradient_percentile)
        self.accelerator.clip_grad_norm_(model.parameters(), clip_value)  

    def training_step(self, batch, batch_idx):
        self.optimizer_small.zero_grad()
        self.optimizer_big.zero_grad()

        xs, target, gccs, num_spk = batch['xs'], batch['ts'], batch["gccs"], batch["num_spks"]


        try:
            if self.num_spk:
                if self.ov:
                    ov = (num_spk >= 2).float().to(xs.device)
                    y_pred = self.model(xs, ov)
                else:
                    if self.spk_prob:
                        num_spk = torch.round(num_spk).long()
                        num_spk = torch.clamp(num_spk, 0, self.model.module.max_num_spk - 1)
                        num_spk = torch.nn.functional.one_hot(
                            num_spk, num_classes=self.model.module.max_num_spk
                        ).float()
                        num_spk = num_spk.squeeze(dim=2)
                        num_spk = num_spk.to(xs.device)
                    y_pred = self.model(xs, num_spk)
            elif self.num_spk_and_gcc:
                y_pred = self.model(gccs, num_spk)
            else:
                y_pred = self.model(xs, gccs)

            if self.bce_loss:
                loss, perm = pit_bce_loss(y_pred, target)
            else:
                # powerset
                multilabel = self.unwrap_model.powerset.to_multilabel(y_pred)
                permutated_target, _ = permutate(multilabel, target)
                permutated_target_powerset = self.unwrap_model.powerset.to_powerset(
                    permutated_target.float()
                )

                loss = nll_loss(
                    y_pred,
                    torch.argmax(permutated_target_powerset, dim=-1)
                )

                # Auxiliary L2 loss for merging layer weights
                if self.aux_loss:
                    aux_weight = 1e-4
                    W = self.model.merged_linear.weight
                    attention_in = W.shape[0]
                    W2 = W[:, attention_in:]
                    l2_loss =  torch.abs(200 - torch.norm(W2, 2))
                    loss = loss + aux_weight * l2_loss

        except Exception as e:
            print(f"Error during training step: {e}", flush=True)
            assert False, f"Error during training step: {e}"

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            # The gradients are added across all processes in this cumulative gradient accumulation step.
            self.auto_clip_grad_norm_(self.model)

        self.optimizer_small.step()
        self.optimizer_big.step()

        return {"Loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        xs, target, gccs, num_spk = batch['xs'], batch['ts'], batch["gccs"], batch["num_spks"]
        sil_all_target = torch.zeros_like(target)

        if self.num_spk:
            if self.ov:
                ov = (num_spk >= 2).float().to(xs.device)
                y_pred = self.model(xs, ov)
            else:
                if self.spk_prob:
                    num_spk = torch.round(num_spk).long()
                    num_spk = torch.clamp(num_spk, 0, self.model.module.max_num_spk - 1)
                    num_spk = torch.nn.functional.one_hot(
                        num_spk, num_classes=self.model.module.max_num_spk
                    ).float()
                    num_spk = num_spk.squeeze(dim=2)
                    num_spk = num_spk.to(xs.device)
                y_pred = self.model(xs, num_spk)
        elif self.num_spk_and_gcc:
            y_pred = self.model(gccs, num_spk)
        else:
            y_pred = self.model(xs, gccs)

        if self.bce_loss:
            loss, perm = pit_bce_loss(y_pred, target)

            probs = torch.sigmoid(y_pred)
            multilabel = (probs > 0.5).float()

            fig = self.plot_probs(probs)
            self.writer.add_figure("Activity/Model_probs", fig, global_step=self.state.epochs_trained)
            plt.close(fig)
            fig = self.plot_probs(target, perm)
            self.writer.add_figure("Activity/Target", fig, global_step=self.state.epochs_trained)
            plt.close(fig)
            target_spk_count = torch.sum(target, dim=-1)  # sum over classes
        else:
            # powerset
            multilabel = self.unwrap_model.powerset.to_multilabel(y_pred)
            permutated_target, _ = permutate(multilabel, target)
            permutated_target_powerset = self.unwrap_model.powerset.to_powerset(
                permutated_target.float()
            )

            loss = nll_loss(y_pred,
                torch.argmax(permutated_target_powerset, dim=-1)
            )

            target_spk_count = torch.sum(target, dim=-1)  # sum over classes
            ### accuracy und confusion matrix
        pred_labels = torch.argmax(multilabel, dim=-1)
        gt_labels = target_spk_count.squeeze()
        self.all_preds.append(pred_labels.cpu())
        self.all_targets.append(gt_labels.cpu())








        val_metrics = self.unwrap_model.validation_metric(
            torch.transpose(multilabel, 1, 2),
            torch.transpose(target, 1, 2),
        )

        rank = self.accelerator.process_index
        der, miss, fa, conf, der_ov, der_s = get_der_ov(multilabel, target, rank, self.exp_dir)

        if not torch.equal(target, sil_all_target):
            val_DER = val_metrics['DiarizationErrorRate']
            val_DER2 = der
            val_DER_ov = val_metrics['OverlapDiarizationErrorRate']
            val_DER_ov2 = der_ov
            val_DER_s = der_s
            val_FA = val_metrics['DiarizationErrorRate/FalseAlarm']
            val_Miss = val_metrics['DiarizationErrorRate/Miss']
            val_Confusion = val_metrics['DiarizationErrorRate/Confusion']
        else:
            val_DER = torch.zeros_like(val_metrics['DiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            val_DER2 = torch.zeros_like(val_metrics['DiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            val_DER_ov = torch.zeros_like(val_metrics['OverlapDiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            val_DER_ov2 =  torch.zeros_like(val_metrics['OverlapDiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            val_DER_s = torch.zeros_like(val_metrics['DiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            val_FA = torch.zeros_like(val_metrics['DiarizationErrorRate/FalseAlarm'], device=val_metrics['DiarizationErrorRate'].device)
            val_Miss = torch.zeros_like(val_metrics['DiarizationErrorRate/Miss'], device=val_metrics['DiarizationErrorRate'].device)
            val_Confusion = torch.zeros_like(val_metrics['DiarizationErrorRate/Confusion'], device=val_metrics['DiarizationErrorRate'].device)

        return {"Loss": loss, "DER": val_DER,  "val_DER2": val_DER2, "val_DER_ov2": val_DER_ov2, "val_DER_s": val_DER_s,  #"DER_ov": val_DER_ov,
               "FA": val_FA, "Miss": val_Miss, "Confusion": val_Confusion}

    def validation_epoch_end(self, validation_epoch_output):
        metric_keys = validation_epoch_output[0].keys()
        # Compute mean loss on all loss items on a epoch
        for key in metric_keys:

            metric_items = [torch.mean(torch.as_tensor(step_out[key], dtype=torch.float32))
                            for step_out in validation_epoch_output]
            # metric_mean = torch.mean(torch.stack(metric_items))

            # metric_items = [torch.mean(step_out[key]) for step_out in validation_epoch_output]
            metric_mean = torch.mean(torch.tensor(metric_items))
            if key == "Loss":
                Loss_val = metric_mean
            if key == "DER":
                DER_val = metric_mean
            if key == "val_DER_ov2":
                DER_val_ov = metric_mean
            self.writer.add_scalar(f"Validation_Epoch/{key}", metric_mean, self.state.epochs_trained)
        logger.info(
            f"Validation Loss/DER/DER_ov on epoch {self.state.epochs_trained}: {round(Loss_val.item(), 3)} / {round(DER_val.item(), 3)} / {round(DER_val_ov.item(), 3)}")

        # ### confusion matrix
        # fig = self.plot_cm()
        # self.writer.add_figure("Confusion Matrix/Validation", fig, global_step=self.state.epochs_trained)
        # plt.close(fig)
        # self.cm_normalized = self.cm.astype(np.float32) / self.cm.sum(axis=1, keepdims=True)
        # self.cm = np.nan_to_num(self.cm_normalized)  # for division by 0
        # fig = self.plot_cm(normalized=True)
        # self.writer.add_figure("Confusion Matrix/Validation_normalized", fig, global_step=self.state.epochs_trained)
        # plt.close(fig)

        self.all_preds = []
        self.all_targets = []

        # metric reset
        self.unwrap_model.validation_metric.reset()
        return Loss_val

    def plot_cm(self, normalized=False):
        print("Plotting confusion matrix, normalized:", normalized)
        fig, ax = plt.subplots(figsize=(6, 5))
        if normalized:
            sns.heatmap(self.cm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=[f"Pred {i}" for i in range(self.num_classes)],
                        yticklabels=[f"True {i}" for i in range(self.num_classes)],
                        ax=ax)
        else:
            sns.heatmap(self.cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[f"Pred {i}" for i in range(self.num_classes)],
                        yticklabels=[f"True {i}" for i in range(self.num_classes)],
                        ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Validation)")
        return fig

    def plot_probs(self, act, perm=None, batch_idx=0, start_frame=100, end_frame=200):
        """
        pred: (B, F, C) logits oder probabilities
        batch_idx: int
        start_frame, end_frame: fix Frames
        """
        if perm is not None:
            act_example = act[batch_idx, :, perm[batch_idx]]
        else:
            act_example = act[batch_idx, :, :]   # (frames, C)  start_frame:end_frame
        # frames, C = act_example.shape

        fig, ax = plt.subplots(figsize=(19, 4))

        for c, act in enumerate(act_example.T):
            ax.plot(act.cpu().numpy() * 0.95 + c, color=plt.get_cmap("tab10")(c))

        ax.set_xlabel("Frames")
        ax.set_ylabel("Speaker Level")
        ax.set_title(f"Predicted speaker activity example {batch_idx}")
        # ax.set_yticks(range(C))
        # ax.set_yticklabels([f"Speaker {c}" for c in range(C)])
        ax.grid(True, axis="y")

        return fig