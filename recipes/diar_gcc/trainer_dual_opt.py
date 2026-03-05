# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
import os
import torch.profiler
import itertools
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import paderbox as pb
from accelerate.logging import get_logger
from padertorch.ops.losses.source_separation import compute_pairwise_losses, pit_loss_from_loss_matrix
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.loss import nll_loss

from diarizen.combine_act import solve_permutation
from diarizen.trainer_dual_opt import Trainer as BaseTrainer
from diarizen.scoring.der_ov import compute_der
from torch import nn
import time

logger = get_logger(__name__)


def pit_bce_loss(logits, target):
    """
    PIT + BCE Loss
        logits: (B, T, S)
        target: (B, T, S)
    """
    B, T, S = logits.shape
    best_losses = []
    best_perms = []

    for b in range(B):
        logits_b = logits[b]  # (T, S)
        target_b = target[b]  # (T, S)
        # perms = list(itertools.permutations(range(S)))
        # losses = []

        # for perm in perms:
        #     perm_target = target_b[:, perm]
        #     loss = F.binary_cross_entropy_with_logits(logits_b, perm_target.float(), reduction='mean')
        #     losses.append(loss)
        # losses = torch.stack(losses)
        # best_loss, best_idx = losses.min(dim=0)
        # best_perm = perms[best_idx]
        # best_losses.append(best_loss)
        # best_perms.append(best_perm)
        #
        # import pdb
        # pdb.set_trace()
        pair_wise_loss_matrix = compute_pairwise_losses(estimate=logits_b, target=target_b.float(), axis=-1,
                                                        loss_fn=F.binary_cross_entropy_with_logits)
        min_loss, col_ind = pit_loss_from_loss_matrix(pair_wise_loss_matrix, return_permutation=True)
        # Col_ind describes the indices to take from target so that it matches the prediction best

        best_losses.append(min_loss)
        best_perms.append(col_ind)

    final_loss = torch.stack(best_losses).mean()

    return final_loss, best_perms


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


def get_der_ov(multilabel, target, rank=0, exp_dir=None):
    os.makedirs(exp_dir / "tmp", exist_ok=True)
    hyp_path = exp_dir / f"tmp/all_hyp_{rank}.rttm"
    ref_path =  exp_dir / f"tmp/ref_{rank}.rttm"
    #### ref
    target = target.cpu().numpy().astype(int)
    ref_rttm, skip_sess = labels_to_rttm(target, frame_shift=320, sample_rate=16000)
    with open(ref_path, "w") as f:
        f.write("\n".join(ref_rttm))
    if len(ref_rttm) == 0:
        print("EMPTY! No reference RTTM, skipping DER computation.")
        return 0, 0, 0, 0, 0, 0
    ## hyp
    multilabel = multilabel.cpu().numpy().astype(int)
    hyp_rttm, _ = labels_to_rttm(multilabel, frame_shift=320, sample_rate=16000, empty=skip_sess)
    # import pdb
    with open(hyp_path, "w") as f:
        # pdb.set_trace()
        f.write("\n".join(hyp_rttm))
    # import pdb
    # pdb.set_trace()

    # multilabel = multilabel.reshape(-1, S)
    # target = target.reshape(-1, S)
    # multilabel = multilabel.cpu().numpy().astype(bool)
    # target = target.cpu().numpy().astype(bool)
    # print("multilabel shape: ", multilabel.shape, "target shape: ", target.shape, flush=True)
    # # go from frames to samples here
    # for
    # for s in range(S):
    #     act = pb.array.ArrayInterval(multilabel[]
    # multilabel = pb.array.interval
    #
    # # save rttms
    # # hyp = pb.array.interval.rttm.to_rttm_str({"0": multilabel})
    # # ref = pb.array.interval.rttm.to_rttm_str({"0": target})
    # pb.array.interval.rttm.to_rttm({"0": multilabel}, hyp_path)
    # pb.array.interval.rttm.to_rttm({"0": target}, ref_path)

    storage_dir = exp_dir / "tmp"
    collar = 0
    out_file = Path(storage_dir) / f"results{collar}_{rank}.json"
    der, der_ov, der_s, fa, miss, conf = compute_der(storage_dir, ref_path, collar, rank)
    with open(out_file, "w") as f:
        header = "File                 DER     Miss     FA       SpkE     DER_ov    DER_s"
        separator = "-" * len(header)
        row = "{:<18}{:7.3f}  {:7.3f}  {:6.3f}  {:7.3f}  {:8.3f}  {:7.3f}".format(
            "*** OVERALL ***", der, miss, fa, conf, der_ov, der_s
        )

        f.write(header + "\n")
        f.write(separator + "\n")
        f.write(row + "\n")

    # print(f"Ergebnisse gespeichert in: {out_file}")


    # ### Accumulate DER counts
    # perm = solve_permutation(multilabel, target)    #
    # def solve_permutation(activities, ref_activities):
    #     spks = np.maximum(len(activities.keys()), len(ref_activities.keys()))
    #     overlaps = np.zeros((spks, spks))
    #     for i, (spk, act) in enumerate(activities.items()):
    #         for j, (ref_spk, ref_act) in enumerate(ref_activities.items()):
    #             ref_act.shape = act.shape
    #             overlaps[i, j] = np.sum(act & ref_act)
    #
    #     permutations = solve_permutation_hungarian(overlaps, minimize=False)
    #     # np.arange (spks), permutations
    #     return permutations
    # multilabel_perm = multilabel[:, perm]

    return der, miss, fa, conf, der_ov, der_s

class FocalLossMC(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C), target: (B,)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        # Gather p_t und log p_t
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)   # (B,)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)         # (B,)
        focal = (1 - pt).pow(self.gamma) * (-logpt)              # (B,)

        if self.weight is not None:
            alpha = self.weight[target]                          # (B,)
            focal = alpha * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal


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

    import torch
    import torch.nn as nn

    def guarded_ce_loss(self, logits, targets, ignore_index=None):
        """
        logits: (N, C, ...) raw scores
        targets: (N, ...) integer class ids
        """
        # Basic shape/dtype checks
        assert logits.dim() >= 2, f"logits shape invalid: {tuple(logits.shape)}"
        C = logits.size(1)
        assert targets.dtype == torch.long, f"targets must be int64, got {targets.dtype}"
        assert logits.device == targets.device, f"device mismatch: {logits.device} vs {targets.device}"

        # Identify invalids (excluding ignore_index if provided)
        invalid = (targets < 0) | (targets >= C)
        if ignore_index is not None:
            invalid = invalid & (targets != ignore_index)

        if invalid.any():
            # Extract a few offenders
            bad_idxs = invalid.nonzero(as_tuple=False)
            uniq_vals = torch.unique(targets[invalid])
            msg = (
                f"[CrossEntropy sanity] Invalid targets detected!\n"
                f"  logits.shape = {tuple(logits.shape)} (C={C})\n"
                f"  targets.shape = {tuple(targets.shape)}\n"
                f"  #invalid = {invalid.sum().item()}\n"
                f"  unique invalid values = {uniq_vals.tolist()}\n"
                f"  first bad indices (up to 10): {bad_idxs[:10].tolist()}"
            )
            print(msg, flush=True)
            import pdb
            pdb.set_trace()
            raise ValueError(msg)

        # Optional numerical guard on logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise FloatingPointError("NaN/Inf in logits before loss.")

        return nn.CrossEntropyLoss(ignore_index=ignore_index)(logits, targets)

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

        # # debug:
        # th = 1.6
        # o = 0
        # for i, t in enumerate(target):
        #     num_spk = torch.sum(t, dim=-1, keepdim=True)
        #     ov = torch.clamp(num_spk, 1, 2)
        #     ov_ratio = torch.sum(ov) / len(ov)
        #     if ov_ratio >= th:
        #         print(names[i])
        #         o += 1
        #
        # print("Overlap detected examples: ", o, " / ", len(target), o/len(target), flush=True)

        if self.num_spk:
            y_pred = self.model(xs, num_spk)
        elif self.num_spk_and_gcc:
            y_pred = self.model(gccs, num_spk)
        else:
            y_pred = self.model(xs, gccs)

        if not self.spk_count_loss:
            if self.bce_loss:
                # print(y_pred.shape, target.shape)
                # print("3spk aktiv count:", (torch.sum(y_pred, dim=-1) == 3).sum().item())
                loss, perm = pit_bce_loss(y_pred, target)
                # probs = torch.sigmoid(y_pred)
                # multilabel = (probs > 0.5).float()
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
        else:
            # spk counting loss
            num_classes = y_pred.shape[-1]
            target_spk_count = torch.sum(target, dim=-1)  # sum over classes
            # loss = self.guarded_ce_loss(y_pred.view(-1, num_classes), target_spk_count.view(-1), ignore_index=None)
            freqs = {0: 0.073, 1: 0.71, 2: 0.17, 3: 0.04, 4: 0.007}
            weights = torch.tensor([1.0 / freqs[c] for c in sorted(freqs)], dtype=torch.float32, device=y_pred.device)
            weights = weights / weights.sum() * len(weights)

            if self.weighted_loss:
                spk_count_loss = nn.CrossEntropyLoss(weight=weights)(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            elif self.focal_loss:
                spk_count_loss = FocalLossMC(gamma=2.0, weight=weights)(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            elif self.ov_loss:
                target_spk_count = torch.clamp(target_spk_count, min=0, max=2)
                spk_count_loss = nn.CrossEntropyLoss()(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            else:
                target_spk_count = torch.clamp(target_spk_count, min=0, max=self.model.module.max_num_spk-1)
                # target_spk_count = torch.clamp(target_spk_count, min=0, max=self.model.max_num_spk-1)
                spk_count_loss = nn.CrossEntropyLoss()(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            loss = spk_count_loss
            # accuracy
            for spk_gt, pred in zip(target_spk_count, y_pred):
                pred_labels = torch.argmax(pred, dim=-1)
                gt_labels = spk_gt.squeeze()

                hits = (pred_labels == gt_labels).sum().item()
                total = pred_labels.numel()
                self.num_correct += hits
                self.num_total += total

                hits_ov = ((pred_labels == gt_labels) & (pred_labels >= 2)).sum().item()
                total_ov = (gt_labels >= 2).sum().item()
                self.num_correct_ov += hits_ov
                self.num_total_ov += total_ov


        # aux_loss_fn = VICRegLoss()
        # todo: get samples selection mit ground truth
        # aux_loss = aux_loss_fn(encoder_output_a, encoder_output_b)
        # loss = loss + self.aux_weight * aux_loss

        # skip batch if something went wrong for some reason
        # import pdb
        # pdb.set_trace()
        if torch.isnan(loss):
            return None

        self.accelerator.backward(loss)
        # print(f"self.model.module.gcc_encoder.conv[0].weight: grad norm = {self.model.module.gcc_encoder.conv[0].weight.grad.norm().item():.6f}",
        #     flush=True)

        # try:
        #     if self.model.module.gcc_encoder.conv[0].weight.grad is not None:
        #         self.writer.add_scalar("Gradients/gcc_encoder_conv0_weight_norm",
        #                                self.model.module.gcc_encoder.conv[0].weight.grad.norm().item(), self.state.steps_trained)
        # except Exception as e:
        #     if self.model.gcc_encoder.conv[0].weight.grad is not None:
        #         self.writer.add_scalar("Gradients/gcc_encoder_conv0_weight_norm",
        #                                self.model.gcc_encoder.conv[0].weight.grad.norm().item(), self.state.steps_trained)

        if self.accelerator.sync_gradients:
            # The gradients are added across all processes in this cumulative gradient accumulation step.
            self.auto_clip_grad_norm_(self.model)

        self.optimizer_small.step()
        self.optimizer_big.step()

        return {"Loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        t1 = time.time()
        xs, target, gccs, num_spk = batch['xs'], batch['ts'], batch["gccs"], batch["num_spks"]

        # # debug:
        # th = 1.6
        # o = 0
        # for t in target:
        #     num_spk = torch.sum(t, dim=-1, keepdim=True)
        #     ov = torch.clamp(num_spk, 1, 2)
        #     ov_ratio = torch.sum(ov) / len(ov)
        #     if ov_ratio >= th:
        #         o += 1
        #
        # print("Overlap detected examples: ", o, " / ", len(target), o/len(target), flush=True)

        sil_all_target = torch.zeros_like(target)

        if self.num_spk:
            y_pred = self.model(xs, num_spk)
        elif self.num_spk_and_gcc:
            y_pred = self.model(gccs, num_spk)
        else:
            y_pred = self.model(xs, gccs)

        t2 = time.time()
        if not self.spk_count_loss:
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
        else:
            # spk counting loss
            num_classes = y_pred.shape[-1]
            target_spk_count = torch.sum(target, dim=-1)  # sum over classes
            # loss = self.guarded_ce_loss(y_pred.view(-1, num_classes), target_spk_count.view(-1), ignore_index=None)
            freqs = {0: 0.073, 1: 0.71, 2: 0.17, 3: 0.04, 4: 0.007}
            weights = torch.tensor([1.0 / freqs[c] for c in sorted(freqs)], dtype=torch.float32, device=y_pred.device)
            weights = weights / weights.sum() * len(weights)

            if self.weighted_loss:
                spk_count_loss = nn.CrossEntropyLoss(weight=weights)(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            elif self.focal_loss:
                spk_count_loss = FocalLossMC(gamma=2.0, weight=weights)(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            elif self.ov_loss:
                target_spk_count = torch.clamp(target_spk_count, min=0, max=2)
                spk_count_loss = nn.CrossEntropyLoss()(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            else:
                # print(y_pred.shape, target_spk_count.shape)
                target_spk_count = torch.clamp(target_spk_count, min=0, max=self.model.module.max_num_spk-1)
                # target_spk_count = torch.clamp(target_spk_count, min=0, max=self.model.max_num_spk-1)
                spk_count_loss = nn.CrossEntropyLoss()(y_pred.view(-1, num_classes), target_spk_count.view(-1))
            loss = spk_count_loss

            # ### accuracy und confusion matrix
            # # TODO multilabel für die 3 fälle oben einstellen
            # pred_labels = torch.argmax(multilabel, dim=-1)
            # # pred_labels = torch.argmax(y_pred, dim=-1)
            # gt_labels = target_spk_count.squeeze()
            # self.all_preds.append(pred_labels.cpu())
            # self.all_targets.append(gt_labels.cpu())
            #
            # hits = (pred_labels == gt_labels).sum().item()
            # total = pred_labels.numel()
            # self.num_correct += hits
            # self.num_total += total
            #
            # hits_ov = ((pred_labels == gt_labels) & (pred_labels >= 2)).sum().item()
            # total_ov = (gt_labels >= 2).sum().item()
            # self.num_correct_ov += hits_ov
            # self.num_total_ov += total_ov

        ### accuracy und confusion matrix
        pred_labels = torch.argmax(multilabel, dim=-1)
        gt_labels = target_spk_count.squeeze()
        self.all_preds.append(pred_labels.cpu())
        self.all_targets.append(gt_labels.cpu())

        t3 = time.time()
        val_metrics = self.unwrap_model.validation_metric(
            torch.transpose(multilabel, 1, 2),
            torch.transpose(target, 1, 2),
        )
        # print(val_metrics, flush=True)

        # store rttms and compute der ov
        # if self.accelerator.is_local_main_process:
        t4 = time.time()
        if self.compute_second_der:
            rank = self.accelerator.process_index
            # das hier kostet Zeit
            der, miss, fa, conf, der_ov, der_s = get_der_ov(multilabel, target, rank, self.exp_dir)
            # print(f"DER: {der:.3f}, DER_ov: {der_ov:.3f}, DER_s: {der_s:.3f}, Miss: {miss:.3f}, FA: {fa:.3f}, Conf: {conf:.3f}", flush=True)
        t5 = time.time()

        if not torch.equal(target, sil_all_target):
            val_DER = val_metrics['DiarizationErrorRate']
            val_DER_ov = val_metrics['OverlapDiarizationErrorRate']
            if self.compute_second_der:
                val_DER2 = der
                val_DER_ov2 = der_ov
                val_DER_s = der_s
            val_FA = val_metrics['DiarizationErrorRate/FalseAlarm']
            val_Miss = val_metrics['DiarizationErrorRate/Miss']
            val_Confusion = val_metrics['DiarizationErrorRate/Confusion']
        else:
            val_DER = torch.zeros_like(val_metrics['DiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            if self.compute_second_der:
                val_DER2 = torch.zeros_like(val_metrics['DiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
                val_DER_ov2 =  torch.zeros_like(val_metrics['DiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
                val_DER_s = torch.zeros_like(val_metrics['DiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            val_DER_ov = torch.zeros_like(val_metrics['OverlapDiarizationErrorRate'], device=val_metrics['DiarizationErrorRate'].device)
            val_FA = torch.zeros_like(val_metrics['DiarizationErrorRate/FalseAlarm'], device=val_metrics['DiarizationErrorRate'].device)
            val_Miss = torch.zeros_like(val_metrics['DiarizationErrorRate/Miss'], device=val_metrics['DiarizationErrorRate'].device)
            val_Confusion = torch.zeros_like(val_metrics['DiarizationErrorRate/Confusion'], device=val_metrics['DiarizationErrorRate'].device)

        t6 = time.time()
        # print(f"Validation step times: forward={t2-t1:.3f}s, Loss={t3-t2:.3f}s, DER={t4-t3:.3f}s, DER2= {t5-t4:.3f}s, Ende={t6-t5:.3f}s", flush=True)
        if self.compute_second_der:
            return {"Loss": loss, "DER": val_DER,  "val_DER2": val_DER2, "val_DER_ov2": val_DER_ov2, "val_DER_s": val_DER_s,  #"DER_ov": val_DER_ov,
                   "FA": val_FA, "Miss": val_Miss, "Confusion": val_Confusion}
        else:
            return {"Loss": loss, "DER": val_DER, "val_DER_ov": val_DER_ov, "FA": val_FA, "Miss": val_Miss, "Confusion": val_Confusion}

    def validation_epoch_end(self, validation_epoch_output):
        metric_keys = validation_epoch_output[0].keys()
        # Compute mean loss on all loss items on a epoch
        for key in metric_keys:
            metric_items = [torch.mean(step_out[key]) for step_out in validation_epoch_output]
            metric_mean = torch.mean(torch.tensor(metric_items))
            if key == "Loss":
                Loss_val = metric_mean
            if key == "DER":
                DER_val = metric_mean
            if key == "val_DER_ov2":
                DER_val_ov = metric_mean
            self.writer.add_scalar(f"Validation_Epoch/{key}", metric_mean, self.state.epochs_trained)
        self.writer.add_scalar(f"Validation_Epoch/accuracy", self.accuracy, self.state.epochs_trained)
        self.writer.add_scalar(f"Validation_Epoch/accuracy_OV", self.accuracy_ov, self.state.epochs_trained)
        logger.info(f"Validation Loss/DER/DER_ov on epoch {self.state.epochs_trained}: {round(Loss_val.item(), 3)} / {round(DER_val.item(), 3)} / {round(DER_val_ov.item(), 3)}")

        ### confusion matrix
        fig = self.plot_cm()
        self.writer.add_figure("Confusion Matrix/Validation", fig, global_step=self.state.epochs_trained)
        plt.close(fig)
        self.cm_normalized = self.cm.astype(np.float32) / self.cm.sum(axis=1, keepdims=True)
        self.cm = np.nan_to_num(self.cm_normalized)  # for division by 0
        fig = self.plot_cm(normalized=True)
        self.writer.add_figure("Confusion Matrix/Validation_normalized", fig, global_step=self.state.epochs_trained)
        plt.close(fig)


        ### examples
        figs = self. plot_predictions_for_examples()
        for i, fig in enumerate(figs):
            self.writer.add_figure(f"Examples/Validation_{i}", fig, global_step=self.state.epochs_trained)
            plt.close(fig)

        self.all_preds = []
        self.all_targets = []

        # metric reset
        self.unwrap_model.validation_metric.reset()
        return Loss_val

    def plot_cm(self, normalized=False):
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

    def plot_predictions_for_examples(self,):
        figs = []
        for idx in range(len(self.ex_target)):
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(self.ex_target[idx], label="Ground Truth", linewidth=2)
            ax.plot(self.ex_pred[idx], label="Prediction", linestyle="--")
            ax.set_title(f"Validation Example {idx}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Speaker Count")
            ax.legend()
            figs.append(fig)
        return figs

class VICRegLoss(nn.Module):
    # TODO: mitteln über frames in denen zielsprecher aktiv ist, aber nur dort
    def __init__(self, lambda_=25.0, mu=25.0, nu=1.0, eps=1e-4):
        super().__init__()
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.eps = eps

    def forward(self, x, y):
        # x, y: (B, D)
        repr_loss = F.mse_loss(x, y)

        # Variance loss
        std_x = torch.sqrt(x.var(dim=0) + self.eps)
        std_y = torch.sqrt(y.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        # Covariance loss
        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)

        cov_x = (x_centered.T @ x_centered) / (x.shape[0] - 1)
        cov_y = (y_centered.T @ y_centered) / (y.shape[0] - 1)

        # remove diagonals
        off_diag_x = cov_x.fill_diagonal_(0)
        off_diag_y = cov_y.fill_diagonal_(0)

        cov_loss = (off_diag_x ** 2).sum() / x.shape[1] + (off_diag_y ** 2).sum() / y.shape[1]

        return self.lambda_ * repr_loss + self.mu * var_loss + self.nu * cov_loss


