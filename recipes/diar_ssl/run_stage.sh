#!/bin/bash

#huggingface neue geprunte modelle large und abse nutzen? wav src anpassen

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

# general setup
stage=1
recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl
exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf
#mkdir -p $exp_root
#mkdir -p $conf_dir
# training setup
use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
#train_conf=$conf_dir/s80_wavlm_base_further_distill_diar_training.toml
# train_conf=$conf_dir/wavlm_frozen_conformer.toml
#train_conf=$conf_dir/wavlm_updated_conformer.toml    #SC
train_conf=$conf_dir/mc_wavlm_updated_conformer.toml #MC

conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

# inference setup
dtype=test
data_dir=$recipe_root/data_aaa/AMI_AliMeeting_AISHELL4      #SC
data_dir=$recipe_root/data_aaa/AMI_AliMeeting_AISHELL4      #MC
seg_duration=8

# clustering setup
clustering_method=AgglomerativeClustering
ahc_threshold=0.70
min_cluster_size=30

infer_affix=_constrained_AHC_thres_${ahc_threshold}_mcs_${min_cluster_size}_1

avg_ckpt_num=5
val_metric=Loss   # Loss or DER
val_mode=best   # [prev, best, center]

# scoring setup
collar=0
REF_DIR=$data_dir
dscore_dir=/mnt/matylda3/ihan/project/scripts/dscore

# =======================================
# =======================================
if [ $stage -le 1 ]; then
    if (! $use_dual_opt); then
        echo "stage1: use single-opt for model training..."
        micromamba activate diarizen && CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
            --num_processes 2 --main_process_port 1134 \
            run_single_opt.py -C $train_conf -M validate
    else
        echo "stage1: use dual-opt for model training..."
        micromamba activate diarizen && CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
            --num_processes 4 --main_process_port 1134 \
            run_dual_opt.py -C $train_conf -M train
##--num_processes 4    CUDA_VISIBLE_DEVICES="0,1,2,3"
## DEBUGGING#
        micromamba activate diarizen && CUDA_VISIBLE_DEVICES="0" accelerate launch \
            --num_processes 1 --main_process_port 1134 \
            run_dual_opt.py -C $train_conf -M train
    fi
fi

#diarization_dir=/mnt/matylda5/qdeegen/deploy/forschung/jsalt/DiariZen/recipes/diar_ssl/exp/wavlm_updated_conformer
diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
#mkdir -p $diarization_dir
config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
#embedding_model=$recipe_root/wespeaker.bin     # it's necessary to have "pyannote" in your directory path
embedding_model=/mnt/matylda3/ihan/project/pretrained/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin

if [ $stage -le 2 ]; then
    echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst

    for dset in AliMeeting AMI  AISHELL4; do
        micromamba activate diarizen && python infer_avg.py -C $config_dir \
            -i ${data_dir}/${dtype}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dtype}/${dset} \  ## bis hier single channel setup
            --embedding_model $embedding_model \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --seg_duration $seg_duration \
            --clustering_method $clustering_method \
            --ahc_threshold $ahc_threshold \
            --min_cluster_size $min_cluster_size

        echo "stage3: scoring..."
        SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
        OUT_DIR=${SYS_DIR}/${dtype}/${dset}
        micromamba activate diarizen && python ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dtype}/${dset}/rttm \   # SC
            -s $OUT_DIR/*.rttm --collar ${collar} \
            > $OUT_DIR/result_collar${collar}
    done
fi
