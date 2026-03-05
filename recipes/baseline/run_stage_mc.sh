#!/bin/bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

# general setup
stage=1
recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/baseline
export exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf


# training setup
use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
export train_conf=$conf_dir/baseline_sc_debug.toml

echo "Using train config: $train_conf"

conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

# inference setup
dtype=test_marc
data_dir=$recipe_root/data_mc
seg_duration=8

# clustering setup
#clustering_method=AgglomerativeClustering  # ??
#ahc_threshold=0.70
#min_cluster_size=30
clustering_method=VBxClustering
ahc_threshold=0.6
Fa=0.07
Fb=0.8
# infer_affix=_vbx_thres_${ahc_threshold}_Fa_${Fa}_Fb_${Fb}
infer_affix=_oracle_clustering

avg_ckpt_num=5
val_metric=Loss   # Loss or DER
val_mode=best   # [prev, best, center]  

# scoring setup
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
            run_dual_opt_mc.py -C $train_conf -M train
####            ### Debugguing: use only one GPU and worker
#        micromamba activate diarizen && CUDA_VISIBLE_DEVICES="0" accelerate launch \
#            --num_processes 1 --main_process_port 1134 \
#            run_dual_opt_mc.py -C $train_conf -M train
    fi
fi

diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
diarizen_hub=/mnt/matylda3/ihan/hugging-face/hub/models--BUT-FIT--diarizen-wavlm-large-md-s80/snapshots/50167e9a5243663ff51777823ff7d53da7b87166
embedding_model=/mnt/matylda3/ihan/project/pretrained/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path

if [ $stage -le 2 ]; then
    echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst

    for dset in  AliMeeting CHiME7 NOTSOFAR1 AMI AISHELL4 ; do
        # conda activate diarizen && python infer_avg_mc.py -C $config_dir \
        micromamba activate diarizen && python infer_avg_mc_oracle.py -C $config_dir \
            -i ${data_dir}/${dtype}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dtype}/${dset} \
            --rttm_file ${data_dir}/${dtype}/${dset}/rttm \
            --embedding_model $embedding_model \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --seg_duration $seg_duration \
            --diarizen_hub $diarizen_hub \
            --clustering_method $clustering_method \
            --ahc_threshold $ahc_threshold \
            --Fa $Fa \
            --Fb $Fb 

        echo "stage3: scoring..."
        SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
        OUT_DIR=${SYS_DIR}/${dtype}/${dset}
        for collar in 0 0.25; do
            micromamba activate diarizen && python ${dscore_dir}/score.py \
                -r ${REF_DIR}/${dtype}/${dset}/rttm \
                -s $OUT_DIR/*.rttm --collar ${collar} \
                > $OUT_DIR/result_collar${collar}
        done
    done
fi
