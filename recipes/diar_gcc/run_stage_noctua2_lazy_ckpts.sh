#!/bin/bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048


recipe_root=/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_gcc
export exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf

#spk_count_linear_noisy_to_gcpsd_encoder_ffn_film_all_layers_finetune_test
# training setup
use_dual_opt=true
export train_conf="$conf_dir/${1}.toml"
#find wavs -type f | xargs -P 40 -I{} rsync -R {} /mnt/ssd/AMI_AIS_ALI_NSF_CHiME7/

# 1_layer_enc  precomp_5fetcher.toml  merge_normalized noisy_mergelayer added_features noisy_mergelayer_without_extra_layernorm
# only_load_wavlm  cross_attention  added_features gcc_pretrained_no_init cross_attention_concat gcc_pretrained_8channels
echo "Using train config: $train_conf"

conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

# inference setup
dtype=test_marc
data_dir=$recipe_root/data_no_chime
seg_duration=8

# clustering setu
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
dscore_dir=/scratch/hpc-prf-nt2/deegen/merlin/dscore/dscore

echo $CUDA_VISIBLE_DEVICES

echo "stage1: use dual-opt for model training..."
source  /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/.diarizen/bin/activate && CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
    --num_processes 4 --main_process_port 1137 \
    eval_ckpts.py -C $train_conf -M train -R