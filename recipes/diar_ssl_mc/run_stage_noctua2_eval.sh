#!/bin/bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

# general setup
stage=2
resume_flag=""  # default: no resume training

recipe_root=/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc
export exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf

#TODO: resume logik umgekehrt für running experiments => Done 19.03
if [ $# -ge 2 ] && [ "$2" == "-r" ]; then
    resume_flag="-R"
fi

# training setup
use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
export train_conf="$conf_dir/${1}.toml" # gcc_encoder_precomputed  baseline_sc_debug.toml   #    mc_wavlm_updated_conformer.toml #MC
# spk_count_linear_noisy_to_gcpsd_encoder_ffn_film_deep_finetune2
# spk_count_linear_noisy_to_gcpsd_encoder_ffn_film_all_layers_finetune
# spk_count_linear_prob_gcpsd  spk_count_linear_noisy_to_gcpsd_ffn_frozen_median spk_count_linear_noisy_to_gcpsd_ffn_film_all layers spk_count_linear_noisy_to_gcpsd_encoder_ffn_film_deep
# spk_count_linear_noisy_modelbased  spk_count_linear_ov spk_count_linear_noisy_to_gcpsd_9layer_frozen_median_lr spk_count_linear_noisy_to_gcpsd_ffn_frozen_median
# spk_count_ref spk_count_linear_noisy_labels precomp_5fetcher_no_chime  precomp_5fetcher_no_chime_random_init  spk_count_linear_noisy_to_gcc


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

# =======================================
# =======================================
if [ $stage -le 1 ]; then
    if (! $use_dual_opt); then
        echo "stage1: use single-opt for model training..."
        source  /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/.diarizen/bin/activate && CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
            --num_processes 2 --main_process_port 1134 \
            run_single_opt.py -C $train_conf -M validate
    else
        echo "stage1: use dual-opt for model training..."
        source  /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/.diarizen/bin/activate && CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
            --num_processes 4 --main_process_port 1137 \
            run_dual_opt_mc_lazy.py -C $train_conf -M train $resume_flag
##################            ### Debugguing: use only one GPU and worker
#        source /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/.diarizen/bin/activate && CUDA_VISIBLE_DEVICES="0" accelerate launch \
#            --num_processes 1 --main_process_port 1134 \
#            run_dual_opt_mc_lazy.py -C $train_conf -M train $resume_flag
    fi
fi

diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
diarizen_hub=/mnt/matylda3/ihan/hugging-face/hub/models--BUT-FIT--diarizen-wavlm-large-md-s80/snapshots/50167e9a5243663ff51777823ff7d53da7b87166
if [ ! -d $diarizen_hub ]; then
    diarizen_hub=/scratch/hpc-prf-nt2/deegen/merlin/hub/models--BUT-FIT--diarizen-wavlm-large-md-s80/snapshots/50167e9a5243663ff51777823ff7d53da7b87166
fi
embedding_model=/mnt/matylda3/ihan/project/pretrained/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
if [ ! -e $embedding_model ]; then
    embedding_model=/scratch/hpc-prf-nt2/deegen/merlin/embedding_model/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin
fi

if [ $stage -le 2 ]; then
    echo "stage2: model inference..."
#    export CUDA_VISIBLE_DEVICES=5

#    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
#    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst

    for dset in   AMI AISHELL4 ; do # NOTSOFAR1 AliMeeting
        echo "Inference on $dset..."
        # conda activate diarizen && python infer_avg_mc.py -C $config_dir \ ### CHiME7
        source  /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/.diarizen/bin/activate && python infer_avg_oracle.py -C $config_dir \
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
            --Fb $Fb \


        echo "stage3: scoring..."
        SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
        OUT_DIR=${SYS_DIR}/${dtype}/${dset}

        if [ -f "${OUT_DIR}/referenz.rttm" ]; then
          mv "${OUT_DIR}/referenz.rttm" "${OUT_DIR}/old_ref"
        fi
        if [ -f "${OUT_DIR}/all_hyp.rttm" ]; then
          mv "${OUT_DIR}/all_hyp.rttm" "${OUT_DIR}/old_all_hyp"
        fi

        for collar in 0 0.25; do
            source  /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/.diarizen/bin/activate && python ${dscore_dir}/score.py \
                -r ${REF_DIR}/${dtype}/${dset}/rttm \
                -s $OUT_DIR/*.rttm --collar ${collar} \
                > $OUT_DIR/result_collar${collar}
        done
        echo "stage4: overlap scoring..."
        for collar in 0 0.25; do
        source  /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/.diarizen/bin/activate && python -m diarizen.scoring.der_ov \
                        --storage_dir $OUT_DIR \
                        --ref ${REF_DIR}/${dtype}/${dset}/rttm \
                        --collar ${collar}
        done
    done
fi
