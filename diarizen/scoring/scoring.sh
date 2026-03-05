#!/bin/bash

recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_gcc
export exp_root=$recipe_root/exp
# TODO: Diarization Dir setzen where you want to score
diarization_dir=$exp_root/8channels/infer_oracle_clustering/metric_Loss_best/avg_ckpt5
    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer

data_dir=$recipe_root/data_mc
dtype=test_marc
echo "stage3: scoring..."
mkdir -p ${diarization_dir}/${dtype}
for dset in  NOTSOFAR1 AliMeeting CHiME7 AMI AISHELL4 ; do
    OUT_DIR=${diarization_dir}/${dtype}/${dset}
    for collar in 0 0.25; do
    micromamba activate diarizen && python -m diarizen.scoring.der_ov \
                    --storage_dir $OUT_DIR \
                    --ref ${data_dir}/${dtype}/${dset}/rttm \
                    --collar ${collar}
#    micromamba activate diarizen && python ${dscore_dir}/score.py \
#                     -r ${data_dir}/${dtype}/${dset}/rttm \
#                    -s $OUT_DIR/*.rttm --collar ${collar} \
#                   > $OUT_DIR/result_collar${collar}
    done
done