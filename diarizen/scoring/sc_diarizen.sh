#!/bin/bash

source /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/configs/local_paths.sh
CONFIG_NAME=leaderboard/mc_real/DiCoW_v3_diarizen
CONFIG_PATH="/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/" # /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/configs/decode/${CONFIG_NAME}.yaml
# change dir not local but in alex
export EXPERIMENT_PATH="/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/baseline_sc"
export EXPERIMENT="DiCoW_v3_baseline_sc_greedy"
export DIAR_EXPERIMENT="test"

DIAR_EXPERIMENT_PATH="${SRC_ROOT}/diar_exp/${DIAR_EXPERIMENT}"
#MODEL="BUT-FIT/diarizen-wavlm-large-s80-md"

# Define cutsets array
CUTSETS=("notsofar1-mdm_cutset_eval")   #notsofar1-sdm_cutset_eval_sc
#CUTSETS=("libri3mix_mix_clean_sc_test_cutset")

# Set up diarized cutsets path
export DIARIZED_CUTSETS_PATH="${DIAR_EXPERIMENT_PATH}/diarized_cutsets"
#mkdir -p "$DIARIZED_CUTSETS_PATH"
INPUT=$EXPERIMENT_PATH/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/NOTSOFAR1/all_hyp.rttm
## Run diarization for each cutset
for CUTSET in "${CUTSETS[@]}"; do
    echo "Processing cutset: $CUTSET"
    pwd

#    # Run diarization
#    "$SRC_ROOT/sge_tools/interactive_python_diarizen" "$SRC_ROOT/utils/diarizen_diar.py" \
#        --model="$MODEL" \
#        --input_cutset="${MANIFEST_DIR_NEW}/${CUTSET}.jsonl.gz" \
#        --output_dir="${DIAR_EXPERIMENT_PATH}/$CUTSET"
#
#    # Prepare diarized cutset from RTTM directory
    python /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/split_rttms.py --input_path $INPUT
    /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/interactive_python_diarizen /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/prepare_diar_cutset_from_rttm_dir.py \
        --lhotse_manifest_path="${MANIFEST_DIR_NEW}/${CUTSET}.jsonl.gz" \
        --rttm_dir=$EXPERIMENT_PATH/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/NOTSOFAR1/split_rttm \
        --out_manifest_path="$EXPERIMENT_PATH/${CUTSET}.jsonl.gz"
done

echo "Diarization completed for all cutsets"

# 1. Create decoding config
mkdir -p "$(dirname "$CONFIG_PATH")"
cat <<'EOF' > "$CONFIG_PATH"
# @package _global_
experiment: ${oc.env:EXPERIMENT}
exp_dir: exp

wandb:
  project: EMMA_leaderboard

model:
  whisper_model: "BUT-FIT/DiCoW_v3"
  reinit_from: null

data:
  eval_cutsets:
  - ${oc.env:MANIFEST_DIR_NEW}/notsofar1-mdm_cutset_eval.jsonl.gz
  dev_cutsets:
  - ${oc.env:MANIFEST_DIR_NEW}/notsofar1-mdm_cutset_eval.jsonl.gz
  use_timestamps: true
  use_diar: true
  eval_diar_cutsets:
  - ${oc.env:EXPERIMENT_PATH}/notsofar1-mdm_cutset_eval.jsonl.gz
  dev_diar_cutsets:
  - ${oc.env:EXPERIMENT_PATH}/notsofar1-mdm_cutset_eval.jsonl.gz

training:
  decode_only: true
  eval_metrics_list: ["tcp_wer"]
  generation_num_beams: 1
  dataloader_num_workers: 2
  per_device_eval_batch_size: 16
  use_fddt: true

decoding:
  decoding_ctc_weight: 0
  condition_on_prev: false
  length_penalty: 1.0
EOF


# 2. Run decoding
/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/interactive_python  $SRC_ROOT/src/main.py +decode=$CONFIG_NAME

# 3. Create submission file
/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/interactive_python $SRC_ROOT/src/utils/generate_emma_submission.py \
 --hyp_dir=$SRC_ROOT/exp/$EXPERIMENT --out_path=$EXPERIMENT_PATH/${EXPERIMENT}_hyp.json





#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/../../configs/local_paths.sh
CONFIG_NAME=leaderboard/sc_real/DiCoW_v3_diarizen
CONFIG_PATH=configs/decode/${CONFIG_NAME}.yaml
export EXPERIMENT="DiCoW_v3_diarizen_greedy"
export DIAR_EXPERIMENT="DiariZen_base"
export EXPERIMENT_PATH="/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/baseline_sc"
echo $EXPERIMENT_PATH
echo "manifest: $MANIFEST_DIR_NEW"
DIAR_EXPERIMENT_PATH="${SRC_ROOT}/diar_exp/${DIAR_EXPERIMENT}"
#MODEL="BUT-FIT/diarizen-wavlm-large-s80-md"

# Define cutsets array
CUTSETS=("notsofar1-mdm_cutset_eval")   #notsofar1-sdm_cutset_eval_sc
#CUTSETS=("libri3mix_mix_clean_sc_test_cutset")

# Set up diarized cutsets path
export DIARIZED_CUTSETS_PATH="${DIAR_EXPERIMENT_PATH}/diarized_cutsets"
mkdir -p "$DIARIZED_CUTSETS_PATH"
mkdir -p /mnt/matylda5/qdeegen/deploy/forschung/NOTSOFAR1-Challenge/exp/$EXPERIMENT
mkdir -p /mnt/matylda5/qdeegen/deploy/forschung/NOTSOFAR1-Challenge/emma_hyp_files
INPUT=$EXPERIMENT_PATH/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/NOTSOFAR1/all_hyp.rttm

# Run diarization for each cutset
for CUTSET in "${CUTSETS[@]}"; do
    echo "Processing cutset: $CUTSET"

#    # Run diarization
#    "$SRC_ROOT/sge_tools/interactive_python_diarizen" "$SRC_ROOT/utils/diarizen_diar.py" \
#        --model="$MODEL" \
#        --input_cutset="${MANIFEST_DIR_NEW}/${CUTSET}.jsonl.gz" \
#        --output_dir="${DIAR_EXPERIMENT_PATH}/$CUTSET"

    # Prepare diarized cutset from RTTM directory
    python /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/split_rttms.py --input_path $INPUT
    /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/interactive_python_diarizen /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/prepare_diar_cutset_from_rttm_dir.py \
        --lhotse_manifest_path="${MANIFEST_DIR_NEW}/${CUTSET}.jsonl.gz" \
        --rttm_dir=$EXPERIMENT_PATH/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/NOTSOFAR1/split_rttm \
        --out_manifest_path="${EXPERIMENT_PATH}/${CUTSET}.jsonl.gz"
done


echo "Diarization completed for all cutsets"

# 1. Create decoding config
mkdir -p "$(dirname "$CONFIG_PATH")"
cat <<'EOF' > "$CONFIG_PATH"
# @package _global_
experiment: ${oc.env:EXPERIMENT}

wandb:
  project: EMMA_leaderboard

model:
  whisper_model: "BUT-FIT/DiCoW_v3"
  reinit_from: null

data:
  eval_cutsets:
  - ${oc.env:MANIFEST_DIR_NEW}notsofar1-mdm_cutset_eval.jsonl.gz
  dev_cutsets:
  - ${oc.env:MANIFEST_DIR_NEW}TESTER/notsofar1-mdm_cutset_eval.jsonl.gz
  use_timestamps: true
  use_diar: true
  eval_diar_cutsets:
  - ${oc.env:EXPERIMENT_PATH}/notsofar1-mdm_cutset_eval.jsonl.gz
  dev_diar_cutsets:
  - ${oc.env:EXPERIMENT_PATH}/notsofar1-mdm_cutset_eval.jsonl.gz

training:
  decode_only: true
  eval_metrics_list: ["tcp_wer"]
  generation_num_beams: 1
  dataloader_num_workers: 2
  per_device_eval_batch_size: 16
  use_fddt: true

decoding:
  decoding_ctc_weight: 0
  condition_on_prev: false
  length_penalty: 1.0
EOF


# 2. Run decoding
/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/interactive_python  /mnt/matylda5/qdeegen/deploy/forschung/NOTSOFAR1-Challenge/src/main.py +decode=$CONFIG_NAME

# 3. Create submission file
/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/diarizen/scoring/interactive_python /mnt/matylda5/qdeegen/deploy/forschung/NOTSOFAR1-Challenge/src/utils/generate_emma_submission.py \
 --hyp_dir=/mnt/matylda5/qdeegen/deploy/forschung/NOTSOFAR1-Challenge/exp/$EXPERIMENT --out_path=/mnt/matylda5/qdeegen/deploy/forschung/NOTSOFAR1-Challenge/emma_hyp_files/${EXPERIMENT}_hyp.json




