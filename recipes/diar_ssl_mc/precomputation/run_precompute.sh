#!/bin/bash
scratch=$1
gcc_dir=$2
dataset=$3
f_max_gcc=$4
#$ -o /mnt/scratch/tmp/qdeegen/logs/output_$TASK_ID.log
#$ -e /mnt/scratch/tmp/qdeegen/logs/error_$TASK_ID.log

recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc
cd $recipe_root/precomputation
export exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf
export train_conf=$conf_dir/gcc_encoder.toml
split_dir=$scratch/split_dir/$dataset
# =======================================
task_id=$SGE_TASK_ID
export SGE_TASK_ID=$task_id
out_dir="$scratch/logs/new_output_$SGE_TASK_ID.log"

JOB_ID=$(printf "%04d" $((SGE_TASK_ID - 1)))
echo "Running job $JOB_ID...Running task ID: $task_id"
scp_file="$split_dir/wav.scp."
uem_file="$split_dir/all.uem.$JOB_ID"
echo $scp_file
echo $uem_file
echo $out_dir

# >>> micromamba setup (EXPLIZIT) <<<
export MAMBA_EXE='/homes/eva/q/qdeegen/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/homes/eva/q/qdeegen/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
echo "starting gccs:"
echo $gcc_dir


micromamba activate diarizen && python precompute.py --config $train_conf --scp_file "$scp_file" --uem_file "$uem_file" --out_dir "$gcc_dir" --dataset "$dataset" --f_max_gcc $f_max_gcc &>> "$out_dir" 2>&1

#python precompute.py --config /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/conf/gcc_encoder.toml --scp_file /mnt/scratch/tmp/qdeegen/split_dir/train/wav.scp. --uem_file /mnt/scratch/tmp/qdeegen/split_dir/train/all.uem.0022 --out_dir /mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_no_freq_filt/train --dataset train --f_max_gcc None &>> /mnt/scratch/tmp/qdeegen/logs/new_output_23.log 2>&1
