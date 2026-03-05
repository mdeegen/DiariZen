#!/bin/bash


# write me a bash script that takes the following python call and executes it for the number 23, 560 and 760
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



out_dir="/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/precomputation/logs/run_$SGE_TASK_ID.log"

recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/precomputation
cd $recipe_root

echo "Running task ID: $SGE_TASK_ID"
echo $out_dir
if [[ $SGE_TASK_ID == 1 ]]; then
    echo AMI
#    micromamba activate diarizen && python /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/precomputation/precompute.py --config /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/conf/gcc_encoder.toml \
#    --scp_file /mnt/scratch/tmp/qdeegen/split_dir/train/wav.scp. --uem_file /mnt/scratch/tmp/qdeegen/split_dir/train/all.uem.0022 \
#    --out_dir /mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_no_freq_filt/train2 --dataset train --f_max_gcc None &>> "$out_dir" 2>&1

  elif [[ $SGE_TASK_ID == 2 ]]; then
    echo NSF
    micromamba activate diarizen && python precompute.py --config /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/conf/gcc_encoder.toml \
    --scp_file /mnt/scratch/tmp/qdeegen/split_dir/train/wav.scp. --uem_file /mnt/scratch/tmp/qdeegen/split_dir/train/all.uem.0560 \
    --out_dir /mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_no_freq_filt/train2 --dataset train --f_max_gcc None &>> "$out_dir" 2>&1

  elif [[ $SGE_TASK_ID == 3 ]]; then
    echo AISHELL
#    micromamba activate diarizen && python precompute.py --config /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/conf/gcc_encoder.toml \
#    --scp_file /mnt/scratch/tmp/qdeegen/split_dir/train/wav.scp. --uem_file /mnt/scratch/tmp/qdeegen/split_dir/train/all.uem.0300 \
#    --out_dir /mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_no_freq_filt/train2 --dataset train --f_max_gcc None &>> "$out_dir" 2>&1

  elif [[ $SGE_TASK_ID == 4 ]]; then
    echo Alimeeting
#    micromamba activate diarizen && python precompute.py --config /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/conf/gcc_encoder.toml \
#    --scp_file /mnt/scratch/tmp/qdeegen/split_dir/train/wav.scp. --uem_file /mnt/scratch/tmp/qdeegen/split_dir/train/all.uem.0340 \
#    --out_dir /mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_no_freq_filt/train2 --dataset train --f_max_gcc None &>> "$out_dir" 2>&1
fi