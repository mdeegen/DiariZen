#!/bin/bash
#$ -l mem_free=10G
#$ -q all.q@@blade
#$ -N precompute_train
#$ -o global_log.out

recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc
export exp_root=$recipe_root/exp
data_dir=$recipe_root/data_mc
# =======================================
subset=train
scp_file_whole=$data_dir/$subset/wav.scp
uem_file_whole=$data_dir/$subset/all.uem

split_dir=$recipe_root/gcc_data/$subset
num_files=1
mkdir -p "$split_dir"
echo $split_dir

cat file1.scp file2.scp file3.scp > combined.scp
cat file1.uem file2.uem file3.uem > combined.uem



cp "$scp_file_whole" "$split_dir/wav.scp."
split -d -l $num_files --numeric-suffixes=0 --suffix-length=3 "$uem_file_whole" "$split_dir/all.uem."

num_jobs=$(ls "$split_dir"/all.uem.* | wc -l)
#qsub -t 1-"$num_jobs" -cwd -V -N precompute_train_array "$recipe_root/precomputation/run_precompute_train.sh"

echo $num_jobs
jobs_path="$recipe_root/precomputation/run_precompute_$subset.sh"
logdir="$recipe_root/precomputation/logs/"
#recipe_root/precomputation/manage_task.sh -sync yes -l ram_free=50G,mem_free=50G -q all.q@@blade $jobs_path &> $out_path
[ -d "$logdir" ] && rm -r $logdir
mkdir -p "$logdir"
#ERRFILE=/home/USER/somedir/errors
#rm run.log
qsub -sync yes -l mem_free=10G -q all.q@@blade -t 1-"$num_jobs" -cwd -V -N precompute_train_array "$recipe_root/precomputation/run_precompute_train.sh" &> "run.log"

echo "Alljobs finished, start Merging..."

micromamba activate diarizen && python merge_index.py --subset "$subset"