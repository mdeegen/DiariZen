#!/bin/bash

#SBATCH -t 30:00:00
#SBATCH --mem-per-cpu 8G
#SBATCH -J libriwasn_spatiospectral_{nickname}
#SBATCH --cpus-per-task 1
#SBATCH -A hpc-prf-nt2
#SBATCH -p normal
#SBATCH -n 61
#SBATCH --output {nickname}_eval_%j.out
#SBATCH --error {nickname}_eval_%j.err

srun python -m {main_python_path} with config.json

# gpu_ram=20G eigentlich
cd /mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc


SHORT_HOST=$(echo "$HOSTNAME" | cut -d. -f1)
GPU_NUM=$(echo "$SHORT_HOST" | grep -oE '[0-9]+$')
GPU_NUM=${GPU_NUM:-"unknown"}

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H")

LOG_DIR="logs/$DATE"
mkdir -p "$LOG_DIR"

LOGFILE="$LOG_DIR/run_gpu${GPU_NUM}_${TIME}_3.log"
echo "Logging to: $LOGFILE"

bash -i run_stage_mc.sh > "$LOGFILE" 2>&1
#bash -i run_stage_eval.sh > "$LOGFILE" 2>&1
#bash -i run_stage_eval_clustering_all_fine.sh > "$LOGFILE" 2>&1
#bash -i run_stage_eval_clustering_oracle.sh > "$LOGFILE" 2>&1
#bash -i run_stage_mc_train.sh > run.log 2>&1

