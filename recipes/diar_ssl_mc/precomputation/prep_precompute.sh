#!/bin/bash
#$ -l mem_free=10G
#$ -q all.q@@blade
#$ -N precompute_train
#$ -o global_log.out

scratch=/mnt/scratch/tmp/qdeegen
f_max_gcc=None
gcc_dir_all=$scratch/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcpsd_1024_new
#gcc_dir_all=$scratch/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcpsd_1024_320_wof
#gcc_dir_all=$scratch/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_no_freq_filt
#gcc_dir_all=$scratch/AMI_AIS_ALI_NSF_CHiME7/data/gccs/standard_gcc
# TODO: set STFT params accordingly
# TODO SET FMAX als parameter giving to the functions!
# TODO: TEST FILES HAVE THEIR OWN GCC PARAMS!!"!!!

recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc
export exp_root=$recipe_root/exp
data_dir=$recipe_root/data_no_chime
# =======================================
scp_train=$data_dir/train/wav.scp
uem_train=$data_dir/train/all.uem
scp_dev=$data_dir/dev/wav.scp
uem_dev=$data_dir/dev/all.uem
scp_test_nsf=$data_dir/test_marc/NOTSOFAR1/wav.scp
uem_test_nsf=$data_dir/test_marc/NOTSOFAR1/all.uem
#scp_test_ch7=$data_dir/test_marc/CHiME7/wav.scp
#uem_test_ch7=$data_dir/test_marc/CHiME7/all.uem
scp_test_ami=$data_dir/test_marc/AMI/wav.scp
uem_test_ami=$data_dir/test_marc/AMI/all.uem
scp_test_ali=$data_dir/test_marc/AliMeeting/wav.scp
uem_test_ali=$data_dir/test_marc/AliMeeting/all.uem
scp_test_ai=$data_dir/test_marc/AISHELL4/wav.scp
uem_test_ai=$data_dir/test_marc/AISHELL4/all.uem

split_dir=$scratch/split_dir
#if [ -d "$gcc_dir_all" ]; then
#    echo "Directory exists but index.json is missing — to delete it use : 'rm -rf $gcc_dir_all'"
#    exit 1
#else
#    echo "Directory does not exist — creating $gcc_dir_all."
#fi
#mkdir -p "$gcc_dir_all"

num_files=1
batch_size=100
mkdir -p "$split_dir/test"
echo $split_dir

# Dev und test data get the same dataset config params and processing, while trian is different (eg chunking)
cat  $scp_test_ami $scp_test_ali $scp_test_ai $scp_test_nsf > "$split_dir/test/combined.scp."  # $scp_test_ch7
cat  $uem_test_ami $uem_test_ali $uem_test_ai $uem_test_nsf > $split_dir/test/all.uem


logdir="$scratch/logs/"
[ -d "$logdir" ] && rm -r $logdir
mkdir -p "$logdir"

dataset=train
dataset_dir=$split_dir/$dataset
mkdir -p "$dataset_dir"
gcc_dir="$gcc_dir_all/$dataset"
mkdir -p "$gcc_dir"
echo $dataset_dir
cp "$scp_train" "$dataset_dir/wav.scp."
split -d -l $num_files --numeric-suffixes=0 --suffix-length=4 "$uem_train" "$dataset_dir/all.uem."
num_jobs=$(ls "$dataset_dir"/all.uem.* | wc -l)
echo $num_jobs

# Step 3: Maximal 40 gleichzeitig
start_train=1
batch_num_train=0
jobids_train=()

while [ $start_train -le $num_jobs ]; do
  end_train=$((start_train + batch_size - 1))
  if [ $end_train -gt $num_jobs ]; then
    end_train=$num_jobs
  fi
  batch_num_train=$((batch_num_train + 1))
  echo "Starte Batch $batch_num_train: Jobs $start_train-$end_train"

  hold_opt_train=""
  if [ ${#jobids_train[@]} -gt 0 ]; then
    # Warte auf den vorherigen Batch
    hold_opt_train="-hold_jid ${jobids_train[-1]}"
  fi

  jobid_train=$(qsub -terse -l ram_free=80G -q all.q@@blade -t ${start_train}-${end_train} \
            -cwd -V $hold_opt_train -N "run_train" \
            "$recipe_root/precomputation/run_precompute.sh" \
            "$scratch" "$gcc_dir" "$dataset" "$f_max_gcc")

  jobid_train=$(echo "$jobid_train" | cut -d. -f1)
  jobids_train+=("$jobid_train")

  start_train=$((end_train + 1))
done
# Step 4: Merge-Job, wartet auf alle Batches
all_jobids_train=$(IFS=, ; echo "${jobids_train[*]}")

echo "Warte auf: $all_jobids_train"
qsub -l mem_free=10G -q all.q@@blade \
     -hold_jid "$all_jobids_train" \
     -cwd -V -N merger \
     "$recipe_root/precomputation/merge_jobs.sh" "$gcc_dir"

##jobid1=$(qsub -terse -l mem_free=10G -q all.q@@blade -t 1-"$num_jobs" -cwd -V -N "run_train" "$recipe_root/precomputation/run_precompute.sh" "$scratch" "$gcc_dir" "$dataset")
##
##jobid1=$(echo "$jobid1" | cut -d. -f1)
##echo "Job-Array gestartet mit ID: $jobid1"
##qsub -l mem_free=5G -q all.q@@blade -hold_jid "$jobid1" -cwd -V -N merger "$recipe_root/precomputation/merge_jobs.sh" $gcc_dir
##
##
##


dataset=dev
dataset_dir=$split_dir/$dataset
mkdir -p "$dataset_dir"
gcc_dir="$gcc_dir_all/$dataset"
mkdir -p "$gcc_dir"
echo $dataset_dir
cp "$scp_dev" "$dataset_dir/wav.scp."
split -d -l $num_files --numeric-suffixes=0 --suffix-length=4 "$uem_dev" "$dataset_dir/all.uem."
num_jobs=$(ls "$dataset_dir"/all.uem.* | wc -l)
echo $num_jobs



# Step 3: Maximal 40 gleichzeitig
start_dev=1
batch_num_dev=0
jobids_dev=()

while [ $start_dev -le $num_jobs ]; do
  end_dev=$((start_dev + batch_size - 1))
  if [ $end_dev -gt $num_jobs ]; then
    end_dev=$num_jobs
  fi
  batch_num_dev=$((batch_num_dev + 1))
  echo "Starte Batch $batch_num_dev: Jobs $start_dev-$end_dev"

  hold_opt_dev=""
  if [ ${#jobids_dev[@]} -gt 0 ]; then
    # Warte auf den vorherigen Batch
    hold_opt_dev="-hold_jid ${jobids_dev[-1]}"
  fi

  jobid_dev=$(qsub -terse -l ram_free=80G -q all.q@@blade -t ${start_dev}-${end_dev} \
            -cwd -V $hold_opt_dev -N "run_dev" \
            "$recipe_root/precomputation/run_precompute.sh" \
            "$scratch" "$gcc_dir" "$dataset" "$f_max_gcc")

  jobid_dev=$(echo "$jobid_dev" | cut -d. -f1)
  jobids_dev+=("$jobid_dev")

  start_dev=$((end_dev + 1))
done
# Step 4: Merge-Job, wartet auf alle Batches
all_jobids_dev=$(IFS=, ; echo "${jobids_dev[*]}")

echo "Warte auf: $all_jobids_dev"
qsub -l mem_free=10G -q all.q@@blade \
     -hold_jid "$all_jobids_dev" \
     -cwd -V -N merger \
     "$recipe_root/precomputation/merge_jobs.sh" "$gcc_dir"

#
#jobid2=$(qsub -terse -l mem_free=10G -q all.q@@blade -t 1-"$num_jobs" -cwd -V -N "run_dev" "$recipe_root/precomputation/run_precompute.sh" "$scratch" "$gcc_dir" "$dataset")
#
#jobid2=$(echo "$jobid2" | cut -d. -f1)
#echo "Job-Array gestartet mit ID: $jobid2"
#qsub -l mem_free=10G -q all.q@@blade -hold_jid "$jobid2" -cwd -V -N merger "$recipe_root/precomputation/merge_jobs.sh" $gcc_dir



dataset="test"
dataset_dir=$split_dir/$dataset
mkdir -p "$dataset_dir"
gcc_dir="$gcc_dir_all/$dataset"
mkdir -p "$gcc_dir"
echo $dataset_dir

split -d -l $num_files --numeric-suffixes=0 --suffix-length=4 "$split_dir/test/combined.scp." "$dataset_dir/wav.scp."

num_jobs=$(ls "$dataset_dir"/wav.scp.* | wc -l)
echo $num_jobs


# Step 3: Maximal 40 gleichzeitig
start=1
batch_num=0
jobids=()

while [ $start -le $num_jobs ]; do
  end=$((start + batch_size - 1))
  if [ $end -gt $num_jobs ]; then
    end=$num_jobs
  fi

  batch_num=$((batch_num + 1))
  echo "Starte Batch $batch_num: Jobs $start-$end"

  hold_opt=""
  if [ ${#jobids[@]} -gt 0 ]; then
    # Warte auf den vorherigen Batch
    hold_opt="-hold_jid ${jobids[-1]}"
  fi
 # ram_free, mem_free, h_vmem
  jobid=$(qsub -terse -l ram_free=300G -q all.q@@blade -t ${start}-${end} \
            -cwd -V $hold_opt -N "run_test" \
            "$recipe_root/precomputation/run_precompute_test.sh" \
            "$scratch" "$gcc_dir" "$dataset" "$f_max_gcc")

  jobid=$(echo "$jobid" | cut -d. -f1)
  jobids+=("$jobid")

  start=$((end + 1))
done

# Step 4: Merge-Job, wartet auf alle Batches
all_jobids=$(IFS=, ; echo "${jobids[*]}")

echo "Warte auf: $all_jobids"
qsub -l mem_free=10G -q all.q@@blade \
     -hold_jid "$all_jobids" \
     -cwd -V -N merger \
     "$recipe_root/precomputation/merge_jobs.sh" "$gcc_dir"
#
##jobid3=$(qsub -terse -l mem_free=20G -q all.q@@blade -t 1-"$num_jobs" -cwd -V -N "run_test" "$recipe_root/precomputation/run_precompute_test.sh" "$scratch" "$gcc_dir" "$dataset")
##
##jobid3=$(echo "$jobid3" | cut -d. -f1)
##echo "Job-Array gestartet mit ID: $jobid3"
##qsub -l mem_free=10G -q all.q@@blade -hold_jid "$jobid3" -cwd -V -N merger "$recipe_root/precomputation/merge_jobs.sh" $gcc_dir
