#!/bin/bash
#$ -l mem_free=10G
#$ -q all.q@@blade
#$ -N precompute_train
#$ -o global_log.out
scratch=/mnt/scratch/tmp/qdeegen


jobid=1001
jobid_1=1002
session="20200616_M_R001S01C01"
dset="dev"
gcc_dir_all=$scratch/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size4096_shift311_2 #no_freq_filt
f_max_gcc=3500
# TODO: set STFT params accordingly


split_dir=$scratch/split_dir
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


#if [ -d "$gcc_dir_all" ]; then
#    echo "Directory exists but index.json is missing — to delete it use : 'rm -rf $gcc_dir_all'"
#    exit 1
#else
#    echo "Directory does not exist — creating $gcc_dir_all."
#fi
#mkdir -p "$gcc_dir_all"

num_files=1
batch_size=30
mkdir -p "$split_dir/test"
echo $split_dir

# Dev und test data get the same dataset config params and processing, while trian is different (eg chunking)
cat  $scp_test_ami $scp_test_ali $scp_test_ai $scp_test_nsf > "$split_dir/test/combined.scp."  # $scp_test_ch7
cat  $uem_test_ami $uem_test_ali $uem_test_ai $uem_test_nsf > $split_dir/test/all.uem



logdir="$scratch/logs/"

if [ "$dset" == "train" ]; then
  dataset=train
  dataset_dir=$split_dir/$dataset
  gcc_dir="$gcc_dir_all/$dataset"
  echo $dataset_dir
  cp "$scp_train" "$dataset_dir/wav.scp."
  #split -d -l $num_files --numeric-suffixes=0 --suffix-length=4 "$uem_train" "$dataset_dir/all.uem."

  # schreibe mir eine all uem zeile für nur die session die oben im file bestimmt ist, lade dafür die entsprechenden zeilen aus dem $uem_train file
  grep "$session" "$uem_train" > "$dataset_dir/all.uem.$jobid"
  #split -d -l $num_files --numeric-suffixes=0 --suffix-length=4 "$dataset_dir/all.uem." "$dataset_dir/all.uem."


  jobid_train=$(qsub -terse -l mem_free=10G -q all.q@@blade -t ${jobid_1}-${jobid_1} \
            -cwd -V -N "run_train" \
            "$recipe_root/precomputation/run_precompute.sh" \
            "$scratch" "$gcc_dir" "$dataset" "$f_max_gcc")

  jobid_train=$(echo "$jobid_train" | cut -d. -f1)
  echo "Warte auf: $jobid_train"
  qsub -l mem_free=10G -q all.q@@blade \
      -hold_jid "$jobid_train" \
       -cwd -V -N merger \
       "$recipe_root/precomputation/merge_jobs.sh" "$gcc_dir"
fi


if [ "$dset" == "dev" ]; then
  dataset=dev
  dataset_dir=$split_dir/$dataset
  gcc_dir="$gcc_dir_all/$dataset"
  echo $dataset_dir
  cp "$scp_dev" "$dataset_dir/wav.scp."
  grep "$session" "$uem_dev" > "$dataset_dir/all.uem.$jobid"
#  split -d -l $num_files --numeric-suffixes=0 --suffix-length=4 "$uem_dev" "$dataset_dir/all.uem."

  jobid_dev=$(qsub -terse -l mem_free=10G -q all.q@@blade -t ${jobid_1}-${jobid_1} \
            -cwd -V -N "run_dev" \
            "$recipe_root/precomputation/run_precompute.sh" \
            "$scratch" "$gcc_dir" "$dataset" $f_max_gcc)
  jobid_dev=$(echo "$jobid_dev" | cut -d. -f1)
  echo "Warte auf: $jobid_dev"
  qsub -l mem_free=10G -q all.q@@blade \
       -hold_jid "$jobid_dev" \
       -cwd -V -N merger \
       "$recipe_root/precomputation/merge_jobs.sh" "$gcc_dir"
fi


# TODO: check test
if [ "$dset" == "test" ]; then
  dataset="test"
  dataset_dir=$split_dir/$dataset
  gcc_dir="$gcc_dir_all/$dataset"
  echo $dataset_dir
  grep "$session" "$uem_dev" > "$dataset_dir/all.uem.$jobid"

#  split -d -l $num_files --numeric-suffixes=0 --suffix-length=4 "$split_dir/test/combined.scp." "$dataset_dir/wav.scp."

  jobid_test=$(qsub -terse -l mem_free=50G -q all.q@@blade -t ${jobid_1}-${jobid_1} \
            -cwd -V -N "run_test" \
            "$recipe_root/precomputation/run_precompute_test.sh" \
            "$scratch" "$gcc_dir" "$dataset")

  jobid_test=$(echo "$jobid_test" | cut -d. -f1)
  echo "Warte auf: $jobid_test"
  qsub -l mem_free=10G -q all.q@@blade \
       -hold_jid "$jobid_test" \
       -cwd -V -N merger \
       "$recipe_root/precomputation/merge_jobs.sh" "$gcc_dir"
fi