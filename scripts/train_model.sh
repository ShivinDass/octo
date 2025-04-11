#!/usr/bin/bash

#SBATCH --job-name=robofast_t_none
#SBATCH --ntasks=1
#SBATCH --nodes=1

#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:8

#SBATCH --array=0
##SBATCH --array=0-26
##SBATCH --array=0-53
#SBATCH --requeue

#SBATCH --exclude=deep-chungus-[1-6]

#SBATCH --time=10-00:00:00
#SBATCH --output=/mnt/xfs/home/alaakh/src/octo_dir/octo/logs/train_%A_%a.log

##SBATCH --partition=low-priority
##SBATCH --partition=background
#SBATCH --partition=high-priority

set -e

cd /mnt/xfs/home/alaakh/src/octo_dir/octo

# Declare the variable NUM_TRIALS
# NUM_TRIALS=10
# NUM_TRIALS=10
NUM_TRIALS=50

# folder_name="test_data_selection_10k"
# folder_name="test_data_selection_3800_12"

# folder_name="test_data_selection_3800_5_10"
# folder_name="test_data_selection_3800_125_20"
# folder_name="test_data_selection_real_data_3800_125_20"

# folder_name="test_data_selection_real_data_ut_3800_125_20"

# 0- folder_name="test_data_selection_real_data_ut_and_real_2000_125_20"
# 1- folder_name="test_data_selection_real_data_dist_ut_and_real_2000_125_20"
# 2- folder_name="test_data_selection_real_data_dist_ut_and_real_4000_125_20"
# 3- folder_name="test_data_selection_real_data_dist_ut_and_real_correct_perc_4000_125_20"

# folder_name="real_data_dist_ut_and_real_correct_perc_4000_125_20_new_data"
# folder_name="real_data_dist_ut_and_real_4000_125_20_new_data"

# folder_name="half_octo_1000_its_125_128p"
folder_name="all_octo_5000_its_125_128p"

# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Use a for loop to iterate over the range
for ((job_id=0; job_id<NUM_TRIALS; job_id++)); do
# for ((job_id=1; job_id<NUM_TRIALS; job_id++)); do
    echo "Running job $job_id"
    # Add your logic here for each job

    # if [[ "$job_id" -ne 1 ]]; then
        # python src/do_vjp_replay.py \
    python src/do_data_selection_parallel.py \
        --config.pretrained_path=hf://rail-berkeley/octo-small \
        --config.job_id $job_id \
        --config.folder_name $folder_name

    # fi

    # ckpt_path="/mnt/xfs/home/alaakh/src/octo_dir/octo/exps/debug/bridge_dataset/$folder_name/iter_$job_id"
    # bash /mnt/xfs/home/alaakh/src/octo_dir/octo/scripts/maniskill3_octo_bridge.sh $ckpt_path

    # python src/do_compute_lds_parallel.py \
    #     --config.pretrained_path=hf://rail-berkeley/octo-small \
    #     --config.job_id $job_id \
    #     --config.folder_name $folder_name

done
