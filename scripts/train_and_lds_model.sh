#!/usr/bin/bash

#SBATCH --job-name=robo_lds
#SBATCH --ntasks=1
#SBATCH --nodes=1

##SBATCH --cpus-per-task=128
##SBATCH --gres=gpu:a100:8

#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1

##SBATCH --array=0
#SBATCH --array=0-9
#SBATCH --requeue

#SBATCH --exclude=deep-chungus-[1-6]

#SBATCH --time=10-00:00:00
#SBATCH --output=/mnt/xfs/home/alaakh/src/octo_dir/octo/logs/lds_%A_%a.log

##SBATCH --partition=low-priority
##SBATCH --partition=background
#SBATCH --partition=high-priority

set -e

cd /mnt/xfs/home/alaakh/src/octo_dir/octo

# Declare the variable NUM_TRIALS
# NUM_TRIALS=10
NUM_TRIALS=10

folder_name="test_data_selection_10k"

# # Use a for loop to iterate over the range
# for ((job_id=0; job_id<NUM_TRIALS; job_id++)); do
# # for ((job_id=10; job_id<NUM_TRIALS; job_id++)); do
#     echo "Running job $job_id"
#     # Add your logic here for each job

#     # python src/do_vjp_replay.py \
#     python src/do_compute_lds_parallel.py \
#         --config.pretrained_path=hf://rail-berkeley/octo-small \
#         --config.job_id $job_id \
#         --config.folder_name $folder_name

# done

# job_id=5
job_id=$SLURM_ARRAY_TASK_ID
echo "Running job $job_id"
# Add your logic here for each job

# python src/do_vjp_replay.py \
python src/do_compute_lds_parallel.py \
    --config.pretrained_path=hf://rail-berkeley/octo-small \
    --config.job_id $job_id \
    --config.folder_name $folder_name