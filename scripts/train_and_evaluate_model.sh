#!/usr/bin/bash

#SBATCH --job-name=fast
#SBATCH --ntasks=1
#SBATCH --nodes=1

#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1

##SBATCH --array=0
#SBATCH --array=0-9
#SBATCH --requeue

##SBATCH --exclude=deep-chungus-[7-11]

#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/xfs/home/alaakh/src/octo_dir/octo/logs/octo_eval_%A_%a.log

##SBATCH --partition=low-priority
##SBATCH --partition=background
#SBATCH --partition=high-priority

set -e

cd /mnt/xfs/home/alaakh/src/octo_dir/octo

# Declare the variable NUM_TRIALS
NUM_TRIALS=10

folder_name="test_data_selection_slow_more_iters"

# # Use a for loop to iterate over the range
# for ((job_id=0; job_id<NUM_TRIALS; job_id++)); do

#     python src/do_data_selection_parallel.py \
#         --config.pretrained_path=hf://rail-berkeley/octo-small \
#         --config.job_id $job_id \
#         --config.folder_name $folder_name

#     ckpt_path="/mnt/xfs/home/alaakh/src/octo_dir/octo/exps/debug/bridge_dataset/$folder_name/iter_$job_id"

#     bash /mnt/xfs/home/alaakh/src/octo_dir/octo/scripts/maniskill3_octo_bridge.sh $ckpt_path

# done

job_id=$SLURM_ARRAY_TASK_ID

# python src/do_data_selection_parallel.py \
#     --config.pretrained_path=hf://rail-berkeley/octo-small \
#     --config.job_id $job_id \
#     --config.folder_name $folder_name

ckpt_path="/mnt/xfs/home/alaakh/src/octo_dir/octo/exps/debug/bridge_dataset/$folder_name/iter_$job_id"

bash /mnt/xfs/home/alaakh/src/octo_dir/octo/scripts/maniskill3_octo_bridge.sh $ckpt_path