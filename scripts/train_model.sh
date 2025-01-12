#!/usr/bin/bash

#SBATCH --job-name=robofast_t_none
#SBATCH --ntasks=1
#SBATCH --nodes=1

#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1

##SBATCH --array=0
##SBATCH --array=0-26
#SBATCH --array=0-53
#SBATCH --requeue

#SBATCH --exclude=deep-chungus-[7-11]

#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/xfs/home/alaakh/src/dmil/logs/metaworld/traj_succ_dm_%A_%a.log

#SBATCH --partition=low-priority
##SBATCH --partition=background
##SBATCH --partition=high-priority

cd /mnt/xfs/home/alaakh/src/octo_dir/octo

python src/do_vjp_replay.py \
    --config.pretrained_path=hf://rail-berkeley/octo-small