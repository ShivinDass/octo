#!/usr/bin/bash

#SBATCH --job-name=simpler_eval
#SBATCH --ntasks=1
#SBATCH --nodes=1

##SBATCH --cpus-per-task=5
##SBATCH --gres=gpu:a100:1

#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:a100:8

##SBATCH --array=0
#SBATCH --requeue

#SBATCH --exclude=deep-chungus-[7-11]

#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/xfs/home/alaakh/src/octo_dir/SimplerEnv/logs/eval_%A_%a.log

##SBATCH --partition=low-priority
##SBATCH --partition=background
#SBATCH --partition=high-priority

set -e

cd /mnt/xfs/home/alaakh/src/octo_dir/SimplerEnv/simpler_evals

declare -a policy_models=(
"octo-small"
# "octo-base"
# "octo-small-1.5"
# "octo-base-1.5"
)

declare -a ckpt_path=(
# "/mnt/xfs/home/alaakh/src/octo_dir/octo/exps/debug/bridge_dataset/test_data_selection_slow/iter_2"
$1
)

# for init_rng in 0 102 104;
for init_rng in 0 1 2 3 4 5 6 7;
do
for i in "${!policy_models[@]}";
do
    # XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py --model ${policy_models[$i]} -e "PutEggplantInBasketScene-v1" \
    # -s ${init_rng} --num-episodes 100 --num-envs 10 --ckpt_path ${ckpt_path[$i]}

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python real2sim_eval_maniskill3.py \
            --model ${policy_models[$i]} \
            -e "PutCarrotOnPlateInScene-v1" \
            -s ${init_rng} \
            --num-episodes 100 \
            --num-envs 10 \
            --ckpt_path ${ckpt_path[$i]} \
            --record_dir ${ckpt_path[$i]}/videos

    # XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py --model ${policy_models[$i]} -e "StackGreenCubeOnYellowCubeBakedTexInScene-v1" \
    # -s ${init_rng} --num-episodes 100 --num-envs 10 --ckpt_path ${ckpt_path[$i]}

    # XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py --model ${policy_models[$i]} -e "PutSpoonOnTableClothInScene-v1" \
    # -s ${init_rng} --num-episodes 100 --num-envs 10 --ckpt_path ${ckpt_path[$i]}
done
done

python calc_avg_success_rate.py \
    --path ${ckpt_path[$i]}/videos/real2sim_eval

# XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/real2sim_eval_maniskill3.py --model ${policy_models[$i]} -e "StackGreenCubeOnYellowCubeBakedTexInScene-v1" \
#     -s ${init_rng} --num-episodes 100 --num-envs 10 --ckpt_path ${ckpt_path[$i]}