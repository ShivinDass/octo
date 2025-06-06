#!/bin/bash

set -e

declare -a task_names=(
# kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_chunk8_prechunk
living_room_scene2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_chunk8_prechunk
living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_chunk8_prechunk
)

declare -a task_shorthands=(
# bowl-cabinet_weighted
soup-sauce_weighted
mug-mug_weighted
)

for i in "${!task_names[@]}";
do
    task_name=${task_names[$i]}
    task_shorthand=${task_shorthands[$i]}
    for seed in 0 1 2 3 4; do

        action_chunks=8
        HORIZON=15
        for ITER in 1 5 10 20; do
            TASK_PATH=libero90_horizon15_ablation/${task_shorthand}_iter${ITER}_top0.1.tfrecord
            # TASK_PATH=libero_w15_with_subopt/${task_shorthand}_iter${ITER}_top0.05.tfrecord
            python scripts/finetune_retrieved.py \
            --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_co_training_config.py:${task_name},${TASK_PATH} \
            --config.action_chunks=$action_chunks \
            --config.seed $seed \
            --config.save_dir /mnt/hdd2/libero/experiments/ablations/${task_shorthand}_dm/  \
            --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_dm0.1-training_w${HORIZON}_n${ITER} \
            --debug true \

        done
    done
done