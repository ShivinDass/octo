
set -e

declare -a task_names=(
living_room_scene2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_h8_prechunk
# living_room_scene2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_h8_prechunk
# kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_h8_prechunk
# kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_h8_prechunk
# living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_h8_prechunk
# study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_h8_prechunk
# living_room_scene6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_h8_prechunk
# living_room_scene1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_h8_prechunk
# kitchen_scene8_put_both_moka_pots_on_the_stove_h8_prechunk
# kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_h8_prechunk
)
declare -a task_shorthands=(
soup-sauce
# cream-butter
# stove-moka
# bowl-cabinet
# mug-mug
# book-caddy
# mug-pudding
# soup-cheese
# moka-moka
# mug-microwave
)

for i in "${!task_names[@]}";
do
    task_name=${task_names[$i]}
    task_shorthand=${task_shorthands[$i]}
    for seed in 0 1 2 3 4; do
        # action_chunks=8
        # python scripts/finetune_retrieved.py \
        # --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_target_only_config.py:${task_name} \
        # --config.seed $seed \
        # --config.action_chunks=$action_chunks \
        # --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}/  \
        # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_target-only_chunk${action_chunks}

        # method=language
        # percent=7563
        # action_chunks=8
        # python scripts/finetune_retrieved.py \
        # --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_config.py:${method},${percent},${task_name} \
        # --config.action_chunks=$action_chunks \
        # --config.seed $seed \
        # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_${method}_${percent}_chunk${action_chunks}

        # method=flow
        # percent=0.1
        # action_chunks=8
        # python scripts/finetune_retrieved.py \
        # --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_config.py:${method},${percent},${task_name} \
        # --config.action_chunks=$action_chunks \
        # --config.seed $seed \
        # --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}/  \
        # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_${method}_${percent}_chunk${action_chunks}

        # method=br
        # percent=0.1
        # action_chunks=8
        # python scripts/finetune_retrieved.py \
        # --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_config.py:${method},${percent},${task_name} \
        # --config.action_chunks=$action_chunks \
        # --config.seed $seed \
        # --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}/  \
        # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_${method}_${percent}_chunk${action_chunks}

        # method=action
        # percent=0.1
        # action_chunks=8
        # python scripts/finetune_retrieved.py \
        # --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_config.py:${method},${percent},${task_name} \
        # --config.action_chunks=$action_chunks \
        # --config.seed $seed \
        # --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}/  \
        # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_${method}_${percent}_chunk${action_chunks}

        # for j in 0 1 2; do
        #     action_chunks=8
        #     python scripts/finetune_retrieved.py \
        #     --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_co_training_config.py:${task_name},${j} \
        #     --config.action_chunks=$action_chunks \
        #     --config.seed $seed \
        #     --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}/  \
        #     --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_random${j}_chunk${action_chunks}
        #     # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_co-training_chunk${action_chunks}
        # done

        action_chunks=8
        # task_name=kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_h8_prechunk
        # task_shorthand=bowl-cabinet
        # TASK_PATH=libero90_horizon30/${task_shorthand}_iter20_top0.1.tfrecord
        # python scripts/finetune_retrieved.py \
        # --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_co_training_config.py:${task_name},${TASK_PATH} \
        # --config.action_chunks=$action_chunks \
        # --config.seed $seed \
        # --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}_dm/  \
        # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_dm0.1-training_w30_n20
        
        task_name=study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_h8_prechunk
        task_shorthand=book-caddy
        # ITER=10
        for ITER in 100; do
            TASK_PATH=libero90_horizon30/${task_shorthand}_iter${ITER}_top0.1.tfrecord
            python scripts/finetune_retrieved.py \
            --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_co_training_config.py:${task_name},${TASK_PATH} \
            --config.action_chunks=$action_chunks \
            --config.seed $seed \
            --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}_dm/  \
            --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_dm0.1-training_w100_top-bottom_n${ITER}
        done

        task_name=kitchen_scene8_put_both_moka_pots_on_the_stove_h8_prechunk
        task_shorthand=moka-moka
        # ITER=10
        for ITER in 30; do
            TASK_PATH=libero90/${task_shorthand}_iter${ITER}_top0.1.tfrecord
            python scripts/finetune_retrieved.py \
            --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_co_training_config.py:${task_name},${TASK_PATH} \
            --config.action_chunks=$action_chunks \
            --config.seed $seed \
            --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}_dm/  \
            --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_dm0.1-training_traj_top-bottom_n${ITER}
        done

        # action_chunks=8
        # python scripts/finetune_retrieved.py \
        # --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_co_training_config.py:${task_name},${task_shorthand},0.01 \
        # --config.action_chunks=$action_chunks \
        # --config.seed $seed \
        # --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}/  \
        # --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_dm0.01-training_chunk${action_chunks}

    done
done
# action_chunks=8
# python scripts/finetune_retrieved.py \
# --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero10_config.py \
# --config.action_chunks=$action_chunks \
# --config.pretrained_path=hf://rail-berkeley/octo-small --name libero10_${task_shorthand}_h1_chunk${action_chunks}
