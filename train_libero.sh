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

        action_chunks=8
        # task_name=kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_h8_prechunk
        # task_shorthand=bowl-cabinet
        
        # task_name=living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_h8_prechunk
        # task_shorthand=mug-mug
        
        task_name=study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_h8_prechunk
        task_shorthand=book-caddy
        
        # task_name=study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_h8_prechunk
        # task_shorthand=book-caddy

        # task_name=kitchen_scene8_put_both_moka_pots_on_the_stove_h8_prechunk
        # task_shorthand=moka-moka
        # ITER=10
        for ITER in 30; do
            TASK_PATH=libero90_horizon30/${task_shorthand}_iter${ITER}_top0.1.tfrecord
            python scripts/finetune_retrieved.py \
            --config scripts/configs/retrieved_finetune_config/retrieved_finetune_libero_co_training_config.py:${task_name},${TASK_PATH} \
            --config.action_chunks=$action_chunks \
            --config.seed $seed \
            --config.save_dir /mnt/hdd2/libero/experiments/${task_shorthand}_dm/  \
            --config.pretrained_path=hf://rail-berkeley/octo-small --name seed${seed}_libero_${task_shorthand}_h1_dm0.1-training_w30_n${ITER} \
            # --debug true \
        done

    done
done