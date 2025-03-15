set -e

NUM_TRIALS=100

# FOLDER="book-caddy_c0.1_bob60"
# FOLDER="book-caddy"
# FOLDER="bowl-cabinet"
FOLDER="mug-mug"
# FOLDER="moka-moka"

# DATASET="libero90_128x128"
DATASET="libero90_horizon30_128x128"

# TASK="study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
# TASK="kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
TASK="living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
# TASK="kitchen_scene8_put_both_moka_pots_on_the_stove"

# Use a for loop to iterate over the range
for ((job_id=0; job_id<NUM_TRIALS; job_id++)); do
    echo "Running job $job_id"
    # Add your logic here for each job

    python src/do_data_selection_parallel.py \
        --config scripts/configs/finetune_dm_libero_config.py \
        --config.pretrained_path=hf://rail-berkeley/octo-small \
        --config.job_id $job_id \
        --config.folder_name $FOLDER \
        --config.val_dataset_kwargs.name $TASK \
        --config.dataset_kwargs.name $DATASET \
        # --config.candidate_size 0.2 \
        # --config.bob_steps 60 
        # --config.loss_type "l1"

done

