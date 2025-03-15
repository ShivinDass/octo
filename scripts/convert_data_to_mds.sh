# set -e
# for FILE in /home/shivin/tensorflow_datasets/libero_val_selected_demos/*; do
#     echo $FILE
#     python src/convert_mds.py $FILE --config scripts/configs/finetune_base_libero_config.py
# done
python src/convert_mds_cluster.py --config scripts/configs/finetune_base_libero_config.py
# python src/convert_mds_cluster.py --config scripts/configs/finetune_base_easy_pick_config.py