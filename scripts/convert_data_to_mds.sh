# set -e

# DATA_NAME=libero_val_first5_pnp
# for FILE in /home/shivin/tensorflow_datasets/${DATA_NAME}/*; do
#     echo $FILE
#     python src/convert_mds.py $FILE --config scripts/configs/finetune_base_libero_config.py \
#     --config.dataset_kwargs.data_dir /home/shivin/tensorflow_datasets/${DATA_NAME} \

# done
# python src/convert_mds_cluster.py --config scripts/configs/finetune_base_libero_config.py


# real
# python src/convert_mds_cluster.py --config scripts/configs/finetune_base_easy_pick_config.py
# python src/convert_mds_cluster.py --config scripts/configs/finetune_base_deo_in_pouch_config.py
# python src/convert_mds_cluster.py --config scripts/configs/finetune_base_tiago_sink_config.py

python src/convert_mds_cluster.py --config scripts/configs/finetune_base_droid_config.py