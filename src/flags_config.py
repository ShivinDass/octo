import os
from absl import flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

config_path = os.environ.get('CONFIG_PATH', "/mnt/xfs/home/alaakh/src/octo_dir/octo/scripts/configs/dm_finetune_config.py")

default_config_file = os.path.join(
    config_path    
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)