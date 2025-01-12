import imp
import os
import types
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from absl import app, flags
import flax
from ml_collections import config_flags, ConfigDict
import tensorflow as tf

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    check_config_diff,
    process_text,
)

from flatten_dict import flatten, unflatten
from ipdb import set_trace as bp
from ipdb import launch_ipdb_on_exception
from IPython import embed

from streaming import MDSWriter
from tqdm import tqdm

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from ipdb import set_trace as bp
from IPython import embed

import flags_config
FLAGS = flags.FLAGS

TRAIN=True
# TRAIN=False

columns = {
    'absolute_action_mask': 'ndarray:uint8',
    'action': 'ndarray:float32',
    'dataset_name': 'bytes',

    # 'index': 'ndarray:int64',
    'index': 'int64',

    'observation.image_primary': 'ndarray:uint8',
    'observation.image_wrist': 'ndarray:uint8',
    'observation.pad_mask': 'ndarray:uint8',
    'observation.pad_mask_dict.image_primary': 'ndarray:uint8',
    'observation.pad_mask_dict.image_wrist': 'ndarray:uint8',
    'observation.pad_mask_dict.proprio': 'ndarray:uint8',
    'observation.pad_mask_dict.timestep': 'ndarray:uint8',
    'observation.proprio': 'ndarray:float32',
    'observation.timestep': 'ndarray:int32',
    'task.language_instruction': 'bytes',

    # 'task.pad_mask_dict.language_instruction': 'ndarray:uint8'
    'task.pad_mask_dict.language_instruction': 'uint8'
}

target_dtypes = {
    k: v.replace('ndarray:', '')
    for k, v in columns.items()
}

def make_replay_dataset(train: bool=True):

    # initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    ############################################
    ############ get text processor ############
    ############################################
    pretrained_model = OctoModel.load_pretrained(
        FLAGS.config.pretrained_path,
        step=FLAGS.config.pretrained_step,
    )

    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    config = config.to_dict()
    check_config_diff(config, pretrained_model.config)

    if train:
        dataset_kwargs = FLAGS.config.dataset_kwargs
        dataset_kwargs['data_dir'] = '/mnt/xfs/home/alaakh/store/oxe/train_val_splits/train'
    else:
        dataset_kwargs = FLAGS.config.val_dataset_kwargs
        dataset_kwargs['data_dir'] = '/mnt/xfs/home/alaakh/store/oxe/train_val_splits/val'

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()
    ############################################

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    del pretrained_model

    # load standardize_fn from `path/to/file.py:fn_name` format
    if (
        standardize_fn := dataset_kwargs.get("standardize_fn", None)
    ) is not None:

        if isinstance(standardize_fn, str):
            path, name = standardize_fn.split(":")
            # imp is deprecated, but it's also what ml_collections uses
            standardize_fn = getattr(imp.load_source("standardize_fn", path), name)
            if train:
                del FLAGS.config["dataset_kwargs"]["standardize_fn"]
                FLAGS.config["dataset_kwargs"]["standardize_fn"] = standardize_fn
            else:
                del FLAGS.config["val_dataset_kwargs"]["standardize_fn"]
                FLAGS.config["val_dataset_kwargs"]["standardize_fn"] = standardize_fn

        elif isinstance(standardize_fn, types.FunctionType):
            standardize_fn = dataset_kwargs["standardize_fn"]

        else:
            raise ValueError

    ############################################
    ########## get tfrecords iterator ##########
    ############################################

    print('Creating dataset')
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(FLAGS.config.seed)
    # create dataset object
    dataset = make_single_dataset(
        # FLAGS.config.dataset_kwargs,
        dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=train,
        shuffle=False,
        num_parallel_calls=1,
        num_parallel_reads=1,
    )
    # dataset_statistics = dataset.dataset_statistics
    # dataset = dataset.cache()
    # dataset.dataset_statistics = dataset_statistics

    return dataset

def main(_):

    ds_name = FLAGS.config.dataset_kwargs.name
    # ds_path = FLAGS.config.dataset_kwargs.data_dir
    if TRAIN:
        ds_path = '/mnt/xfs/home/alaakh/store/oxe/train_val_splits/train'
    else:
        ds_path = '/mnt/xfs/home/alaakh/store/oxe/train_val_splits/val'

    print('dataset:', ds_name)
    print('path:', ds_path)

    dataset = make_replay_dataset(train=TRAIN)
    dataset_statistics = dataset.dataset_statistics
    dataset = dataset.unbatch().iterator()

    # i = 0
    # columns = {}
    # for item in dataset:
    #     item_flat = flatten(item, 'dot')
    #     for k, v in item_flat.items():
    #         try:
    #             columns[k] = f'ndarray:{v.numpy().dtype}'
    #             print(k, ':', v.numpy().dtype, 'and', v.numpy().shape)
    #         except:
    #             if type(v) != bytes:
    #                 columns[k] = f'ndarray:{v.dtype}'
    #                 print(k, ':', v.dtype, 'and', v.shape)
    #     print()
    #     if i > 5:
    #         break
    #     i += 1

    # bp()

    # from pprint import pprint
    # pprint(columns)

    target_size = 150 * 1024 * 1024 # 130MB
    item_size = 640 # 640B
    num_items = np.ceil(target_size / item_size)
    shard_size = int(num_items * item_size)

    if TRAIN:
        out_root = "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/train"
    else:
        out_root = "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset/val"

    out_path = os.path.join(out_root, ds_name)
    os.makedirs(out_path, exist_ok=True)

    with MDSWriter(columns=columns,
                out=out_path,
                size_limit=shard_size) as out:

        i = 0
        # for item in tqdm(dataset, total=dataset.cardinality().numpy()):
        for item in tqdm(dataset):
        # for sample in loader:
            # out.write({'tokens': sample['tokens'][0]})
            item_flat = {}
            for k, v in flatten(item, 'dot').items():
                if k in ['dataset_name', 'task.language_instruction']:
                    # item_flat[k] = pickle.dumps(v.numpy())
                    item_flat[k] = v
                else:
                    # item_flat[k] = v.numpy().astype(target_dtypes[k])
                    item_flat[k] = v.astype(target_dtypes[k])

            out.write(item_flat)

            i += 1

            # if i > 100:
            #     break

    def numpy_to_list(obj):
        if isinstance(obj, dict):
            return {k: numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    dataset_statistics = numpy_to_list(dataset_statistics)
    json_path = os.path.join(out_path, 'dataset_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_statistics, f, indent=4)

if __name__ == "__main__":
    app.run(main)
