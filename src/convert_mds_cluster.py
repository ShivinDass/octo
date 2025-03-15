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
from pathlib import Path
from copy import deepcopy


from flatten_dict import flatten, unflatten
# from ipdb import set_trace as bp
# from ipdb import launch_ipdb_on_exception
# from IPython import embed

from streaming import MDSWriter
from tqdm import tqdm

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

# from ipdb import set_trace as bp
# from IPython import embed

import flags_config
FLAGS = flags.FLAGS

TRAIN=True

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
    # pretrained_model = OctoModel.load_pretrained(
    #     FLAGS.config.pretrained_path,
    #     step=FLAGS.config.pretrained_step,
    # )

    # flat_config = flax.traverse_util.flatten_dict(
    #     pretrained_model.config, keep_empty_nodes=True
    # )
    # for d_key in flax.traverse_util.flatten_dict(
    #     FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    # ):
    #     for c_key in list(flat_config.keys()):
    #         if ".".join(c_key).startswith(".".join(d_key)):
    #             del flat_config[c_key]

    # config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    # config.update(FLAGS.config.get("update_config", ConfigDict()))
    # config = config.to_dict()
    # check_config_diff(config, pretrained_model.config)

    if train:
        dataset_kwargs = FLAGS.config.dataset_kwargs
        # dataset_kwargs['data_dir'] = '/mnt/xfs/home/alaakh/store/oxe/train_val_splits/train'
    else:
        dataset_kwargs = FLAGS.config.val_dataset_kwargs
        # dataset_kwargs['data_dir'] = '/mnt/xfs/home/alaakh/store/oxe/train_val_splits/val'

    # create text processor
    # if config["text_processor"] is None:
    #     text_processor = None
    # else:
    #     text_processor = ModuleSpec.instantiate(config["text_processor"])()
    ############################################

    # def process_batch(batch):
    #     batch = process_text(batch, text_processor)
    #     del batch["dataset_name"]
    #     return batch

    # del pretrained_model

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

def create_unified_index(dataset_dir):
    dataset_dir = Path(dataset_dir)

    indices_json = list(dataset_dir.rglob('index.json'))
    parent_json_path = dataset_dir / 'index.json'

    parent_data = {
        'shards': [],
        'version': 2
    }

    for json_path in tqdm(indices_json):
        assert json_path.exists()

        with open(json_path, 'r') as f:
            data_child = json.load(f)

        assert data_child.keys() == parent_data.keys()

        for sample in data_child['shards']:
            correct_path_sample = deepcopy(sample)

            new_dir = str(json_path).replace('/index.json', '').replace(
                str(dataset_dir),
                ''
            )[1:]

            old_raw_data_path = sample['raw_data']['basename']
            new_raw_data_path = os.path.join(new_dir, old_raw_data_path)
            correct_path_sample['raw_data']['basename'] = new_raw_data_path

            if sample['zip_data'] is not None:
                old_zip_data_path = sample['zip_data']['basename']
                new_zip_data_path = os.path.join(new_dir, old_zip_data_path)
                correct_path_sample['zip_data']['basename'] = new_zip_data_path

            parent_data['shards'].append(deepcopy(correct_path_sample))
            del correct_path_sample

    parent_json_path = dataset_dir / 'index.json'

    if parent_json_path.exists():
        raise FileExistsError('Delete the old `index.json` file then re-run the code')

    with open(parent_json_path, 'w') as f:
        json.dump(parent_data, f)


def main(_):
    do_val = False
    if do_val:
        import sys
        ds_name = os.path.basename(sys.argv[1])
        print('dataset:', ds_name)
        FLAGS.config.dataset_kwargs.name = ds_name
    else:
        ds_name = FLAGS.config.dataset_kwargs.name

    ds_path = FLAGS.config.dataset_kwargs.data_dir

    print('dataset:', ds_name)
    print('path:', ds_path)

    dataset = make_replay_dataset(train=TRAIN)
    dataset_statistics = dataset.dataset_statistics
    dataset = dataset.unbatch().iterator()

    target_size = 150 * 1024 * 1024 # 130MB
    item_size = 640 # 640B
    num_items = np.ceil(target_size / item_size)
    shard_size = int(num_items * item_size)

    out_root = "/mnt/hdd2/libero/mpt_dataset"
    if do_val:
        out_root = os.path.join(out_root, 'libero_val')

    out_path = os.path.join(out_root, ds_name+'_128x128')

    if True:
        os.makedirs(out_path, exist_ok=True)

        prev_index = None
        writer = None
        for item in tqdm(dataset):

            index = item['index'][0]

            if index != prev_index:
                path = os.path.join(out_path, f"traj_{index}")
                assert not os.path.exists(path), f"File {path} already exists"
                if writer is not None:
                    writer.finish()
                writer = MDSWriter(columns=columns, out=path, size_limit=shard_size)
                prev_index = index
            
            item_flat = {}
            for k, v in flatten(item, 'dot').items():
                if k in ['dataset_name', 'task.language_instruction']:
                    item_flat[k] = v
                else:
                    item_flat[k] = v.astype(target_dtypes[k])

            writer.write(item_flat)

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
    

    create_unified_index(out_path)


if __name__ == "__main__":
    app.run(main)