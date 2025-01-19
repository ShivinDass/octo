import imp
import os
import types
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from absl import app, flags
import flax
import tensorflow as tf


from flatten_dict import flatten, unflatten
from ipdb import set_trace as bp
from ipdb import launch_ipdb_on_exception
from IPython import embed

from streaming import StreamingDataset, StreamingDataLoader
from torch.utils.data import DataLoader
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

columns = {
    'absolute_action_mask': 'ndarray:uint8',
    'action': 'ndarray:float32',
    'dataset_name': 'bytes',
    'index': 'ndarray:int64',
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
    'task.pad_mask_dict.language_instruction': 'ndarray:uint8'
}

target_dtypes = {
    k: v.replace('ndarray:', '')
    for k, v in columns.items()
}

def main(_):

    ds_name = FLAGS.config.dataset_kwargs.name
    # ds_path = FLAGS.config.dataset_kwargs.data_dir

    out_root = "/mnt/xfs/home/alaakh/store/oxe/mpt_dataset"
    mds_path = os.path.join(out_root, ds_name)

    batch_size = 32

    dataset = StreamingDataset(
        local=mds_path,
        remote=None,
        shuffle=False,
        shuffle_seed=42,
        batch_size=batch_size,
    )

    def numpy_collate(batch):
        return {
            k: np.array([sample[k] for sample in batch], dtype=object if isinstance(batch[0][k], bytes) else None)
            for k in batch[0].keys()
        }

    dataloader = StreamingDataLoader(
        dataset,
        drop_last=True,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=2,  # Optional: controls samples prefetched per worker
        collate_fn=numpy_collate,
    )

    i = 0

    dataset.epoch_size // batch_size

    state = {
        'epoch': 0,
        # 'sample_in_epoch': 704,
        'sample_in_epoch': batch_size * (dataset.epoch_size // batch_size - 2),
        'num_canonical_nodes': 1,
        'shuffle_seed': 42,
        'initial_physical_nodes': 1
    }

    # dataloader.load_state_dict(state)

    # bp()
    states = [dataloader.state_dict()]
    while True:
        for item_flat in dataloader:

            if i > 20:
                break

            # item_flat = next(dataloader)
            states.append(dataloader.state_dict())
            # print(states[-1])
            # item = unflatten(item_flat, 'dot')

            # for k, v in item_flat.items():
            #     if type(v) != bytes:
            #         print(k, ':', v.dtype, 'and', v.shape)
            #     else:
            #         print(k, ': bytes and ()')
            # print()

            i += 1

        if i > 20:
            break

    # print(dataset.epoch_size)
    for state in states:
        print(state)
        # print()

    bp()


if __name__ == "__main__":
    app.run(main)
