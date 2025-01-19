import imp
import os
import types
import json

import jax
import jax.numpy as jnp
import numpy as np

import flax
import tensorflow as tf
from absl import app, flags
from ml_collections import config_flags, ConfigDict
from flatten_dict import flatten, unflatten

from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    check_config_diff,
    process_text,
)

from streaming import StreamingDataset, StreamingDataLoader

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from ipdb import set_trace as bp
from IPython import embed

import flags_config
FLAGS = flags.FLAGS

from typing import Callable, Any
class OctoDataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable=None,
                 **kwargs,
                ) -> None:
        super().__init__(
            local=local,
            remote=remote,
            shuffle=shuffle,
            batch_size=batch_size,
            **kwargs,
        )
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        item = super().__getitem__(idx)
        item = unflatten(item, 'dot')

        if self.transforms is None:
            return flatten(item, 'dot')
        else:
            return flatten(self.transforms(item), 'dot')

def slice_batch(batch, sel):
    if isinstance(batch, dict):
        return {k: slice_batch(v, sel) for k, v in batch.items()}
    else:  # numpy array case
        return batch[sel]

class REPLAYBatch:
    def __init__(self, bs):
        # batch size of this batch
        self.bs = bs

    def get_minibatches(self, part):
        # part: 'train', 'val', or 'meta'
        raise NotImplementedError

class REPLAYMinibatches:
    def __init__(self, bs):
        self.bs = bs

    def __iter__(self):
        # iterates over minibatches, each minibatch is a (ixs, (x, y)) tuple
        # where ixs, x and y are jax arrays or numpy arrays
        # ixs: indices of the data points in the minibatch
        # x: input data
        # y: output data
        raise NotImplementedError

class OctoREPLAYBatch(REPLAYBatch):
    def __init__(self,
                 batch: dict,
                 bs: int,
                 minibs: int,
                 sharding: str,
                 state=None,
                 batch_idx:int=0,
                 offset:int=0):

        assert bs % minibs == 0, "Minibatch size does not divise batch size"

        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.sharding = sharding
        self.state = state
        self.batch_idx = batch_idx
        self.offset = offset

    def get_state(self):
        return self.state

    def get_batch_idx(self):
        return self.batch_idx

    def set_offset(self, offset):
        self.offset = offset

    def get_minibatches(self, part: str):
        if self.sharding is not None:
            batch = jax.device_put(self.batch, self.sharding)
        else:
            batch = self.batch
        minibs = int({
            # 'train': 1,
            # 'val': 1,
            'train': 0.5,
            'val': 0.5,
            'meta': 0.5
        }[part] * self.minibs)

        return OctoREPLAYMinibatches(self.bs, batch, minibs=minibs, offset=self.offset)

class OctoREPLAYMinibatches(REPLAYMinibatches):
    def __init__(self,
                 bs: int,
                 batch: dict,
                 minibs: int,
                 offset: int=0):

        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.offset = offset

    def make_iterator(self, batch, minibs):
        s = 0
        this_bs = self.bs
        while True:
            e = min(s + minibs, this_bs)
            sel = slice(s, e)
            s = e

            mini_batch = slice_batch(batch, sel)

            indices = mini_batch.get('index', None) + self.offset
            if 'index' in mini_batch:
                mini_batch['index'] += self.offset
            y = None
            yield indices, (mini_batch, y)

            if e == this_bs:
                break

    def __iter__(self):
        return self.make_iterator(self.batch, self.minibs)

def make_replay_dataset(start_batch: int,
                        end_batch: int,
                        sharding: str,
                        train: bool=True,
                        return_dw_only: bool=False):

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

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_item(item):
        def process_text(item, text_processor):
            if text_processor is None:
                item["task"].pop("language_instruction")

            else:
                item["task"]["language_instruction"] = text_processor.encode(
                    item["task"]["language_instruction"].decode("utf-8")
                )
                item["task"]["language_instruction"]["input_ids"] = item["task"]["language_instruction"]["input_ids"].squeeze()
                item["task"]["language_instruction"]["attention_mask"] = item["task"]["language_instruction"]["attention_mask"].squeeze()

            return item

        item = process_text(item, text_processor)
        del item["dataset_name"]
        return item

    del pretrained_model

    if train:
        mds_path = os.path.join(
            FLAGS.config.dataset_kwargs.data_dir,
            FLAGS.config.dataset_kwargs.name,
        )
    else:
        mds_path = os.path.join(
            FLAGS.config.val_dataset_kwargs.data_dir,
            FLAGS.config.val_dataset_kwargs.name,
        )

    if train:
        batch_size = FLAGS.config.batch_size
    else:
        batch_size = FLAGS.config.val_batch_size

    # dataset = StreamingDataset(
    dataset = OctoDataset(
        local=mds_path,
        remote=None,
        shuffle=train,
        shuffle_seed=FLAGS.config.seed,
        batch_size=batch_size,
        transforms=process_item,
    )

    def numpy_collate(batch):
        return {
            k: np.array([sample[k] for sample in batch], dtype=object if isinstance(batch[0][k], bytes) else None)
            for k in batch[0].keys()
        }

    dataloader = StreamingDataLoader(
        dataset,
        drop_last=train,
        batch_size=batch_size,
        num_workers=FLAGS.config.num_workers,
        prefetch_factor=2,  # Optional: controls samples prefetched per worker
        collate_fn=numpy_collate,
    )

    current_epoch, remaining_batches = divmod(start_batch, len(dataloader))
    remaining_samples = remaining_batches * batch_size

    state = {
        'epoch': current_epoch,
        'sample_in_epoch': remaining_samples,
        'num_canonical_nodes': 1,
        'shuffle_seed': FLAGS.config.seed,
        'initial_physical_nodes': 1
    }

    dataloader.load_state_dict(state)

    unique_indices = np.empty(1_000_000)
    data_weights = jax.numpy.ones((len(unique_indices),), dtype=jnp.float32)

    if return_dw_only:
        return None, data_weights

    return dataloader, data_weights

def make_special_dataset(start_batch: int,
                         end_batch: int,
                         sharding: str,
                         return_dw_only: bool=False):

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

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_item(item):
        def process_text(item, text_processor):
            if text_processor is None:
                item["task"].pop("language_instruction")

            else:
                item["task"]["language_instruction"] = text_processor.encode(
                    item["task"]["language_instruction"].decode("utf-8")
                )
                item["task"]["language_instruction"]["input_ids"] = item["task"]["language_instruction"]["input_ids"].squeeze()
                item["task"]["language_instruction"]["attention_mask"] = item["task"]["language_instruction"]["attention_mask"].squeeze()

            return item

        item = process_text(item, text_processor)
        del item["dataset_name"]
        return item

    del pretrained_model

    mds_path = os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        FLAGS.config.dataset_kwargs.name,
    )

    checkpoint_path = FLAGS.config.checkpoint_path
    special_index_path = os.path.join(
        checkpoint_path,
        'special_index.json'
    )

    with open(special_index_path, 'r') as f:
        index = json.load(f)

    batch_size = FLAGS.config.batch_size

    # dataset = StreamingDataset(
    dataset = OctoDataset(
        local=mds_path,
        remote=None,
        shuffle=False,
        shuffle_seed=FLAGS.config.seed,
        batch_size=batch_size,
        transforms=process_item,
        index_filename=special_index_path
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
        num_workers=FLAGS.config.num_workers,
        prefetch_factor=2,  # Optional: controls samples prefetched per worker
        collate_fn=numpy_collate,
    )

    return dataloader, None

def make_replay_iterators(start_batch, end_batch, sharding, data_iterators, mode, global_seed, batch_size):

    ds_iter, ds_attrib = data_iterators
    if mode != 'train':
        assert ds_attrib is None

    # start returning batches
    batch_idx = start_batch

    if mode == 'train':
        minibs = FLAGS.config.mini_batch_size
    else:
        minibs = FLAGS.config.mini_val_batch_size

    special_batch = FLAGS.config.num_steps - FLAGS.config.bob_steps

    if batch_idx <= special_batch:
        global_iter = batch_idx
    else:
        global_iter = start_batch + len(ds_attrib)

    entering_special = False
    while batch_idx < end_batch:

        if mode != 'train':
            data_iterator = ds_iter
        else:
            if batch_idx == special_batch:
                data_iterator = ds_attrib
            else:
                data_iterator = ds_iter

        for batch in data_iterator:
            if batch_idx >= end_batch:
                break

            # seed = batch_idx * batch_size + np.arange(batch_size) + global_seed * 1e9
            seed = global_iter * batch_size + np.arange(batch_size) + global_seed * 1e9
            seed = seed.astype('int64')
            batch['seed'] = seed

            batch = unflatten(batch, 'dot')

            replay_batch = OctoREPLAYBatch(
                batch=batch,
                bs=batch_size,
                minibs=minibs,
                sharding=sharding,
                state=data_iterator.state_dict(),
                batch_idx=batch_idx,
            )

            if batch_idx == special_batch:
                replay_batch.set_offset(1_000_000)

            yield replay_batch

            global_iter += 1
            if mode != 'train' or batch_idx != special_batch:
                batch_idx += 1
                if batch_idx == special_batch:
                    entering_special = True
                    break

            entering_special = False

        if mode == 'train' and batch_idx == special_batch and not entering_special:
            global_iter += 1
            batch_idx += 1

def create_special_index():

    checkpoint_path = FLAGS.config.checkpoint_path
    special_index_path = os.path.join(
        checkpoint_path,
        'special_index.json'
    )

    if os.path.exists(special_index_path):
        return

    index_path = os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        FLAGS.config.dataset_kwargs.name,
        'index.json'
    )

    with open(index_path, 'r') as f:
        index = json.load(f)

    num_candidate_shards = int(
        FLAGS.config.candidate_size * len(index['shards'])
    )

    rng = np.random.default_rng(FLAGS.config.seed)
    candidate_shards = rng.choice(index['shards'], num_candidate_shards, replace=False).tolist()

    special_index = {
        'shards': candidate_shards,
        'version': 2
    }

    with open(special_index_path, 'w') as f:
        json.dump(special_index, f)

def make_split_loader_and_data_weights(start_batch: int,
                                       end_batch: int,
                                       sharding: str,
                                       mode: str='train',
                                       seed: int=42):

    assert mode in ['train', 'val', 'test']

    create_special_index()

    ds_iter, _ = make_replay_dataset(start_batch, end_batch, sharding, train=(mode=='train'))
    if mode == 'train':
        ds_attrib, _ = make_special_dataset(start_batch, end_batch, sharding)
    else:
        ds_attrib = None

    if mode == 'train':
        batch_size = FLAGS.config.batch_size
    else:
        batch_size = FLAGS.config.val_batch_size

    # return make_replay_iterators(start_batch, end_batch, sharding, ds_iter, global_seed=seed, batch_size=FLAGS.config.batch_size)
    return make_replay_iterators(
        start_batch,
        end_batch,
        sharding,
        (ds_iter, ds_attrib),
        mode=mode,
        global_seed=seed,
        batch_size=batch_size
    )


def main(_):
    start_batch = 0
    end_batch = 10
    sharding = None

    train_batcher_iter, data_weights = make_replay_dataset(start_batch, end_batch, sharding, train=True)
    train_replay_iterator = make_replay_iterators(start_batch, end_batch, '', train_batcher_iter, global_seed=FLAGS.config.seed, batch_size=FLAGS.config.batch_size)

    # start_batch = 0
    # end_batch = 5
    # sharding = None
    # val_batcher_iter, _ = make_replay_dataset(start_batch, end_batch, sharding, train=False)
    # val_replay_iterator = make_replay_iterators(start_batch, end_batch, val_batcher_iter)

    # train_its = FLAGS.config.num_steps
    # val_its = FLAGS.config.num_val_steps # this needs to be updated to include all samples in val

    # return train_replay_iterator, val_replay_iterator, data_weights, train_its, vals_its

    i = 0
    # replay_iterators = make_replay_iterators(start_batch, end_batch, sharding)
    for replay_iterator in train_replay_iterator:
        j = 0
        for item in replay_iterator.get_minibatches('train'):
            bp()
            print(f'i, j = {i}, {j}')
            j += 1

        i += 1

    print('hola')

    raise NotImplementedError
    for replay_iterator in val_replay_iterator:
        j = 0
        for item in replay_iterator.get_minibatches('val'):
            item = item.data
            bp()
            print(f'i, j = {i}, {j}')
            j += 1

        i += 1

if __name__ == "__main__":
    app.run(main)
