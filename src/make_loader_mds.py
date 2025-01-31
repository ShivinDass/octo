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
        shuffle=True,
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

class SpecialReplayBatch(REPLAYBatch):
    """
    Similar to OctoREPLAYBatch, but under the hood it doesn't load the entire batch.
    Instead, it will stream from data_iter chunk by chunk in get_minibatches(...).
    """

    def __init__(
        self,
        data_iter,
        global_iter: int,
        global_seed: int,
        batch_size: int,
        iter_bs: int,
        minibs: int,
        sharding,
        offset: int
    ):
        # Like OctoREPLAYBatch, we call super().__init__(bs)
        super().__init__(batch_size)

        self.data_iter = data_iter
        self.global_iter = global_iter
        self.global_seed = global_seed
        # self.bs = batch_size
        self.iter_bs = iter_bs
        self.minibs = minibs
        self.sharding = sharding
        self.offset = offset

        self.batch_idx = FLAGS.config.num_steps - FLAGS.config.bob_steps

        # We yield ourselves exactly once, just like your original design
        self._exhausted = False

    def __len__(self):
        # If data_iter is sized, you can do len(self.data_iter),
        # otherwise remove or approximate
        return len(self.data_iter)

    def __iter__(self):
        """So `for x in special_batch:` yields exactly one x (self)."""
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        self._exhausted = True
        return self

    def get_minibatches(self, part: str):
        """
        Instead of yielding directly, we return a SpecialReplayMinibatches
        object (like OctoREPLAYBatch returns OctoREPLAYMinibatches).
        That object will handle chunked iteration under the hood.
        """
        return SpecialReplayMinibatches(
            bs=self.bs,
            iter_bs=self.iter_bs,
            data_iter=self.data_iter,
            global_iter=self.global_iter,
            global_seed=self.global_seed,
            minibs=self.minibs,
            offset=self.offset,
            sharding=self.sharding,
            batch_idx=self.batch_idx,
            part=part
        )

class SpecialReplayMinibatches(REPLAYMinibatches):
    """
    Similar to OctoREPLAYMinibatches, but it streams multiple 512-chunks from `data_iter`.
    Each chunk is turned into an OctoREPLAYBatch, and we yield from that batch's minibatches.
    """

    def __init__(
        self,
        bs: int,
        iter_bs: int,
        data_iter,
        global_iter,
        global_seed,
        minibs: int,
        offset: int,
        sharding,
        batch_idx: int,
        part: str
    ):
        # Like OctoREPLAYMinibatches, we call super().__init__(bs).
        super().__init__(bs)

        self.data_iter = data_iter
        self.global_iter = global_iter
        self.global_seed = global_seed
        self.minibs = minibs
        self.offset = offset
        self.sharding = sharding
        self.batch_idx = batch_idx
        self.part = part

        # This object can have .bs, .minibs, etc. that calling code might expect
        self.bs = bs
        self.iter_bs = iter_bs

    def __iter__(self):
        """
        We'll fetch each 512-sized chunk from `self.data_iter`, create an OctoREPLAYBatch,
        then yield all sub-minibatches from that batch.
        """
        for chunk in self.data_iter:
            # Possibly check if chunk is empty (shape[0] == 0). If so, skip or break
            first_key = next(iter(chunk.keys()))
            if chunk[first_key].shape[0] == 0:
                break

            # Build seed array
            seed = (self.global_iter * self.iter_bs) \
                   + np.arange(self.iter_bs) \
                   + int(self.global_seed * 1e9)
            seed = seed.astype('int64')
            chunk['seed'] = seed

            # unflatten if needed
            chunk = unflatten(chunk, 'dot')

            # Wrap in an OctoREPLAYBatch (just like your normal pipeline)
            replay_batch = OctoREPLAYBatch(
                batch=chunk,
                bs=self.iter_bs,
                minibs=self.minibs,
                sharding=self.sharding,
                state=self.data_iter.state_dict() if hasattr(self.data_iter, 'state_dict') else None,
                batch_idx=self.batch_idx
            )

            if self.offset:
                replay_batch.set_offset(1_000_000)

            # Increase global_iter for the next chunk
            self.global_iter += 1

            # "Yield from" the sub-minibatches of this chunk
            yield from replay_batch.get_minibatches(self.part)

# class SpecialReplayBatch:
#     def __init__(self,
#                  data_iter,
#                  global_iter: int,
#                  global_seed: int,
#                  batch_size: int,
#                  minibs: int,
#                  sharding,
#                  offset: int):

#         self.data_iter = data_iter
#         self.global_iter = global_iter
#         self.global_seed = global_seed
#         self.batch_size = batch_size
#         self.minibs = minibs
#         self.sharding = sharding
#         self.offset = offset

#         self.batch_idx = FLAGS.config.num_steps - FLAGS.config.bob_steps

#         self._exhausted = False  # Will track if we've already yielded once

#     def __len__(self):
#         return len(self.data_iter)

#     def __iter__(self):
#         """
#         Make this object an iterable by returning 'self'.
#         Python expects an iterator to return itself from __iter__.
#         """
#         return self

#     def __next__(self):
#         """
#         We'll yield ourselves exactly once. After that, raise StopIteration.
#         This way:
#             next(big_replay_batch)  -> returns big_replay_batch (the first time)
#             next(big_replay_batch)  -> StopIteration
#         And 'for x in big_replay_batch:' also yields exactly one 'x'.
#         """
#         if self._exhausted:
#             raise StopIteration

#         self._exhausted = True
#         return self

#     def get_minibatches(self, part: str):
#         """
#         Generator that yields sub-chunks (each of size 512, if ds_attrib is 512-sized),
#         wrapped in OctoREPLAYBatch. Streams from ds_attrib without storing everything.
#         """

#         for batch in self.data_iter:

#             # Build seed array for this chunk
#             seed = (self.global_iter * self.batch_size) \
#                    + np.arange(self.batch_size) \
#                    + int(self.global_seed * 1e9)

#             seed = seed.astype('int64')
#             batch['seed'] = seed

#             # Suppose you have a function unflatten:
#             batch = unflatten(batch, 'dot')

#             # Wrap the chunk in your usual OctoREPLAYBatch (split into minibs=64, etc.)
#             replay_batch = OctoREPLAYBatch(
#                 batch=batch,
#                 bs=self.batch_size,
#                 minibs=self.minibs,
#                 sharding=self.sharding,
#                 state=self.data_iter.state_dict(),  # if ds_attrib has state
#                 batch_idx=self.batch_idx  # or pass something meaningful
#             )

#             if self.offset:
#                 replay_batch.set_offset(1_000_000)

#             for minibatch in replay_batch.get_minibatches(part):
#                 yield minibatch

#             self.global_iter += 1

def make_replay_iterators(
        start_batch,
        end_batch,
        sharding,
        data_iterators,
        mode,
        global_seed,
        batch_size
    ):

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

    if ds_attrib is not None:
        ds_special = SpecialReplayBatch(
            data_iter=ds_attrib,
            global_iter=global_iter,
            global_seed=global_seed,
            batch_size=batch_size*len(ds_attrib),
            iter_bs=batch_size,
            minibs=minibs,
            sharding=sharding,
            offset=1_000_000,
        )
    else:
        ds_special = None

    while batch_idx < end_batch:

        if mode != 'train':
            data_iterator = ds_iter
        else:
            if batch_idx == special_batch:
                data_iterator = ds_special
            else:
                data_iterator = ds_iter

        for batch in data_iterator:
            if batch_idx >= end_batch:
                break

            if batch_idx == special_batch:
                yield ds_special

            else:
                # here it's the regular iterator
                # we keep the same
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

                yield replay_batch

                global_iter += 1

            batch_idx += 1

        # what to do here?
        # if mode == 'train' and batch_idx == special_batch and not entering_special:
        #     global_iter += 1
        #     batch_idx += 1

def create_special_index(iter_seed:int=0):

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

    rng = np.random.default_rng(iter_seed + FLAGS.config.seed)
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
                                       seed: int=42,
                                       iter_seed: int=0):

    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        create_special_index(iter_seed=iter_seed)

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
