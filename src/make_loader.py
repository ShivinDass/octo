import imp
import os
import types

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

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from ipdb import set_trace as bp
from IPython import embed

import flags_config
FLAGS = flags.FLAGS

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
                 sharding: str):

        assert bs % minibs == 0, "Minibatch size does not divise batch size"

        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.sharding = sharding

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

        return OctoREPLAYMinibatches(self.bs, batch, minibs=minibs)

class OctoREPLAYMinibatches(REPLAYMinibatches):
    def __init__(self,
                 bs: int,
                 batch: dict,
                 minibs: int):

        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs

    def make_iterator(self, batch, minibs):
        s = 0
        this_bs = self.bs
        while True:
            e = min(s + minibs, this_bs)
            sel = slice(s, e)
            s = e

            mini_batch = slice_batch(batch, sel)

            indices = mini_batch.get('index', None)
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

    if train:
        dataset_kwargs = FLAGS.config.dataset_kwargs
    else:
        dataset_kwargs = FLAGS.config.val_dataset_kwargs

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

    print('Collecting unique indices')
    print('###########################')
    print('Need to fix this at the end')
    print('###########################')
    unique_indices = np.empty(1_000_000)
    # unique_indices = set()
    # one_time_iter = (
    #     dataset
    #     .unbatch()
    #     .batch(FLAGS.config.batch_size)
    #     .prefetch(buffer_size=tf.data.AUTOTUNE)
    #     .iterator()
    # )
    # for el in one_time_iter:
    #     unique_indices |= set(el['index'])

    data_weights = jax.numpy.ones((len(unique_indices),), dtype=jnp.float32)

    if return_dw_only:
        return None, data_weights

    # create iterator over dataset
    # skip first few batches
    if train:
        batch_size = FLAGS.config.batch_size
    else:
        batch_size = FLAGS.config.val_batch_size

    # skip_samples = start_batch * FLAGS.config.batch_size
    skip_samples = start_batch * batch_size
    # prefetch_buffer_size = tf.data.AUTOTUNE
    # prefetch_buffer_size = jax.device_count() * 1
    data_iterator = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size, seed=FLAGS.config.seed)
        .skip(skip_samples)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .iterator()
    )

    data_iterator = map(process_batch, data_iterator)

    return data_iterator, data_weights

    ############################################

def make_replay_iterators(start_batch, end_batch, sharding, data_iterator, mode, global_seed, batch_size):

    # start returning batches
    batch_idx = start_batch

    if mode == 'train':
        minibs = FLAGS.config.mini_batch_size
    else:
        minibs = FLAGS.config.mini_val_batch_size

    while batch_idx < end_batch:
        batch = next(data_iterator)
        seed = batch_idx * batch_size + np.arange(batch_size) + global_seed * 1e9
        seed = seed.astype('int64')
        batch['seed'] = seed

        batch_idx += 1

        replay_batch = OctoREPLAYBatch(
            batch=batch,
            bs=batch_size,
            minibs=minibs,
            sharding=sharding,
        )

        yield replay_batch

def make_split_loader_and_data_weights(start_batch: int,
                                       end_batch: int,
                                       sharding: str,
                                       mode: str='train',
                                       seed: int=42):

    assert mode in ['train', 'val', 'test']

    ds_iter, _ = make_replay_dataset(start_batch, end_batch, sharding, train=(mode=='train'))
    if mode == 'train':
        batch_size = FLAGS.config.batch_size
    else:
        batch_size = FLAGS.config.val_batch_size

    # return make_replay_iterators(start_batch, end_batch, sharding, ds_iter, global_seed=seed, batch_size=FLAGS.config.batch_size)
    return make_replay_iterators(start_batch, end_batch, sharding, ds_iter, mode=mode, global_seed=seed, batch_size=batch_size)

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
