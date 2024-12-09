import imp
import os
import types
from functools import cache
from absl import app, flags
import flax
import jax
from fnmatch import fnmatch
from ml_collections import config_flags, ConfigDict
import tensorflow as tf

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    check_config_diff,
    merge_params,
    process_text,
)

from ipdb import set_trace as bp
from IPython import embed

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from ipdb import set_trace as bp
from IPython import embed

import flags_config
FLAGS = flags.FLAGS

# @cache
def make_model():
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
    ############################################

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    # load standardize_fn from `path/to/file.py:fn_name` format
    if (
        standardize_fn := FLAGS.config["dataset_kwargs"].get("standardize_fn", None)
    ) is not None:

        if isinstance(standardize_fn, str):
            path, name = standardize_fn.split(":")
            # imp is deprecated, but it's also what ml_collections uses
            standardize_fn = getattr(imp.load_source("standardize_fn", path), name)
            del FLAGS.config["dataset_kwargs"]["standardize_fn"]
            FLAGS.config["dataset_kwargs"]["standardize_fn"] = standardize_fn

        elif isinstance(standardize_fn, types.FunctionType):
            standardize_fn = FLAGS.config["dataset_kwargs"]["standardize_fn"]

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
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
        shuffle=False,
        num_parallel_calls=1,
        num_parallel_reads=1,
    )
    dataset_statistics = dataset.dataset_statistics
    dataset = dataset#.cache()
    dataset.dataset_statistics = dataset_statistics

    print('Collecting unique indices')
    print('###########################')
    print('Need to fix this at the end')
    print('###########################')
    unique_indices = 1_000 * [0]
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

    data_iterator = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size, seed=FLAGS.config.seed)
        .batch(FLAGS.config.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .iterator()
    )
    data_iterator = map(process_batch, data_iterator)
    example_batch = next(data_iterator)

    #########
    #
    # Load Pretrained Model
    #
    #########
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    frozen_keys = FLAGS.config.optimizer.frozen_keys

    flat_params = flax.traverse_util.flatten_dict(params)
    frozen_params = {}
    trainable_params = {}
    for key in flat_params:
        flat_key = ".".join(key)

        match = False
        for frozen_key in frozen_keys:
            if fnmatch(flat_key, frozen_key):
                match = True

        if match:
            frozen_params[key] = flat_params[key]
        else:
            trainable_params[key] = flat_params[key]

    frozen_params = flax.traverse_util.unflatten_dict(frozen_params)
    trainable_params = flax.traverse_util.unflatten_dict(trainable_params)

    return model, frozen_params, trainable_params

def main(_):
    model = make_model()

if __name__ == "__main__":
    app.run(main)
