import os
import gc
import jax
import jax.numpy as jnp
import json
import flax
import numpy as np
import dill
from copy import deepcopy
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from collections import defaultdict

from functools import partial
from functools import cache
from scipy.stats import spearmanr, pearsonr

# from domains.vjp_lm import vjp_lm
# from domains.vjp_blocks import one_sample_vjp_head, sample_loss_vjp_head, \
#     example_loss_vjp_skeleton

from octo.utils.jax_utils import initialize_compilation_cache

from make_loader_mds import make_split_loader_and_data_weights, make_replay_dataset
# from make_loader import make_split_loader_and_data_weights, make_replay_dataset

from make_model import make_model
from jax_lm.domains.vjp_robodm import vjp_robodm
from jax_lm.metagradients.optimizers.adam import make_adam_optimizer
from jax_lm.metagradients.optimizers.interpolation import interp_from, interp_from_mom
from jax_lm.metagradients.utils import make_shardings
from jax_lm.domains.vjp_blocks import example_loss_vjp_skeleton, sample_loss_vjp_head

from ipdb import set_trace as bp
from IPython import embed

from absl import flags, app
from ml_collections import config_flags

# jax.config.update("jax_disable_jit", True)

import flags_config
FLAGS = flags.FLAGS

EPS = 1.0000000000000001e-11
SEED_SPACING = 100000

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def grad_from_store(deps, batch_indices):
    flat_deps = {k: v for d in deps.values() for k, v in d.items()}
    # num_datapoints = max(b.max() for b in batch_indices) + 1
    num_datapoints = max(b.max() for b in batch_indices.values()) + 1
    gradient = np.zeros((num_datapoints,), dtype=np.float32)

    # for i, bixs in enumerate(batch_indices):
        # gradient[bixs] += flat_deps[i]
    for batch_n, indices in batch_indices.items():
        gradient[indices] += flat_deps[batch_n]

    return gradient

@cache
def make_vjp_skele(bs):
    return jax.tree_util.Partial(partial(example_loss_vjp_skeleton, bs=bs))

@partial(jax.jit, static_argnames=['train', 'divisor'])
def per_sample_loss_fn(params,
                       batch,
                       model,
                       frozen_params,
                       train,
                       data_weights=None,
                       divisor=1.0):

    """
    inputs:
        params: trainable parameters
        frozen_params: non-trainable parameters
    """
    assert divisor == 1.0, divisor

    flat_params = flax.traverse_util.flatten_dict(params)
    flat_frozen_params = flax.traverse_util.flatten_dict(frozen_params)

    all_params = flat_params | flat_frozen_params
    all_params = flax.traverse_util.unflatten_dict(all_params)

    model = model.replace(params=all_params)

    _, (data, _) = batch[:2]
    assert 'seed' in data

    seed = data['seed'][0]
    rng = jax.random.PRNGKey(seed)

    bound_module = model.module.bind({"params": all_params}, rngs={"dropout": rng})
    transformer_embeddings = bound_module.octo_transformer(
        data["observation"],
        data["task"],
        data["observation"]["pad_mask"],
        # train=train,
        train=False,
    )
    action_loss, action_metrics = bound_module.heads["action"].per_sample_loss(
        transformer_embeddings,  # Action head knows to pull out the action readout_key
        data["action"],
        pad_mask=data["observation"]["pad_mask"],
        # train=train,
        train=False,
    )

    if data_weights is not None:
        indices = data['index']
        these_data_weights = data_weights[indices]
        action_loss = action_loss * these_data_weights

    return action_loss / divisor

def data_selection_iter(data_weights: jax.numpy.array,
                        checkpoint_path: str,
                        job_id: int=0):

    initialize_compilation_cache()
    devices = jax.devices()

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    print('Checkpoint path:', checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    FLAGS.config.checkpoint_path = checkpoint_path

    train_batcher = partial(make_split_loader_and_data_weights, mode='train', seed=FLAGS.config.seed, iter_seed=job_id)
    train_its = FLAGS.config.num_steps
    bob_its = FLAGS.config.bob_steps
    forward_its = train_its - bob_its

    # # special_batch = FLAGS.config.num_steps - FLAGS.config.bob_steps
    # for batch in train_batcher(0, 5, None):
    #     pass
    # # for batch in train_batcher(20, 21, None):
    # # for batch in train_batcher(special_batch, special_batch+1, None):
    #     for item in batch.get_minibatches('train'):
    #         bpss()
    #         pass

    val_batcher = partial(make_split_loader_and_data_weights, mode='val', seed=FLAGS.config.seed)
    # val_its = FLAGS.config.num_val_steps
    val_its = 13 # this is the number of iterations to do a full pass over the val datalaoder
    # val_batcher(0, 5, "")

    model, frozen_params, trainable_params = make_model(train_batcher)

    num_trainable_params = sum(x.size for x in jax.tree_util.tree_leaves(trainable_params))
    num_frozen_params = sum(x.size for x in jax.tree_util.tree_leaves(frozen_params))
    num_total_params = num_trainable_params + num_frozen_params

    print(f'Trainable params: {num_trainable_params:,}')
    print(f'Frozen params: {num_frozen_params:,}')
    print(f'Total params: {num_total_params:,}')

    psl = jax.tree_util.Partial(
        per_sample_loss_fn,
        frozen_params=frozen_params,
        model=model,
    )

    optimizer_dict = FLAGS.config.optimizer.to_dict()
    lr_scheduler_dict = optimizer_dict['learning_rate']

    OPTIMIZER_KWARGS = {
        'lr': lr_scheduler_dict['peak_value'],
        # 'wd': 1e-5,
        'wd': optimizer_dict['weight_decay'],
        'pct_start': lr_scheduler_dict['warmup_steps'] / lr_scheduler_dict['decay_steps'],
        'pct_final': 1,
        'b1': 0.9,
        'b2': 0.95,
        'min_lr_relative': max(lr_scheduler_dict['init_value'], EPS),
        'final_min_lr_relative': max(lr_scheduler_dict['end_value'], EPS),
        'eps': EPS,
        'eps_sqrt': EPS,
        'selective_wd': True,
        'dtype': jax.numpy.float32,
        'factored_lr_wd': False,
        'anneal_type': 'linear',
        'eps_schedule': jax.tree_util.Partial(interp_from, steps=200,
                                              eps0=1e-08, eps_root0=1e-08, space='geometric'),
        'mom_schedule': jax.tree_util.Partial(interp_from_mom, steps=25, mom0=0.85,
                                        mom1=1, space='linear'),
        'per_param_lr': None,
        'reuse_optimizer': False,
    }

    state0 = make_adam_optimizer(
        initial_params=trainable_params,
        train_its=train_its,
        **OPTIMIZER_KWARGS,
    )

    aux_datasets = {}
    return_kw = False

    sharding, replicated_sharding = make_shardings()
    head_val_batcher = jax.tree_util.Partial(val_batcher, sharding=sharding)

    # vjp_skele = jax.tree_util.Partial(partial(example_loss_vjp_skeleton, bs=FLAGS.config.batch_size))
    vjp_skele = make_vjp_skele
    vjp_head = partial(
        sample_loss_vjp_head,
        per_sample_loss=psl,
        val_batcher=head_val_batcher,
        val_its = val_its,
    )

    vjp_kw = dict(
        state=state0,
        vjp_head=vjp_head,
        vjp_skele=vjp_skele,
        data_weights=data_weights,
        return_kw=return_kw,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        psl=psl,
        n_train_ba=forward_its, # not train_its
        n_val_ba=val_its,
        aux_datasets=aux_datasets,
        return_state=True,
        forward_only=True,
        # segment_size=2,
        # forward_only=False,
    )

    ret = vjp_robodm(**vjp_kw)

    vjp_kw.update(dict(
        state=ret['final_state'],
        n_train_ba=train_its,
        forward_only=False,
    ))

    final_ret = vjp_robodm(**vjp_kw)

    y0 = float(final_ret['primal'])
    deps = final_ret['deps']
    batch_indices = final_ret['batch_indices']
    final_state = final_ret['final_state']
    num_datapoints = data_weights.size

    # merging final params

    final_params = final_state.params

    final_params = flax.traverse_util.flatten_dict(final_params)
    flat_frozen_params = flax.traverse_util.flatten_dict(frozen_params)

    all_params = final_params | flat_frozen_params
    all_params = flax.traverse_util.unflatten_dict(all_params)

    model = model.replace(params=all_params)

    # save model
    model.save_pretrained(step=train_its, checkpoint_path=checkpoint_path)
    # model.load_pretrained(step=train_its, checkpoint_path=checkpoint_path)

    import json
    with open(os.path.join(checkpoint_path, 'hparams_config.json'), 'w') as f:
        json.dump(FLAGS.config.to_dict(), f, indent=4)

    # indices = set()
    # special_batch = FLAGS.config.num_steps - FLAGS.config.bob_steps
    # for batch in train_batcher(special_batch, special_batch+1, None):
    #     for item in batch.get_minibatches('train'):
    #         pass
            # indices |= set(item[0].tolist())

    grad = grad_from_store(deps, batch_indices)
    print(grad)
    if len(grad) > num_datapoints:
        print('>> Shrinking grad array from', len(grad), 'to', num_datapoints)
        assert (grad[num_datapoints:] == 0).all()
        grad = grad[:num_datapoints]
        print(grad)
    elif len(grad) < num_datapoints:
        print('>> Growing grad array from', len(grad), 'to', num_datapoints)
        grad = np.concatenate([grad, np.zeros((num_datapoints - len(grad),))])
        print(grad)

    grad_path = os.path.join(checkpoint_path, 'datamodels.npy')
    np.save(grad_path, grad)

    state_path = os.path.join(checkpoint_path, 'state.dill')
    with open(state_path, "wb") as file:
        dill.dump(ret, file)

    return grad

def create_include_index(grad):

    prev_index_path = FLAGS.config.include_index_path
    all_index_path = os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        FLAGS.config.dataset_kwargs.name,
        'index.json'
    )

    with open(prev_index_path, 'r') as f:
        previous_include_index = json.load(f)

    with open(all_index_path, 'r') as f:
        all_index = json.load(f)

    selected_shards_idxs = set()
    for shard in previous_include_index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[1])
        if grad[shard_idx] <= 0:
            selected_shards_idxs.add(shard_idx)

    for shard in all_index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[1])
        if grad[shard_idx] < 0: # select the sample
            selected_shards_idxs.add(shard_idx)

    selected_shards = []
    for shard in all_index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[1])
        if shard_idx in selected_shards_idxs:
            selected_shards.append(shard)

    current_include_index = {
        'shards': selected_shards,
        'version': 2
    }

    current_include_index_path = os.path.join(FLAGS.config.checkpoint_path, 'include_index.json')
    with open(current_include_index_path, 'w') as f:
        json.dump(current_include_index, f)


def create_include_index_perc():

    if '-' in FLAGS.config.dataset_kwargs.all_train_datasets:
        formatted_date_time = FLAGS.config.folder_name
        all_index_path = os.path.join(
            FLAGS.config.save_dir, 
            FLAGS.config.dataset_kwargs.name, 
            formatted_date_time,
            'merged_index.json'
        )
    else:
        all_index_path = os.path.join(
            FLAGS.config.dataset_kwargs.data_dir,
            FLAGS.config.dataset_kwargs.name,
            'index.json'
        )

    with open(all_index_path, 'r') as f:
        all_index = json.load(f)

    job_id = FLAGS.config.job_id
    include_idx = set([int(x['raw_data']['basename'].split('/')[-2]) for x in all_index['shards']])
    removed = set()
    for i in range(job_id+1):
        grad_path = os.path.join(
            os.path.dirname(FLAGS.config.checkpoint_path),
            f'iter_{i}',
            'datamodels.npy'
        )

        grad_i = np.load(grad_path)[1_000_000:]
        non_zero = np.where(grad_i != 0)[0]

        perc_10 = np.percentile(grad_i, 20)
        perc_90 = np.percentile(grad_i, 80)
        # perc_10 = np.percentile(grad_i[non_zero], 20)
        # perc_90 = np.percentile(grad_i[non_zero], 80)
        for shard in all_index['shards']:
            shard_idx = int(shard['raw_data']['basename'].split('/')[-2])

            if grad_i[shard_idx] <= perc_10:
                include_idx.add(shard_idx)
                if shard_idx in removed:
                    removed.remove(shard_idx)
    
            elif grad_i[shard_idx] == 0:
                if shard_idx not in removed:
                    include_idx.add(shard_idx)
    
            elif grad_i[shard_idx] >= perc_90:
                if shard_idx in include_idx:
                    include_idx.remove(shard_idx)
                
                removed.add(shard_idx)
            
            else:
                pass

    # always keep real data
    for i in range(999_990, 1_000_000):
        include_idx.add(i)

    include_shards = []
    for shard in all_index['shards']:
        shard_idx = int(shard['raw_data']['basename'].split('/')[-2])
        if shard_idx in include_idx:
            include_shards.append(shard)

    print('Total shards:', len(all_index['shards']))
    print('Included shards:', len(include_shards))

    current_include_index = {
        'shards': include_shards,
        'version': 2
    }

    current_include_index_path = os.path.join(FLAGS.config.checkpoint_path, 'include_index.json')
    with open(current_include_index_path, 'w') as f:
        json.dump(current_include_index, f)

def create_original_index():
    
    formatted_date_time = FLAGS.config.folder_name
    index_out_dir = os.path.join(
        FLAGS.config.save_dir, 
        FLAGS.config.dataset_kwargs.name, 
        formatted_date_time,
        'merged_index.json'
    )

    if os.path.exists(index_out_dir):
        return
    
    list_of_datasets = FLAGS.config.dataset_kwargs.all_train_datasets.split('-')
    
    multi_dataset = {
        'shards': [],
        'version': 2
    }

    for dataset_name in list_of_datasets:

        ##################################################################
        if 'easy_pick' in dataset_name:
            continue
        ##################################################################

        dataset_index_path = os.path.join(
            FLAGS.config.dataset_kwargs.data_dir,
            dataset_name,
            'index.json'
        )

        with open(dataset_index_path, 'r') as f:
            index = json.load(f)

        for sample in index['shards']:
            correct_path_sample = deepcopy(sample)

            old_raw_data_path = sample['raw_data']['basename']
            new_raw_data_path = os.path.join(dataset_name, old_raw_data_path)
            correct_path_sample['raw_data']['basename'] = new_raw_data_path

            if sample['zip_data'] is not None:
                old_zip_data_path = sample['zip_data']['basename']
                new_zip_data_path = os.path.join(dataset_name, old_zip_data_path)
                correct_path_sample['zip_data']['basename'] = new_zip_data_path

            multi_dataset['shards'].append(deepcopy(correct_path_sample))
            del correct_path_sample

    with open(index_out_dir, 'w') as f:
        json.dump(multi_dataset, f)

    with open(os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        'index.json'
    ), 'w') as f:
        json.dump(multi_dataset, f)

def combine_stats(dicts):
    """
    Combine a list of dictionaries (dicts) of stats into a single dictionary
    in a principled way. Each dictionary has keys:
      - '<prefix>.mean': list of floats
      - '<prefix>.std': list of floats
      - '<prefix>.min': list of floats
      - '<prefix>.max': list of floats
      - '<prefix>.mask': list (all are the same mask)
      - 'num_transitions': int
      - 'num_trajectories': int
    where <prefix> could be 'action', 'proprio', etc.

    Returns a single dictionary with combined statistics.
    """

    # We assume all dictionaries have the same set of keys (except we only 
    # store one copy of the mask). We'll systematically handle each prefix.

    # Identify the keys that indicate "mean", "std", "min", "max"
    # We'll do this dynamically based on any dictionary's keys.
    if not dicts:
        raise ValueError("No dictionaries provided.")

    # We'll just pick the keys from the first dictionary.
    first_keys = dicts[0].keys()

    # We'll find which prefixes we have: e.g. "action.mean", "proprio.mean", etc.
    # For each prefix, we'll store data in arrays and combine them.
    # Let's group keys by the portion before the dot: "action", "proprio", etc.
    # Then within each group, identify "mean", "std", "min", "max", "mask".
    
    # A function to parse "action.mean" -> prefix="action", stat="mean"
    def split_key(key):
        parts = key.split('.')
        if len(parts) == 2:
            return parts[0], parts[1]  # e.g. ("action", "mean")
        else:
            return None, None
    
    # We'll build up a structure like:
    # stats_info = {
    #   "action": {
    #       "mean": [...list of arrays from each dict...],
    #       "std":  [...],
    #       "min":  [...],
    #       "max":  [...],
    #       "mask": [...],  # Should be identical
    #   },
    #   "proprio": {
    #       ...
    #   }
    # }
    
    # For convenience, also keep a list/array of n_i for each dict.
    num_trajectories_list = [d["num_trajectories"] for d in dicts]
    total_trajectories = sum(num_trajectories_list)

    num_transitions_list = [d["num_transitions"] for d in dicts]
    total_transitions = sum(num_transitions_list)

    # We'll gather data by prefix & statistic
    stats_info = {}

    # Collect all prefix/stat pairs
    for d in dicts:
        for key, val in d.items():
            if key in ("num_transitions", "num_trajectories"):
                # We'll handle these separately
                continue
            prefix, stat = split_key(key)
            if prefix is None or stat is None:
                # Not of the form prefix.stat -- skip or handle differently
                continue
            
            if prefix not in stats_info:
                stats_info[prefix] = {}
            
            if stat not in stats_info[prefix]:
                stats_info[prefix][stat] = []
            
            stats_info[prefix][stat].append(val)  # store the list (will convert to array later)

    # Now we combine each prefix's stats
    # We'll build a result dictionary
    result = {}
    combined_dict = {}

    for prefix, subdict in stats_info.items():
        # subdict might look like {"mean": [...], "std": [...], "min": [...], "max": [...], "mask": [...]}
        
        # Convert each list of lists to list of np.arrays for easy manipulation
        # For example, subdict["mean"] is a list of length n (one per dictionary),
        # each item is a python list for the vector.
        
        # We only combine mean and std using the weighting approach.
        # For min and max, we do elementwise min across all, elementwise max across all.
        # mask: we assume they are all identical, so we just take the first.
        
        # 1) handle the mask if present
        if "mask" in subdict:
            # let's assume they're all identical
            # we just take the first dictionary's mask
            combined_mask = subdict["mask"][0]
            result[f"{prefix}.mask"] = combined_mask  # keep it as list
        else:
            combined_mask = None
        
        # 2) handle mean + std
        mean_arrays = None
        std_arrays  = None
        if "mean" in subdict:
            mean_arrays = [np.array(arr) for arr in subdict["mean"]]
        if "std" in subdict:
            std_arrays = [np.array(arr) for arr in subdict["std"]]
        
        if mean_arrays is not None and std_arrays is not None:
            # We'll do the standard combination
            # sum of n_i * mu_i
            sum_n_mu = np.zeros_like(mean_arrays[0], dtype=float)
            # sum of n_i * (sigma_i^2 + mu_i^2)
            sum_n_var_term = np.zeros_like(mean_arrays[0], dtype=float)

            for i in range(len(dicts)):
                n_i   = num_trajectories_list[i]
                mu_i  = mean_arrays[i]
                sig_i = std_arrays[i]
                sum_n_mu       += n_i * mu_i
                sum_n_var_term += n_i * (sig_i**2 + mu_i**2)

            # final mean
            mean_combined = sum_n_mu / total_trajectories
            # final var
            var_combined = (sum_n_var_term / total_trajectories) - (mean_combined**2)
            # numerical safety: clamp any small negative values to 0
            var_combined = np.where(var_combined < 0, 0, var_combined)
            std_combined = np.sqrt(var_combined)

            result[f"{prefix}.mean"] = mean_combined.tolist()
            result[f"{prefix}.std"]  = std_combined.tolist()
        
        # 3) handle min
        if "min" in subdict:
            min_arrays = [np.array(arr) for arr in subdict["min"]]
            # elementwise min across all i
            combined_min = min_arrays[0].copy()
            for i in range(1, len(min_arrays)):
                combined_min = np.minimum(combined_min, min_arrays[i])
            result[f"{prefix}.min"] = combined_min.tolist()

        # 4) handle max
        if "max" in subdict:
            max_arrays = [np.array(arr) for arr in subdict["max"]]
            # elementwise max across all i
            combined_max = max_arrays[0].copy()
            for i in range(1, len(max_arrays)):
                combined_max = np.maximum(combined_max, max_arrays[i])
            result[f"{prefix}.max"] = combined_max.tolist()

    # Finally, set the total transitions and total trajectories
    result["num_transitions"] = total_transitions
    result["num_trajectories"] = total_trajectories

    return result

def create_data_stats():
    from pathlib import Path
    from flatten_dict import flatten, unflatten

    data_path = Path(FLAGS.config.dataset_kwargs.data_dir)

    all_stats = []
    for data_stat_json in data_path.rglob('dataset_statistics.json'):
        if len(str(data_stat_json).split('/')) != 11: continue

        with open(data_stat_json, 'r') as f:
            data_stat = json.load(f)

        all_stats.append(flatten(data_stat, 'dot'))

    combined = combine_stats(all_stats)
    combined = unflatten(combined, 'dot')

    with open(os.path.join(
        FLAGS.config.dataset_kwargs.data_dir,
        'dataset_statistics.json'
    ), 'w') as f:
        json.dump(combined, f)

def main(_):

    job_id = FLAGS.config.job_id

    # formatted_date_time = datetime.now().strftime("%d-%b-%Y_%I-%M-%S%p").lower()
    formatted_date_time = FLAGS.config.folder_name
    meta_checkpoint_path = os.path.join(FLAGS.config.save_dir, FLAGS.config.dataset_kwargs.name, formatted_date_time)

    checkpoint_path = os.path.join(meta_checkpoint_path, f'iter_{job_id}')
    os.makedirs(checkpoint_path, exist_ok=True)

    FLAGS.config.checkpoint_path = checkpoint_path
    FLAGS.config.include_index_path = None

    if job_id == 0:
        _, data_weights = make_replay_dataset(0, 1e5, None, train=True, return_dw_only=True)
        data_weights = jax.numpy.concatenate(
            [data_weights, jax.numpy.zeros_like(data_weights)],
            axis=0
        )

        create_original_index()
        create_data_stats()

        FLAGS.config.include_index_path = os.path.join(
            FLAGS.config.save_dir, 
            FLAGS.config.dataset_kwargs.name, 
            formatted_date_time,
            'merged_index.json'
        )

    else:
        prev_job = job_id - 1
        # prev_dw_path = os.path.join(meta_checkpoint_path, f'iter_{prev_job}', 'data_weights.npy')
        # assert os.path.exists(prev_dw_path)

        _, data_weights = make_replay_dataset(0, 1e5, None, train=True, return_dw_only=True)
        data_weights = jax.numpy.concatenate(
            [data_weights, jax.numpy.zeros_like(data_weights)],
            axis=0
        )

        prev_index_path = os.path.join(meta_checkpoint_path, f'iter_{prev_job}', 'include_index.json')
        FLAGS.config.include_index_path = prev_index_path

    grad = data_selection_iter(
        data_weights,
        checkpoint_path,
        job_id=job_id,
    )

    candidate_grad = grad[1_000_000:]
    # create_include_index(candidate_grad)
    
    create_include_index_perc()

    # # old
    # include_samples = jnp.where(candidate_grad < 0)[0]
    # exclude_samples = jnp.where(candidate_grad > 0)[0]
    # data_weights = data_weights.at[include_samples].set(1)
    # data_weights = data_weights.at[exclude_samples].set(0)

    # step_path = os.path.join(checkpoint_path, 'data_weights.npy')
    # np.save(step_path, np.array(data_weights))

if __name__ == '__main__':
    app.run(main)