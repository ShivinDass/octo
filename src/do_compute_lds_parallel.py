import os
import gc
import jax
import pickle
import jax.numpy as jnp
import flax
import numpy as np
import dill
from copy import deepcopy
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime

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
    raise NotImplementedError
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

def lds_for_run(y0, num_datapoints, num_drop, lds_seed, grad, data_weights,
                vjp_kw, num_trials=20):

    ys = [y0]
    y_hats = [y0]

    candidate_grad = grad[1_000_000:]
    candidate_samples = 1_000_000 + np.where(candidate_grad != 0)[0]
    # num_leave_out = int(num_datapoints * drop_frac)
    # num_leave_out = int(len(candidate_samples) * drop_frac)
    num_leave_out = num_drop
    rng = np.random.default_rng(lds_seed + SEED_SPACING)
    all_drop_indices = []
    for _ in range(num_trials):
        # drop_indices = rng.choice(num_datapoints, num_leave_out, replace=False)
        drop_indices = rng.choice(candidate_samples, num_leave_out, replace=False)
        all_drop_indices.append(drop_indices)
        # this_data_weights = data_weights.at[drop_indices].set(0)
        this_data_weights = data_weights.at[drop_indices].set(1)
        this_jvp_kw = {k: v for k, v in vjp_kw.items()}
        this_jvp_kw.update(dict(
            data_weights=this_data_weights,
            forward_only=True
        ))

        # ret = vjp_lm(**this_jvp_kw)
        ret = vjp_robodm(**this_jvp_kw)
        ys.append(float(ret['primal']))

        y_hat = y0 + grad @ (this_data_weights - data_weights)
        y_hats.append(float(y_hat))

    ys = np.array(ys)
    y_hats = np.array(y_hats)

    sr = spearmanr(ys, y_hats)
    pr = pearsonr(ys, y_hats)
    print("Spearman:", sr)
    print("Pearson:", pr)

    ret = dict(
        sr=sr,
        pr=pr,
        ys=ys,
        y_hats=y_hats,
        all_drop_indices=all_drop_indices
    )

    # return sr, pr, y_hats, ys, all_drop_indices
    return ret

def setup_and_compute_lds(data_weights: jax.numpy.array,
                          checkpoint_path: str):

    initialize_compilation_cache()
    devices = jax.devices()

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    print('Checkpoint path:', checkpoint_path)
    FLAGS.config.checkpoint_path = checkpoint_path

    train_batcher = partial(make_split_loader_and_data_weights, mode='train', seed=FLAGS.config.seed)
    train_its = FLAGS.config.num_steps
    # train_its = 20
    # train_its = 2

    bob_its = FLAGS.config.bob_steps
    forward_its = train_its - bob_its

    val_batcher = partial(make_split_loader_and_data_weights, mode='val', seed=FLAGS.config.seed)
    # val_its = FLAGS.config.num_val_steps
    val_its = 1
    # val_batcher(0, 5, "")

    # load model
    model, frozen_params, trainable_params = make_model(train_batcher)
    model.load_pretrained(checkpoint_path=checkpoint_path)

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
    return_state = True
    return_kw = False

    sharding, replicated_sharding = make_shardings()
    head_val_batcher = jax.tree_util.Partial(val_batcher, sharding=sharding)

    vjp_skele = jax.tree_util.Partial(partial(example_loss_vjp_skeleton, bs=FLAGS.config.batch_size))
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
        # forward_only=False,
    )

    grad_path = os.path.join(checkpoint_path, 'datamodels.npy')
    grad = np.load(grad_path)

    state_path = os.path.join(checkpoint_path, 'state.dill')
    with open(state_path, "rb") as file:
        final_ret = dill.load(file)

    y0 = float(final_ret['primal'])
    num_datapoints = data_weights.size
    lds_seed = FLAGS.config.seed

    num_drops = [32, 64, 128, 256]

    vjp_kw.update(dict(
        state=final_ret['final_state'],
        n_train_ba=train_its,
        forward_only=False,
    ))

    ldses = []
    for num_drop in num_drops:
        lds_res = lds_for_run(y0,
                              num_datapoints,
                            #   drop_frac,
                              num_drop,
                              lds_seed,
                              grad,
                              data_weights,
                              vjp_kw,
                              num_trials=5)

        ldses.append(lds_res)

    return {k: v for k, v in zip(num_drops, ldses)}

def main(_):

    job_id = FLAGS.config.job_id

    # formatted_date_time = datetime.now().strftime("%d-%b-%Y_%I-%M-%S%p").lower()
    # formatted_date_time = 'test_data_selection_fast'
    # formatted_date_time = 'test_data_selection_slow'
    # formatted_date_time = 'test_data_selection_slow_more_iters'
    formatted_date_time = FLAGS.config.folder_name
    meta_checkpoint_path = os.path.join(FLAGS.config.save_dir, FLAGS.config.dataset_kwargs.name, formatted_date_time)

    checkpoint_path = os.path.join(meta_checkpoint_path, f'iter_{job_id}')
    assert os.path.exists(checkpoint_path)

    if job_id == 0:
        _, data_weights = make_replay_dataset(0, 1e5, None, train=True, return_dw_only=True)
        data_weights = jax.numpy.concatenate(
            [data_weights, jax.numpy.zeros_like(data_weights)],
            axis=0
        )

    else:
        prev_job = job_id - 1
        prev_dw_path = os.path.join(meta_checkpoint_path, f'iter_{prev_job}', 'data_weights.npy')
        assert os.path.exists(prev_dw_path)

        data_weights = jnp.array(
            np.load(prev_dw_path)
        )

    ldses = setup_and_compute_lds(data_weights, checkpoint_path)
    ldses_path = os.path.join(checkpoint_path, 'ldses.pkl')

    with open(ldses_path, 'wb') as f:
        pickle.dump(ldses, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    app.run(main)