import os
import jax
import flax
import numpy as np
import tensorflow as tf
from datetime import datetime

from functools import partial
from functools import cache
from scipy.stats import spearmanr, pearsonr

# from domains.vjp_lm import vjp_lm
# from domains.vjp_blocks import one_sample_vjp_head, sample_loss_vjp_head, \
#     example_loss_vjp_skeleton

from octo.utils.jax_utils import initialize_compilation_cache

from make_loader import make_split_loader_and_data_weights, make_replay_dataset
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

import flags_config
FLAGS = flags.FLAGS

EPS = 1.0000000000000001e-11
SEED_SPACING = 100000

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
                       divisor=None):

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
        jax.debug.print('---{dw}',dw=these_data_weights)
        action_loss = action_loss * these_data_weights

    return action_loss / divisor

def compute_datamodels_lds():

    initialize_compilation_cache()
    devices = jax.devices()

    # # create a 1D mesh with a single axis named "batch"
    # mesh = Mesh(jax.devices(), axis_names="batch")
    # # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    # dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    # replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    ######### need to implement this #########
    # if test_sample is None:
    #     vjp_head = sample_loss_vjp_head
    # else:
    #     vjp_head = partial(one_sample_vjp_head, test_index=test_sample)

    # vjp_skele = make_vjp_skele(bs)
    ##########################################

    ######### need to implement this #########
    # model, params = model_maker(model_seed)
    ##########################################

    _, data_weights = make_replay_dataset(0, 1e5, None, train=True, return_dw_only=True)
    train_batcher = partial(make_split_loader_and_data_weights, mode='train', seed=FLAGS.config.seed)
    # train_its = FLAGS.config.num_steps
    # train_its = 1_000
    # train_its = 10
    train_its = 2

    # # quick test to see if this works
    # for batch in train_batcher(0, 5, ""):
    #     for item in batch.get_minibatches('train'):
    #         pass

    val_batcher = partial(make_split_loader_and_data_weights, mode='val', seed=FLAGS.config.seed)
    # val_its = FLAGS.config.num_val_steps
    val_its = 1

    # # quick test to see if this works
    # for batch in val_batcher(0, 5, ""):
    #     for item in batch.get_minibatches('val'):
    #         bp()
    #         pass

    model, frozen_params, trainable_params = make_model()

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

    # # tesing per-sample loss fn
    # for batch in train_batcher(0, 5, ""):
    #     for minibatch in batch.get_minibatches('train'):
    #         break
    #     break

    # psl(
    #     params=trainable_params,
    #     batch=minibatch,
    #     data_weights=data_weights,
    #     train=True
    # )

    optimizer_dict = FLAGS.config.optimizer.to_dict()
    lr_scheduler_dict = optimizer_dict['learning_rate']

    OPTIMIZER_KWARGS = {
        'lr': lr_scheduler_dict['peak_value'],
        'wd': 1e-5,
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
        'eps_schedule': jax.tree_util.Partial(interp_from, steps=200, eps0=1e-08,
                                        eps_root0=1e-08, space='geometric'),
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

    # state0 = optimizer_maker(params, train_its)

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
        n_train_ba=train_its,
        n_val_ba=val_its,
        aux_datasets=aux_datasets,
        return_state=return_state,
        # forward_only=True,
        forward_only=False,
    )

    ret = vjp_robodm(**vjp_kw)

    y0 = float(ret['primal'])
    deps = ret['deps']
    batch_indices = ret['batch_indices']
    num_datapoints = data_weights.size

    # merging final params
    final_params = ret['final_state'].params

    final_params = flax.traverse_util.flatten_dict(final_params)
    flat_frozen_params = flax.traverse_util.flatten_dict(frozen_params)

    all_params = final_params | flat_frozen_params
    all_params = flax.traverse_util.unflatten_dict(all_params)

    model = model.replace(params=all_params)

    # save model
    formatted_date_time = datetime.now().strftime("%d-%b-%Y_%I-%M-%S%p").lower()
    checkpoint_path = os.path.join(FLAGS.config.save_dir, FLAGS.config.dataset_kwargs.name, formatted_date_time)

    bp()
    model.save_pretrained(step=train_its, checkpoint_path=checkpoint_path)
    # model.load_pretrained(step=train_its, checkpoint_path=checkpoint_path)

    raise NotImplementedError

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

    return lds_for_run(y0, num_datapoints, drop_frac, lds_seed, grad,
                       data_weights, vjp_kw)

def lds_for_run(y0, num_datapoints, drop_frac, lds_seed, grad, data_weights,
                vjp_kw, num_trials=20):
    raise NotImplementedError
    ys = [y0]
    y_hats = [y0]

    num_leave_out = int(num_datapoints * drop_frac)
    rng = np.random.default_rng(lds_seed + SEED_SPACING)
    for _ in range(num_trials):
        drop_indices = rng.choice(num_datapoints, num_leave_out, replace=False)
        this_data_weights = data_weights.at[drop_indices].set(0)
        this_jvp_kw = {k: v for k, v in vjp_kw.items()}
        this_jvp_kw.update(dict(
            data_weights=this_data_weights,
            forward_only=True
        ))

        ret = vjp_lm(**this_jvp_kw)
        ys.append(float(ret['primal']))

        y_hat = y0 + grad @ (this_data_weights - data_weights)
        y_hats.append(float(y_hat))

    ys = np.array(ys)
    y_hats = np.array(y_hats)

    sr = spearmanr(ys, y_hats)
    pr = pearsonr(ys, y_hats)
    print("Spearman:", sr)
    print("Pearson:", pr)
    import pdb; pdb.set_trace()
    return sr, pr

def grad_from_store(deps, batch_indices):
    import numpy as np
    flat_deps = {k: v for d in deps.values() for k, v in d.items()}
    num_datapoints = max(b.max() for b in batch_indices) + 1
    gradient = np.zeros((num_datapoints,), dtype=np.float32)

    for i, bixs in enumerate(batch_indices):
        gradient[bixs] += flat_deps[i]

    return gradient

def main(_):
    # compute_datamodels_lds(model_seed,
    #                        data_seed,
    #                        test_sample,
    #                        bs,
    #                        drop_frac,
    #                        lds_seed,
    #                        load_and_data_weight_maker,
    #                        model_maker,
    #                        optimizer_maker)

    compute_datamodels_lds()

if __name__ == '__main__':
    app.run(main)