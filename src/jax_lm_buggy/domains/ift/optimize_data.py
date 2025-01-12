import jax
import jax.numpy as jnp
from tqdm import tqdm
from types import SimpleNamespace
import dill as pickle
import os
import numpy as np
from types import SimpleNamespace
from domains.ift.instruction_ds import loader_from
from functools import partial
from .opt_metagradient import OPT_WEIGHTS_LB
from .instruction_ds import HetSeqBatch
import scipy as sp
from metagradients.vjp import replay_vjp

def flatten_data_func(data_func, num_its):
    all_ixs = []
    all_xs = []
    all_ys = []
    for ii in tqdm(range(num_its)):
        ixs, (xs, ys) = data_func(ii)
        assert isinstance(ixs, np.ndarray)
        all_ixs.append(ixs)
        all_xs.extend(xs.xs)
        all_ys.extend(ys.xs)

    all_ixs = np.concatenate(all_ixs)
    metadata = {
        'xs_filler': xs.filler,
        'ys_filler': ys.filler,
        'bucket_size': xs.bucket_size
    }

    return all_ixs, all_xs, all_ys, metadata

def _make_new_indices(data_counts, data_seed, bs):
    assert data_counts.dtype == np.int64
    assert data_counts.min() >= 0

    flat_counts = []
    for i in range(len(data_counts)):
        flat_counts.extend([i] * int(data_counts[i]))

    flat_counts = np.array(flat_counts, dtype=np.int64)

    # round to nearest bs
    num_sampled = len(flat_counts)
    num_sampled = (num_sampled // bs) * bs
    print('>> Shaving from %d to %d' % (len(flat_counts), num_sampled))
    flat_counts = flat_counts[:num_sampled]

    # shuffle + check shuffle
    rng = np.random.RandomState(data_seed)
    orig_fc0 = flat_counts[0]
    rng.shuffle(flat_counts)
    assert flat_counts[0] != orig_fc0
    return flat_counts

from .instruction_ds import INPUT_ID_FILLER, LABEL_FILLER

def loader_for_weights(data_counts, orig_loader, maxits, data_seed,
                       insert_plus_eps_at_iterate, bs, data_weights):
    assert insert_plus_eps_at_iterate > 0, insert_plus_eps_at_iterate
    # make dataset with two kinds of data:
    # 1. primal data with original indices
    # 2. vjp-only data with new indices simulating adding +eps at a given iterate
    # The second class of data is offset by OPT_WEIGHTS_LB in index space

    flat_counts = _make_new_indices(data_counts, data_seed, bs)
    primal_train_its = len(flat_counts) // bs
    assert len(flat_counts) % bs == 0

    # make primal get_batch by swapping out the indices
    orig_get_batch = orig_loader.keywords['get_batch']
    get_batch_keywords = {k:v for k, v in orig_get_batch.keywords.items()}
    del get_batch_keywords['indices']
    get_batch_keywords['indices'] = flat_counts
    # import pdb; pdb.set_trace()
    primal_get_batch = partial(loader_from, **get_batch_keywords)

    # make vjp-only get_batch manually
    def overall_get_batch(it):
        if int(it) != int(insert_plus_eps_at_iterate):
            return primal_get_batch(it)

        dataset = orig_get_batch.keywords['ds']
        bucket_size = orig_get_batch.keywords['bucket_size']
        # ret = flatten_data_func(orig_get_batch, num_its)
        # all_ixs, all_xs, all_ys, metadata = ret
        if maxits not in [0, -1, None]:
            dataset = dataset[:maxits * 128]

        all_ixs = np.arange(0, len(dataset))
        all_xs = [dataset[int(i)][1]['input_ids'] for i in all_ixs]
        all_ys = [dataset[int(i)][1]['labels'] for i in all_ixs]

        metadata = {
            'xs_filler': INPUT_ID_FILLER,
            'ys_filler': LABEL_FILLER,
            'bucket_size': bucket_size
        }

        # make sure the indices are the same
        assert np.max(all_ixs) < OPT_WEIGHTS_LB
        all_ixs += OPT_WEIGHTS_LB

        def sort_ordering(x):
            _, (x, ix) = x
            x_length = len(x)
            assert ix >= OPT_WEIGHTS_LB
            is_in_dataset = data_weights[ix] > 0
            # sort by (a) in dataset and then (b) by index
            return x_length - 1e6 * is_in_dataset

        ordering = sorted(list(enumerate(zip(all_xs, all_ixs))), key=sort_ordering)
        ordering = [x[0] for x in ordering]
        all_ixs = all_ixs[ordering]
        all_xs = [all_xs[i] for i in ordering]
        all_ys = [all_ys[i] for i in ordering]

        assert all_ixs.max() < len(data_weights), (all_ixs.max(), len(data_weights))

        oixs, (oxs, oys) = primal_get_batch(it)
        all_ixs = np.concatenate([oixs, all_ixs])
        all_xs = oxs.xs + all_xs
        all_ys = oys.xs + all_ys

        assert len(all_ixs) == len(all_xs) == len(all_ys), (len(all_ixs), len(all_xs), len(all_ys))

        bucket_size = metadata['bucket_size']
        xs_filler = metadata['xs_filler']
        ys_filler = metadata['ys_filler']

        new_xs = HetSeqBatch(all_xs, bucket_size, xs_filler)
        new_ys = HetSeqBatch(all_ys, bucket_size, ys_filler)
        return all_ixs, (new_xs, new_ys)

    new_loader_kw = {k:v for k, v in orig_loader.keywords.items()}
    new_loader_kw['num_batches'] = primal_train_its
    new_loader_kw['get_batch'] = overall_get_batch
    new_train_loader = partial(orig_loader.func, **new_loader_kw)
    assert not 'func' in dir(orig_loader.func)

    return new_train_loader, primal_train_its

from .opt_metagradient import initialize_data_weights, do_gemma_vjp, \
    get_state_from_vjp_kw
from metagradients.utils import make_shardings
from .gemma_vjp import make_skele_for_bs
from ..calculate_lds import grad_from_store

def do_truncated_vjp(state, vjp_kw):
    vjp_kw = {k: v for k, v in vjp_kw.items()}
    vjp_kw['state'] = state
    sharding, _ = make_shardings()
    truncated_train_its = int(state.opt_state.count)
    total_train_its = vjp_kw['train_its']
    state_ixs = range(truncated_train_its, total_train_its)
    vjp_skeles = {}
    new_batcher = vjp_kw['train_batcher']
    for state_ix, batch in zip(state_ixs, new_batcher(truncated_train_its, total_train_its, sharding)):
        if state_ix == truncated_train_its:
            vjp_skeles[state_ix] = make_skele_for_bs(batch.bs)
        else:
            vjp_skeles[state_ix] = make_skele_for_bs(batch.bs, dstate_only=True)

    state_to_vjp_sekele = lambda state: vjp_skeles[int(state.opt_state.count)]
    vjp_kw['vjp_skele'] = state_to_vjp_sekele
    vjp_kw['per_state_skele'] = True
    vjp_kw['return_state'] = True
    return replay_vjp(**vjp_kw)

def make_vjp_kw(train_cfg, data_weights, bucket_size, maxits):
    vjp_kw = do_gemma_vjp(data_weights, return_state=True, forward_only=False,
                          cfg=train_cfg, maxits=maxits, bucket_size=bucket_size,
                          include_test=False, return_final_kw=True)
    return vjp_kw


def initialize_data_counts(train_cfg):
    vjp_kw = make_vjp_kw(train_cfg, initialize_data_weights(), 128, maxits=0)
    loader = vjp_kw['train_batcher']
    get_batch = loader.keywords['get_batch']
    dataset = get_batch.keywords['ds']
    num_datapoints = len(dataset)
    return np.ones(num_datapoints, dtype=np.int64)

def vjp_for_data_weights(data_counts, train_cfg, vjp_iterate, data_seed,
                         maxits, bucket_size=128, no_grad=False):
    assert data_counts.dtype == np.int64
    data_weights = initialize_data_weights()

    print('*' * 80)
    print('VJP FOR CFG: ', train_cfg)
    print('*' * 80)

    vjp_kw = make_vjp_kw(train_cfg, data_weights, bucket_size, maxits)
    bs = train_cfg.bs
    assert data_counts.min() >= 0
    num_samples = data_counts.sum()
    num_its = num_samples // bs
    vjp_iterate = vjp_iterate if vjp_iterate > 0 else num_its + vjp_iterate

    new_train_loader, new_train_its = loader_for_weights(data_counts,
                                                         vjp_kw['train_batcher'],
                                                         vjp_kw['train_its'],
                                                         data_seed, vjp_iterate,
                                                         train_cfg.bs,
                                                         data_weights)
    assert new_train_its == num_its, (new_train_its, num_its)
    if maxits not in [0, -1, None]:
        new_train_its = min(maxits, new_train_its)

    print(f'>> Setup:')
    print(f'    >> statek: train from 0 to {vjp_iterate}')
    print(f'    >> vjp: {vjp_iterate} to {new_train_its} (total {new_train_its - vjp_iterate})')


    vjp_kw.update({
        'train_batcher': new_train_loader,
        'train_its': new_train_its
    })

    if not no_grad:
        statek = get_state_from_vjp_kw(vjp_kw, vjp_iterate)
    else:
        statek, stats = get_state_from_vjp_kw(vjp_kw, new_train_its, get_stats=True)
        return {
            'primal': stats['primal'],
            'valval_loss': stats['valval_loss'],
            'statek': statek,
            'final_state': statek
        }

    print('>> CALCULATING METAGRADIENT')
    ret = do_truncated_vjp(statek, vjp_kw)
    deps = ret['deps']
    batch_indices = ret['batch_indices']
    grad = grad_from_store(deps, batch_indices)
    grad = grad[OPT_WEIGHTS_LB:]

    return {
        'grad': grad,
        'data_weights': data_weights,
        'statek': statek,
        'final_state': ret['final_state'],
        'primal': ret['primal'],
        'valval_loss': ret['valval_loss']
    }

class Optimizer:
    def __init__(self, state, params):
        # store stuff
        raise NotImplementedError

    def step(self, g):
        # returns next state given gradient
        raise NotImplementedError

    @classmethod
    def init(cls, **kwargs):
        # returns initial state
        raise NotImplementedError

import optax
# THESE MINIMIZE
class SGDOptimizer(Optimizer):
    def __init__(self, state, params):
        self.state = state
        self.params = params

    def step(self, g):
        updates, opt_state = optax.sgd(g, self.state, self.params)
        params = optax.apply_updates(self.params, updates)
        return SGDOptimizer(opt_state, params)

    @classmethod
    def init(cls, params, steps, max_lr, final_lr, momentum):
        sched = optax.linear_schedule(max_lr, final_lr, steps)
        solver = optax.sgd(learning_rate=sched, momentum=momentum)
        opt_state = solver.init(params)
        return cls(opt_state, params)

def _zeroone_initialize(num_datapoints, frac_included, seed):
    rng = np.random.RandomState(seed)
    num_to_include = int(frac_included * num_datapoints)
    indices = np.arange(num_datapoints)
    rng.shuffle(indices)
    included_indices = indices[:num_to_include]
    data_weights = np.zeros(num_datapoints, dtype=np.float32)
    data_weights[included_indices] = 1
    return data_weights

def _latent_initialize(num_datapoints, frac_included, seed, std=0.5):
    # solve for mean for which normal distribution with std 1 has frac_included samples > 0.5
    # then sample from this distribution
    rng = np.random.RandomState(seed)
    new_mean = 0.5 - sp.stats.norm.ppf(1 - frac_included) * std
    return rng.normal(new_mean, std, num_datapoints).astype(np.float32)

def initialize_data_latent(num_datapoints, strategy, frac_included, seed):
    assert strategy in ['greedy_flip', 'l1_normed', 'sgd', 'random_descent']
    if strategy == 'greedy_flip' or strategy == 'random_descent':
        return _zeroone_initialize(num_datapoints, frac_included, seed)
    else:
        return _latent_initialize(num_datapoints, frac_included, seed)

class ProjectedOptimizer(Optimizer):
    def __init__(self, state, params):
        self.state = state
        self.params = params

    def step(self, g):
        constraint_for_this_step = self.state.constraint_at(self.state.count)
        projector = self.state.projector
        params = projector(g, self.params, constraint_for_this_step)
        new_state = SimpleNamespace(**{
            'projector': projector,
            'constraint_at': self.state.constraint_at,
            'count': self.state.count + 1
        })

        return ProjectedOptimizer(new_state, params)

    @classmethod
    def init(cls, params, strategy, constraint_steps, constraint_start,
             constraint_end):
        assert strategy in ['greedy_flip', 'l1_normed', 'random_descent']
        def constraint_at(count):
            y0 = constraint_start
            y1 = constraint_end
            x0 = 0
            x1 = constraint_steps

            return y0 + (y1 - y0) * (count - x0) / (x1 - x0)

        def projector(g, params, constraint):
            if strategy == 'greedy_flip':
                # give biggest step according to l0 constraint
                # only remove the top constraint fraction of weights
                # sort by magnitude
                g_abs = np.absolute(g)
                indices = np.argsort(g_abs)
                assert g_abs[indices[-1]] == g_abs.max()
                num_to_flip = int(constraint * len(g))
                assert num_to_flip > 0
                indices_to_flip = indices[-num_to_flip:]
                new_params = params.copy()
                new_params[indices_to_flip] -= np.sign(g[indices_to_flip]).astype(np.float32)
            elif strategy == 'random_descent':
                assert constraint < 1.0
                num_to_flip = int(constraint * len(g))
                indices = np.arange(len(g))
                np.random.shuffle(indices)
                indices_to_flip = indices[:num_to_flip]
                new_params = params.copy()
                new_params[indices_to_flip] -= np.sign(g[indices_to_flip]).astype(np.float32)
            elif strategy == 'l1_normed':
                # give biggest step according to l1 constraint
                g_normalized = g / np.linalg.norm(g, 1)
                new_params = params - constraint * g_normalized

            return new_params

        state = SimpleNamespace(**{
            'projector': projector,
            'constraint_at': constraint_at,
            'count': 0
        })

        return cls(state, params)

def optimize(train_cfg, strategy, init_frac_included, steps, max_lr, final_lr,
             vjp_iterate, data_seed, maxits=0, bucket_size=None, initial_g=None):
    if bucket_size is None:
        bucket_size = int(os.environ.get('BUCKET_SIZE', 128))

    initial_counts = initialize_data_counts(train_cfg)
    latent_params = initialize_data_latent(len(initial_counts), strategy,
                                           init_frac_included,
                                           seed=data_seed)
    if strategy == 'sgd':
        optimizer = SGDOptimizer.init(latent_params, steps, max_lr, final_lr)
    elif strategy in ['greedy_flip', 'l1_normed', 'random_descent']:
        optimizer = ProjectedOptimizer.init(latent_params, strategy, steps,
                                            max_lr, final_lr)
    else:
        raise ValueError(f'strategy {strategy} not recognized')

    results = {
        'old_params': [],
        'new_params': [],
        'primal': [],
        'valval_loss': [],
        'grad': [],
        'state': []
    }

    def grad_for(params, data_seed, no_grad=False):
        this_counts = np.round(params)
        this_counts = np.clip(this_counts, 0, 10).astype(np.int64)
        if not no_grad:
            ret = vjp_for_data_weights(this_counts, train_cfg, vjp_iterate,
                                        data_seed, maxits, bucket_size)
            primal, valval_loss, grad = ret['primal'], ret['valval_loss'], ret['grad']
            final_state = ret['final_state']
            return primal, valval_loss, grad, final_state
        else:
            return vjp_for_data_weights(this_counts, train_cfg, vjp_iterate,
                                        data_seed, maxits, bucket_size,
                                        no_grad=True)

    one_step_mode = initial_g is not None
    if one_step_mode:
        steps = 1

    for i in range(steps):
        old_params = optimizer.params
        # this_counts = np.round(optimizer.params)
        # this_counts = np.clip(this_counts, 0, 10).astype(np.int64)
        # if initial_g is not None and i == 0:
        #     primal, valval_loss, grad = 0.001, 0.001, initial_g
        # else:
        #     ret = vjp_for_data_weights(this_counts, train_cfg, vjp_iterate,
        #                                data_seed + i, maxits, bucket_size)
        #     primal, valval_loss, grad = ret['primal'], ret['valval_loss'], ret['grad']
        if initial_g is not None and i == 0:
            primal, valval_loss, grad, state = 0.001, 0.001, initial_g, None
        elif not one_step_mode:
            ret = grad_for(optimizer.params, data_seed + i)
            primal, valval_loss, grad, state = ret
        else:
            raise ValueError('One step mode not implemented')

        final_grad = np.zeros(len(latent_params), dtype=np.float32)
        final_grad[:len(grad)] = grad

        optimizer = optimizer.step(final_grad)

        # LOGGING
        delta_params = optimizer.params - old_params
        print('-' * 80)
        print(f'********** Iterate @ {i}: val loss {primal}, valval loss {valval_loss}')
        ming, maxg, avgg, mediang = np.min(grad), np.max(grad), np.mean(np.absolute(grad)), np.median(grad)
        print(f'********** Grad stats: min {ming:.5f}, max {maxg:.5f}, avg {avgg:.5f}, median {mediang:.5f}')
        mind, maxd, avgd = np.min(delta_params), np.max(delta_params), np.mean(np.absolute(delta_params))
        print(f'********** Delta param stats: min {mind:.5f}, max {maxd:.5f}, avg {avgd:.5f}')
        print('-' * 80)

        results['old_params'].append(old_params)
        results['new_params'].append(optimizer.params)
        results['primal'].append(primal)
        results['valval_loss'].append(valval_loss)
        results['grad'].append(grad)
        results['state'].append(state)

    # evaluate at the end
    # ret = vjp_for_data_weights(this_counts, train_cfg, vjp_iterate,
    #                            # data_seed + i, maxits, bucket_size)
    # primal, valval_loss, grad = ret['primal'], ret['valval_loss'], ret['grad']
    if one_step_mode or steps > 1:
        ret = grad_for(optimizer.params, data_seed + 1999, no_grad=True)
        primal, valval_loss, final_state = ret['primal'], ret['valval_loss'], ret['final_state']
        print('>> FINAL PRIMAL/VALVAL LOSS', primal, valval_loss)
        results['primal'].append(primal)
        results['valval_loss'].append(valval_loss)
        results['state'].append(final_state)

    return results

# TODO:
# [ ] MINIBATCHING

def run_optimize(strategy, init_frac_included, steps, max_lr, final_lr, vjp_iterate,
                 train_data_frac, task, data_seed, root_eps, b1, b2, model_lr,
                 wd, LDS, bs, lora_width, lora_std, start_grad_path=None,
                 warmup_its=None, final_model_lr=None):
    print("*** RUNNING OPTIMIZE WITH LOCALS", locals())
    DEBUG = int(os.environ.get('DEBUG', 0))
    if DEBUG and False:
        vjp_iterate = 5
        maxits = 10
        bucket_size = 1024
        steps = 2
    else:
        maxits = 0
        bucket_size = 128

    eps = root_eps
    cfg = SimpleNamespace(
        bs=bs,
        lr_warmup_its=warmup_its,
        min_lr_relative=1e-6,
        selective_wd=True,
        dtype='float32',
        factored_lr_wd=True,
        anneal_type='linear',
        epochs=1,
        mom_steps=10,
        minibatch_fraction=1.,
        train_dataset_name='COMBO',
        val_dataset_name=task,
        data_seed=16,
        model_seed=16,
        lora_d=lora_width,
        lora_multiple=512.)

    update = {
        'train_frac': train_data_frac,
        'b1': b1,
        'b2': b2,
        'eps_steps': 100,
        'eps': eps,
        'final_min_lr_relative': final_model_lr,
        'eps_sqrt': eps,
        'eps0': eps,
        'mom0': 1,
        'lora_std': lora_std,
        'lr': model_lr,
        'wd': wd
    }

    train_cfg = SimpleNamespace(**(cfg.__dict__ | update))
    print('FINAL TRAIN CFG', train_cfg)
    if LDS:
        from .opt_metagradient import calculate_LDS_at_iterate
        from uuid import uuid4
        from pathlib import Path
        save_dir = Path('/tmp/' + str(uuid4()))
        return calculate_LDS_at_iterate(train_cfg, vjp_iterate, 0,
                                        save_dir=save_dir, maxits=maxits)

    # initial_g = None if start_grad_path is None else np.load(start_grad_path)
    if start_grad_path is None:
        initial_g = None
    else:
        with open(start_grad_path, 'rb') as f:
            # if isinstan
            initial_g = pickle.load(f)
            if isinstance(initial_g, tuple):
                initial_g = initial_g[1]['grad'][0]

    return optimize(train_cfg, strategy, init_frac_included, steps, max_lr,
                    final_lr, vjp_iterate, data_seed, maxits, bucket_size,
                    initial_g)

