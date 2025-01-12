# in this file: plan
# 1. calculate state, dstate at a given iterate
# 2. calculate the d(data weights) as if we added points at that epsilon
#    for the *entire* training dataset
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from metagradients.smoothness import agreement
import os
from metagradients.vjp import replay_vjp
import numpy as np
import dill
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from metagradients.utils import make_shardings

from .instruction_ds import HetSeqBatch
from .gemma_vjp import OPT_WEIGHTS_LB
from .gemma_vjp import do_gemma_vjp, initialize_data_weights
from ..calculate_lds import grad_from_store
from .gemma_vjp import make_skele_for_bs

BUCKET_SIZE = int(os.environ.get('BUCKET_SIZE', 512))

def get_truncated_train_its(vjp_kw, at_train_iterate):
    num_its = vjp_kw['train_its']
    assert at_train_iterate <= num_its, (at_train_iterate, num_its)
    if at_train_iterate >= 0:
        return at_train_iterate
    else:
        return num_its + at_train_iterate

def get_state_from_vjp_kw(vjp_kw, at_train_iterate, get_stats=False):
    vjp_kw = {k: v for k, v in vjp_kw.items()}
    at_train_iterate = get_truncated_train_its(vjp_kw, at_train_iterate)
    vjp_kw['train_its'] = at_train_iterate
    vjp_kw['forward_only'] = True
    vjp_kw['return_state'] = True
    ret = replay_vjp(**vjp_kw)
    final_state = ret['final_state']
    assert int(final_state.opt_state.count) == at_train_iterate, (int(final_state.opt_state.count), at_train_iterate)
    if not get_stats:
        return final_state
    else:
        return final_state, {
            k: ret[k] for k in ['primal', 'valval_loss']
        }

def make_plus_eps_batcher(train_batcher, at_iter, total_train_its, data_weights):
    naive_batch_maker = train_batcher.func
    bm_kwargs = train_batcher.keywords
    new_kwargs = {k: v for k, v in bm_kwargs.items() if k != 'get_batch'}
    get_batch = bm_kwargs['get_batch']
    def new_get_batch(i):
        if i < at_iter:
            raise ValueError(f'Cannot get batch {i} before truncated it {at_iter}')
        elif i > at_iter:
            return get_batch(i)

        assert i == at_iter
        standard_batch = get_batch(at_iter)
        all_ixs = []
        all_xs = []
        all_ys = []
        for ii in tqdm(range(total_train_its)):
            ixs, (xs, ys) = get_batch(ii)
            # assert isinstance(ixs, np.ndarray)
            # all_ixs.append(ixs)
            # all_xs.extend(xs.xs)
            # all_ys.extend(ys.xs)
            break

        ds = get_batch.keywords['ds']
        for i, xx in ds:
            all_ixs.append(i)
            all_xs.append(xx['input_ids'])
            all_ys.append(xx['labels'])

        bucket_size = xs.bucket_size
        xs_filler = xs.filler
        ys_filler = ys.filler

        all_ixs = np.array(all_ixs) # np.concatenate(all_ixs)
        all_ixs += OPT_WEIGHTS_LB
        def sort_ordering(x):
            _, (x, ix) = x
            x_length = len(x)
            assert ix >= OPT_WEIGHTS_LB
            is_in_dataset = data_weights[ix] > 0
            return x_length - 1e6 * is_in_dataset

        ordering = sorted(list(enumerate(zip(all_xs, all_ixs))), key=sort_ordering)
        ordering = [x[0] for x in ordering]
        all_ixs = all_ixs[ordering]
        all_xs = [all_xs[i] for i in ordering]
        all_ys = [all_ys[i] for i in ordering]

        assert all_ixs.max() < len(data_weights), (all_ixs.max(), len(data_weights))

        oixs, (oxs, oys) = standard_batch
        all_ixs = np.concatenate([oixs, all_ixs])
        all_xs = oxs.xs + all_xs
        all_ys = oys.xs + all_ys

        assert len(all_ixs) == len(all_xs) == len(all_ys), (len(all_ixs), len(all_xs), len(all_ys))

        new_xs = HetSeqBatch(all_xs, bucket_size, xs_filler)
        new_ys = HetSeqBatch(all_ys, bucket_size, ys_filler)
        return all_ixs, (new_xs, new_ys)

    new_kwargs['get_batch'] = new_get_batch
    new_kwargs['num_batches'] = total_train_its
    new_train_batcher = partial(naive_batch_maker, **new_kwargs)
    return new_train_batcher

def do_truncated_vjp(statek, data_weights_to_vjp_kw, at_train_iterate,
                     forward_only, data_weights, return_state):
    vjp_kw = data_weights_to_vjp_kw(data_weights)
    assert len(data_weights) > OPT_WEIGHTS_LB

    vjp_kw['state'] = statek
    # TODO: refactor this to be in terms of the vjp_kw dict
    vjp_kw['forward_only'] = forward_only

    train_batcher = vjp_kw['train_batcher']
    total_train_its = vjp_kw['train_its']
    print('>> Total train its:', total_train_its)
    truncated_train_its = get_truncated_train_its(vjp_kw, at_train_iterate)
    assert int(statek.opt_state.count) == truncated_train_its, (truncated_train_its, final_state.opt_state.count)

    new_batcher = make_plus_eps_batcher(train_batcher, truncated_train_its,
                                        total_train_its=total_train_its,
                                        data_weights=data_weights)
    vjp_kw['train_batcher'] = new_batcher

    sharding, _ = make_shardings()
    state_ixs = range(truncated_train_its, total_train_its)
    vjp_skeles = {}
    for state_ix, batch in zip(state_ixs, new_batcher(truncated_train_its, total_train_its, sharding)):
        if state_ix == truncated_train_its:
            vjp_skeles[state_ix] = make_skele_for_bs(batch.bs)
        else:
            vjp_skeles[state_ix] = make_skele_for_bs(batch.bs, dstate_only=True)

    state_to_vjp_sekele = lambda state: vjp_skeles[int(state.opt_state.count)]
    vjp_kw['vjp_skele'] = state_to_vjp_sekele
    vjp_kw['per_state_skele'] = True
    vjp_kw['return_state'] = True
    res = replay_vjp(**vjp_kw)
    final_state = res['final_state']

    assert int(final_state.opt_state.count) == vjp_kw['train_its'], (int(final_state.opt_state.count), vjp_kw['train_its'])

    if not return_state:
        del res['final_state']

    return res

import traceback

def print_current_traceback():
    print("Current traceback:")
    traceback.print_stack()

# Example usage

def calculate_smoothness_at_iterate(cfg, start_iterate, num_to_select, eps):
    maxits = int(os.environ.get('MAXITS', 0))
    data_weights = initialize_data_weights()
    print('>> Data weights:', data_weights, data_weights.shape)
    def vjp_kw_for_data_weights(data_weights):
        vjp_kw = do_gemma_vjp(data_weights, False, False, cfg, maxits, BUCKET_SIZE,
                              include_test=False, return_final_kw=True)
        return vjp_kw

    vjp_kw = vjp_kw_for_data_weights(data_weights)
    start_iterate = get_truncated_train_its(vjp_kw, start_iterate)

    rng = np.random.default_rng(0)
    # choose 128 slots to upweight, include 32 in dataset
    # some fraction of them will be chosen (roughly 25)
    # 150 * 0.2 chance of being included in dataset
    assert num_to_select < 10_000
    to_upweight = rng.choice(10_000, num_to_select, replace=False) + OPT_WEIGHTS_LB
    eps1_weight = np.copy(data_weights)
    eps1_weight[to_upweight] = eps
    eps2_weight = np.copy(data_weights)
    eps2_weight[to_upweight] = 2 * eps
    zero_weight = np.copy(data_weights)

    statek = get_state_from_vjp_kw(vjp_kw, start_iterate)
    assert int(statek.opt_state.count) == start_iterate, (int(statek.opt_state.count), start_iterate)

    def get_params_for_weights(dw):
        ret = do_truncated_vjp(statek, vjp_kw_for_data_weights, start_iterate,
                               True, dw, return_state=True)
        state = ret['final_state']
        primal = float(ret['primal'])
        return state.params, primal

    params_zero, primal = get_params_for_weights(zero_weight)
    params_eps1, _ = get_params_for_weights(eps1_weight)
    params_eps2, _ = get_params_for_weights(eps2_weight)

    ret = {
        'smoothness': agreement(params_zero, params_eps1, params_eps2),
        'primal': primal
    }

    print('> RESULT: ', ret)

    return ret

def calculate_LDS_at_iterate(cfg, start_iterate, lds_seed=0, trials=8,
                             save_dir=None, maxits=None):
    assert save_dir is not None
    if maxits is None:
        maxits = int(os.environ.get('MAXITS', 0))

    save_dir = Path(save_dir)
    data_weights = initialize_data_weights()
    print('>> Data weights:', data_weights, data_weights.shape)
    def vjp_kw_for_data_weights(data_weights):
        vjp_kw = do_gemma_vjp(data_weights, False, False, cfg, maxits, BUCKET_SIZE,
                                include_test=False, return_final_kw=True)
        return vjp_kw

    # get the state at the start iterate
    statek_path = save_dir / f'statek_{start_iterate}.pkl'
    save_dir.mkdir(parents=True, exist_ok=True)
    print('RETRIEVING STATEK')
    if not statek_path.exists():
        vjp_kw = vjp_kw_for_data_weights(data_weights)
        statek = get_state_from_vjp_kw(vjp_kw, start_iterate)
        with open(statek_path, 'wb') as f:
            dill.dump(statek, f)
    else:
        with open(statek_path, 'rb') as f:
            statek = dill.load(f)

    # now get the deps for adding a point at that iterate
    grad_path = save_dir / f'grad_{start_iterate}.pkl'
    if not grad_path.exists():
        print('RETRIEVING GRAD')
        ret = do_truncated_vjp(statek, vjp_kw_for_data_weights, start_iterate, False,
                               data_weights, False)
        deps = ret['deps']
        batch_indices = ret['batch_indices']
        grad = grad_from_store(deps, batch_indices)
        y0 = float(ret['primal'])
        with open(grad_path, 'wb') as f:
            dill.dump((grad, y0), f)
    else:
        with open(grad_path, 'rb') as f:
            grad, y0 = dill.load(f)

    n = grad.size
    print('>> Grad size', n)

    ys = [y0]
    y_hats = [y0]

    rng = np.random.default_rng(lds_seed)
    data_weights = np.zeros((n,), dtype=np.float32)

    # keep original data weights good
    data_weights[:OPT_WEIGHTS_LB] = 1.

    for _ in range(trials):
        poss_indices_to_add = np.arange(OPT_WEIGHTS_LB, n)
        # poss_indices_to_add = poss_indices_to_add[]
        assert (data_weights[poss_indices_to_add] == 0).all()
        poss_indices_to_add = poss_indices_to_add[grad[poss_indices_to_add] != 0]
        # choose 32 indices
        add_indices = rng.choice(poss_indices_to_add, 32, replace=False)
        print('ADDING INDICES:', add_indices[grad[add_indices] != 0])
        print('ADDING INDICES:', add_indices)
        print('PREDICTED DIFFERENCE:', grad[add_indices].sum())

        this_data_weights = np.copy(data_weights)
        this_data_weights[add_indices] = 1.
        ret = do_truncated_vjp(statek, vjp_kw_for_data_weights, start_iterate, True,
                               this_data_weights, return_state=False)
        ys.append(float(ret['primal']))
        y_hat = y0 + grad @ (this_data_weights - data_weights)
        y_hats.append(float(y_hat))

    ys = np.array(ys)
    y_hats = np.array(y_hats)
    print('>> Ys:', ys)
    print('>> Y hats:', y_hats)

    sr = spearmanr(ys, y_hats)
    pr = pearsonr(ys, y_hats)
    print('>> Spearman:', sr)
    print('>> Pearson:', pr)

    return {
        'spearman': sr,
        'pearson': pr,
        'ys': ys,
        'y_hats': y_hats,
        'loss': y0,
        'grad': grad
    }

def main():
    cfg = SimpleNamespace(
        bs=128,
        lr=0.0003,
        wd=1e-6,
        lr_warmup_its=40,
        b1=0.975,
        b2=0.9875,
        min_lr_relative=1e-8,
        final_min_lr_relative=0.1,
        eps=1e-13,
        eps_sqrt=1e-13,
        selective_wd=True,
        dtype=jnp.float32,
        factored_lr_wd=True,
        anneal_type='linear',
        eps_steps=100,
        eps0=1e-13,
        epochs=1,
        mom_steps=10,
        mom0=1.,
        minibatch_fraction=1.,
        train_frac=0.4,
        train_dataset_name='COMBO',
        val_dataset_name='MMLU',
        data_seed=16,
        model_seed=16,
        lora_d=128,
        lora_std=1.)

    start_iterate = -5
    os.environ['MAXITS'] = '100'

    lds_seed = 0
    trials = 8
    name = os.environ['NAME']
    save_dir = Path('/mnt/xfs/home/engstrom/store/opt_metagradient/') / name
    print('>> SAVING IN DIRECTORY:', save_dir)
    calculate_LDS_at_iterate(cfg, start_iterate, lds_seed, trials, save_dir)

if __name__ == '__main__':
    main()

