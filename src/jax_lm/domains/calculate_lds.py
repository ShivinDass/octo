import jax
import numpy as np
from domains.vjp_lm import vjp_lm
from domains.vjp_blocks import one_sample_vjp_head, sample_loss_vjp_head, \
    example_loss_vjp_skeleton
from functools import partial
from functools import cache
from scipy.stats import spearmanr, pearsonr
import os

SEED_SPACING = 100000

@cache
def make_vjp_skele(bs):
    return jax.tree_util.Partial(partial(example_loss_vjp_skeleton, bs=bs))

def calculate_lds(model_seed, data_seed, test_sample, bs, drop_frac, lds_seed,
                  load_and_data_weight_maker, model_maker, optimizer_maker):
    if test_sample is None:
        vjp_head = sample_loss_vjp_head
    else:
        vjp_head = partial(one_sample_vjp_head, test_index=test_sample)

    vjp_skele = make_vjp_skele(bs)
    model, params = model_maker(model_seed)

    ret = load_and_data_weight_maker(data_seed)
    train_batcher, val_batcher, data_weights, train_its, val_its = ret

    state0 = optimizer_maker(params, train_its)

    vjp_kw = dict(state=state0, vjp_head=vjp_head, vjp_skele=vjp_skele,
                  data_weights=data_weights, return_kw=False,
                  train_batcher=train_batcher, val_batcher=val_batcher,
                  model=model, n_train_ba=train_its, n_val_ba=val_its,
                  aux_datasets={}, forward_only=False, bs=bs, return_state=True)

    ret = vjp_lm(**vjp_kw)
    y0 = float(ret['primal'])
    deps = ret['deps']
    final_state = ret['final_state']
    batch_indices = ret['batch_indices']
    num_datapoints = data_weights.size

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

    ret = lds_for_run(y0, num_datapoints, drop_frac, lds_seed, grad,
                      data_weights, vjp_kw)
    sr, pr, yhat, y_true, all_drop_indices = ret
    # dict with: 
    # - train_ds: tuple of train_loader_fn and n_train_iter
    # - val_ds: tuple of val_loader_fn and n_val_iter
    # - test_indices: indices to test set
    # - domain: 'wikitext' or 'cifar'
    # - params: model parameters WITHOUT state
    to_save = {
        'train_ds': (train_batcher, train_its),
        'val_ds': (val_batcher, val_its),
        'test_indices': np.arange(val_its * bs),
        'trial_index': test_sample,
        'domain': 'wikitext',
        'params': final_state.params,
        'yhat_mg': yhat,
        'y_true': y_true,
        'all_drop_indices': all_drop_indices,
        'spearman': sr,
        'pearson': pr
    }

    return to_save

def lds_for_run(y0, num_datapoints, drop_frac, lds_seed, grad, data_weights,
                vjp_kw, num_trials=10):
    ys = [y0]
    y_hats = [y0]

    num_leave_out = int(num_datapoints * drop_frac)
    rng = np.random.default_rng(lds_seed + SEED_SPACING)
    all_drop_indices = []
    for _ in range(num_trials):
        drop_indices = rng.choice(num_datapoints, num_leave_out, replace=False)
        all_drop_indices.append(drop_indices)
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
    return sr, pr, y_hats, ys, all_drop_indices

def grad_from_store(deps, batch_indices):
    flat_deps = {k: v for d in deps.values() for k, v in d.items()}
    num_datapoints = max(b.max() for b in batch_indices.values()) + 1
    gradient = np.zeros((num_datapoints,), dtype=np.float32)

    for batch_n, indices in batch_indices.items():
        print('>> RESTORING: Batch', batch_n, 'has', len(indices), 'indices')
        gradient[indices] += flat_deps[batch_n]

    return gradient
