from .generalized_project import lowrank_vjp_with_lds
import numpy as np
import jax
import os
from .calculate_lds import make_vjp_skele
from .wikitext_lds import BS, MODEL_SEED, DATA_SEED, model_maker, \
    make_loaders_and_data_weights, make_wikitext_optimizer
from .vjp_lm import vjp_lm
from metagradients.vjp import replay_vjp

def to_vjp(vjp_head, return_state=False, forward_only=False, drop_frac=None,
           drop_seed=None, return_data_weight=False, get_vjp_head_kw=False):
    vjp_skele = make_vjp_skele(BS)

    model, params = model_maker(MODEL_SEED)
    ret = make_loaders_and_data_weights(DATA_SEED)
    train_batcher, val_batcher, data_weights, train_its, val_its = ret
    state0 = make_wikitext_optimizer(params, train_its)
    vjp_lm_kw = dict(state=state0, vjp_head=vjp_head, vjp_skele=vjp_skele, 
                     data_weights=data_weights,
                     train_batcher=train_batcher, val_batcher=val_batcher,
                     model=model, n_train_ba=train_its, n_val_ba=val_its,
                     aux_datasets={}, return_kw=True,
                     return_state=return_state, forward_only=forward_only)

    # data_weights is all 1 rn
    if drop_frac is not None:
        assert drop_seed is not None
        assert return_data_weight
        key = jax.random.PRNGKey(drop_seed)
        num_to_drop = int(data_weights.shape[0] * drop_frac)
        indices_to_drop = jax.random.choice(key, data_weights.shape[0],
                                            shape=(num_to_drop,), replace=False)
        data_weights = data_weights.at[indices_to_drop].set(0.0)
        vjp_lm_kw['data_weights'] = data_weights

    replay_kw = vjp_lm(**vjp_lm_kw)
    if get_vjp_head_kw:
        return {
            'per_sample_loss': replay_kw['psl_test'],
            'val_batcher': val_batcher,
            'val_its': val_its
        }

    ret = replay_vjp(**replay_kw)
    if not return_data_weight:
        return ret

    ret['data_weights'] = data_weights
    return ret

from .generalized_svd import basis_for

def main():
    os.environ['DEBUG'] = '1'

    seed = 0
    proj_dim = 1024

    one_head = ('test_loss', {
        'test_index': 0
    })

    out_p = '/mnt/xfs/home/engstrom/scratch/generalized_scratch_dir_svd'
    projectors = basis_for(to_vjp, out_p, proj_dim, seed, 1)
    lowrank_vjp_with_lds(to_vjp, projectors, [one_head], out_p, 2)

def main_randomized():
    os.environ['DEBUG'] = '1'

    seed = 0
    key = jax.random.PRNGKey(seed)
    proj_dim = 1024
    linear_combo = np.array(jax.random.normal(key, (proj_dim,))).astype(np.float32)

    one_projector = ('random', {
        'proj_dim': proj_dim,
        'seed': seed,
        'linear_combo': linear_combo
    })

    one_head = ('test_loss', {
        'test_index': 0
    })

    out_p = '/mnt/xfs/home/engstrom/scratch/generalized_scratch_dir'
    lowrank_vjp_with_lds(to_vjp, [one_projector], [one_head], out_p, 2)

if __name__ == '__main__':
    main()
