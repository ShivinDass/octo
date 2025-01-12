import os
import dill
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from scipy.stats import spearmanr, pearsonr
import jax

from slapreduce import slap, collect
from pathlib import Path
import numpy as np
from functools import cache
from .calculate_lds import grad_from_store

DEFAULT_GRES = {
    'exclude': 'deep-gpu-[10,11],deep-chungus-[1-5]',
    'gres': 'gpu:1',
    'cpus_per_task':1,
    'nodes': 1,
}

from .projector_heads import random_projector_for_proj_dim
from .vjp_blocks import one_sample_vjp_head

def construct_head(name, metadata):
    if name == 'random':
        pre_projector = construct_pre_projector(name, metadata)
        out_head = jax.tree_util.Partial(random_project_vjp_head,
                                         projector=pre_projector)
        return out_head
    elif name == 'test_loss':
        test_index = metadata['test_index']
        out_head = jax.tree_util.Partial(one_sample_vjp_head, test_index=test_index)
        return out_head

    raise ValueError(f'Unknown head name {name}')

def construct_pre_projector(name, metadata):
    metadata = {k: v for k, v in metadata.items()}
    if name == 'random': 
        proj_dim = metadata['proj_dim']
        f = random_projector_for_proj_dim(proj_dim)
        other_kw = {k: v for k, v in metadata.items() if k != 'proj_dim'}
        return jax.tree_util.Partial(f, **other_kw)
    elif name == 'svd':
        proj_dim = metadata['proj_dim']
        svd_dim = metadata['svd_dim']
        basis_path = metadata['basis_path']
        basis_num = metadata['basis_num']
        seed = metadata['seed']

        with open(basis_path, 'rb') as f:
            basis = dill.load(f)
            assert basis.shape == (proj_dim, svd_dim)
            this_basis_vector = np.array(basis[:, basis_num]).astype(np.float32)

        f = random_projector_for_proj_dim(proj_dim)
        return jax.tree_util.Partial(f, seed=seed, linear_combo=this_basis_vector)


    raise ValueError(f'Unknown projector name {name}')

DEFAULT_PARTITION = 'h100'

def _to_vjp(vjp_head, return_state, forward_only, drop_frac=None,
            drop_seed=None, return_data_weight=False, get_vjp_head_kw=False):
    raise NotImplementedError

def left_project_one_d(pre_projector, to_vjp):
    # returns: meta-gradient sized JzPT 
    projector = make_projector(pre_projector)
    ret = to_vjp(projector, return_state=False, forward_only=False,
                 drop_frac=None, drop_seed=None, return_data_weight=False)
    grad = grad_from_store(ret['deps'], ret['batch_indices'])
    return grad

@jax.jit
def calculate_random_project(state, projector):
    print('JITTING RANDOM PROJECT', projector)
    assert 'params' in dir(state) or (isinstance(state, dict) and 'params' in state), f'Expected state to have params, got {state}'
    def project_it(state):
        projected = projector(state['params'] if isinstance(state, dict) else state.params)
        return projected

    primal, vjp_fn = jax.vjp(project_it, state)
    assert primal.size == 1, f'Expected 1D primal, got {primal.shape}'
    g, = vjp_fn(jnp.ones_like(primal))
    return g, primal

def random_project_vjp_head(state, *, per_sample_loss=None, val_batcher=None,
                            val_its=None, projector):
    return calculate_random_project(state, projector)

def right_project(collect_dir, out_heads, state_path, to_vjp):
    head = construct_head(*out_heads[0])
    vjp_head_kw = to_vjp(head, return_state=True, forward_only=True, get_vjp_head_kw=True)
    with open(state_path, 'rb') as f:
        state = dill.load(f)

    projector_mg_tuples = []
    for kw, Jz_i in collect(collect_dir):
        pp = kw['pre_projector']
        this_projector = make_projector(pp)
        projector_mg_tuples.append((this_projector, Jz_i))

    print(f'Retrieved {len(projector_mg_tuples)} projection dimensions')

    # now for each projector, get the vjp
    total_grads = []
    primals = []
    for pre_out_head in out_heads:
        out_head = make_head(pre_out_head)
        g, primal = out_head(state, **vjp_head_kw)
        total_grad = 0.
        for projector, Jz_i in projector_mg_tuples:
            _, Pg = projector(g)
            total_grad += Pg * Jz_i

        total_grads.append(total_grad)
        primals.append(primal)

    return total_grads, primals

def make_projector(pre_projector):
    pre_projector = construct_pre_projector(*pre_projector)
    projector = jax.tree_util.Partial(random_project_vjp_head,
                                      projector=pre_projector)
    return projector

def make_head(pre_head):
    head = construct_head(*pre_head)
    return head

def train_one_state(to_vjp, pre_head, state_path):
    head = make_head(pre_head)
    ret = to_vjp(head, return_state=True, forward_only=True)

    with open(state_path, 'wb') as f:
        dill.dump(ret['final_state'], f)

def train_model_at_location(scratch_dir, to_vjp, pre_head, gres):
    state_path = scratch_dir / 'state.dill'

    print('@@ TRAINING MODEL')
    state_kw = {
        'to_vjp': to_vjp,
        'pre_head': pre_head,
        'state_path': state_path
    }

    slap(train_one_state, [state_kw], scratch_dir / 'train_one_model',
        gres=gres, partition='high-priority', block=True, job_name='train_model')

    return state_path

def lowrank_vjp_with_lds(to_vjp, pre_projectors, out_heads, scratch_dir,
                         num_trials, drop_frac=0.1, gres=DEFAULT_GRES,
                         partition=DEFAULT_PARTITION):
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(exist_ok=True, parents=True)
    # pre_projectors: list of differentiable (linear) projectors according to "state.params -> scalar"
    # first get JzPT
    map_xs_per_proj_dimension = [
        {'pre_projector': P_i, 'to_vjp': to_vjp} for P_i in pre_projectors
    ]

    project_dir = scratch_dir / 'projections'

    slap(left_project_one_d, map_xs_per_proj_dimension, project_dir,
         gres=gres, partition=partition, block=True, job_name='left_project')

    state_path = scratch_dir / 'state.dill'
    if not state_path.exists():
        train_model_at_location(scratch_dir, to_vjp, out_heads[0], gres)

    # now get the vjps
    # total_grads, primals = right_project(project_dir, out_heads, state_path)
    # TODO: ensure this is in the right order?
    right_project_kw = {
        'to_vjp': to_vjp,
        'collect_dir': project_dir,
        'out_heads': out_heads,
        'state_path': state_path
    }

    right_project_path = scratch_dir / 'right_project'
    slap(right_project, [right_project_kw], right_project_path,
         gres=gres, partition='high-priority', block=True, job_name='right_project')

    # train models for LDS
    lds_model_seeds = [{
        'drop_seed': i,
        'to_vjp': to_vjp,
        'drop_frac': drop_frac,
        'pre_projector': pre_projectors[0]
    } for i in range(num_trials)]

    lds_model_path = scratch_dir / 'lds_models'
    slap(calculate_lds_state, lds_model_seeds, lds_model_path, gres=gres,
         partition='high-priority', block=True, job_name='calculate_lds models')

    calculate_lds_kw = {
        'lds_model_path': lds_model_path,
        'right_project_path': right_project_path,
        'out_heads': out_heads,
        'to_vjp': to_vjp
    }

    calculate_lds_path = scratch_dir / 'calculate_lds'
    slap(calculate_lds, [calculate_lds_kw], calculate_lds_path,
         gres=gres, partition='high-priority', block=True, job_name='calculate_lds')

def calculate_lds_state(pre_projector, to_vjp, drop_frac, drop_seed):
    projector_head = make_projector(pre_projector)
    ret = to_vjp(projector_head, return_state=True, forward_only=True,
                   drop_frac=drop_frac, drop_seed=drop_seed,
                   return_data_weight=True)
    state, data_weights = ret['final_state'], ret['data_weights'] 
    return {
        'final_state': state,
        'data_weights': data_weights
    }

def replace_state(base_state, new_state):
    return base_state.replace(opt_state=new_state.opt_state,
                              params=new_state.params)

def wipe_state(base_state):
    return base_state.replace(opt_state=None, params=None)

import jax.numpy as jnp

def calculate_lds(lds_model_path, right_project_path, out_heads, to_vjp):
    # ys: length num states list of out_head outputs (# num states x # out_heads)
    head = construct_head(*out_heads[0])
    vjp_head_kw = to_vjp(head, return_state=True, forward_only=True,
                         get_vjp_head_kw=True)

    ys = []
    yhats = []

    for kw, ret in collect(right_project_path):
        projected_metagradients, primals = ret

    out_heads = [make_head(pre_out_head) for pre_out_head in out_heads]

    state_cache = None
    for kw, ret in collect(lds_model_path):
        state, data_weights = ret['final_state'], ret['data_weights']
        if state_cache is None:
            state_cache = wipe_state(state)

        state = replace_state(state_cache, state)
        primals_for_state = []
        yhats_for_state = []
        for out_head, projected_mg in zip(out_heads, projected_metagradients):
            _, primal = out_head(state, **vjp_head_kw)
            delta_x = data_weights - jnp.ones_like(data_weights)
            delta_primal = float(projected_mg @ delta_x)
            primals_for_state.append(float(primal))
            yhats_for_state.append(delta_primal)

        ys.append(np.array(primals_for_state))
        yhats.append(np.array(yhats_for_state))

    yhats = np.stack(yhats)
    ys = np.stack(ys)
    assert yhats.shape == ys.shape
    assert yhats.shape[1] == len(out_heads)

    ys = ys.T
    yhats = yhats.T

    def stats_for_correlator(correlator):
        all_corrs = []
        for i in range(len(out_heads)):
            corr = correlator(ys[i], yhats[i]).statistic
            all_corrs.append(corr)

        print('Correlations:', all_corrs)
        mean_corr, std_corr = np.mean(all_corrs), np.std(all_corrs)
        print('Mean correlation:', mean_corr, 'Std correlation:', std_corr)
        return all_corrs

    print('>> SPEARMANR')
    stats_for_correlator(spearmanr)
    print('\n\n')
    print('>> PEARSONR')
    stats_for_correlator(pearsonr)
