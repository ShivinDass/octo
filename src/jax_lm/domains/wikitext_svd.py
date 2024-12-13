from .wikitext_projected import seed_to_vjp_lm_kw, random_projector, wikitext_kw, \
    Path, vjp_lm, dill, replay_vjp, make_shardings
from functools import partial
import os
from slapreduce import slap
import jax, numpy as np
from tqdm import tqdm
from metagradients.vjp import async_iterator
from .vjp_blocks import one_sample_vjp
import jax.numpy as jnp
from .pallas_projector import rademacher_project_state, pallas_rademacher_vjp_head
from .wikitext_projected import grad_from_store

def val_projected_grads(val_batcher, final_state, psl_test, n_val_ba, proj_dim,
                        projector, seed):
    sharding, replicated_sharding = make_shardings()
    val_batcher = jax.tree_util.Partial(val_batcher, sharding=sharding)
    batches = async_iterator(val_batcher, 0, n_val_ba, 'meta')
    n_tot = 0
    batch_projections = []

    for _ in tqdm(range(n_val_ba)):
        _, minibatches = next(batches)
        bproj = {}
        iterator = tqdm(minibatches, total=minibatches.bs)
        for idx, (x, y) in iterator:
            for i in range(len(idx)):
                sel = slice(i, i+1)
                this_idx = idx[sel]
                this_x = x[sel]
                this_y = y[sel]
                sample = this_idx, (this_x, this_y)
                grad, primal = one_sample_vjp(sample, final_state, psl_test)
                proj_grad = projector(grad, seed, proj_dim)
                assert proj_grad.shape == (proj_dim,)
                key = int(this_idx[0])
                bproj[key] = proj_grad
                iterator.update(1)

        keys_sorted = list(sorted(list(bproj.keys())))
        print('keys sorted', keys_sorted)
        arrays_sorted = [bproj[key] for key in keys_sorted]
        # assert batch_projections[-1] 
        stacked = (jnp.stack(arrays_sorted))
        assert keys_sorted[0] == n_tot, (keys_sorted[0], n_tot)
        assert keys_sorted[-1] == n_tot + stacked.shape[0] - 1
        n_tot += stacked.shape[0]
        batch_projections.append(stacked)
        # collect all projected gradients

    final_mat = jnp.concatenate(batch_projections, axis=0)
    final_mat = np.array(final_mat, dtype=np.float32)
    return final_mat

def full_calc_svd_lds(seed, test_index, save_dir, proj_dim, svd_dim):
    # first calculate the projected gradient for each seed
    vjp_lm_kw = seed_to_vjp_lm_kw(0, random_projector, **wikitext_kw)
    vjp_lm_kw.update(dict(
        return_state=True,
        forward_only=True
    ))

    vjp_kw = vjp_lm(**(vjp_lm_kw | {'return_kw': True}))
    final_state_path = Path(save_dir) / 'final_state.pkl'
    if not final_state_path.exists():
        final_state = replay_vjp(**vjp_kw)['final_state']
        with open(final_state_path, 'wb') as f:
            dill.dump(final_state, f)
    else:
        with open(final_state_path, 'rb') as f:
            final_state = dill.load(f)

    sharding, replicated_sharding = make_shardings()
    final_state = jax.device_put(final_state, replicated_sharding)
    proj_grad_path = Path(save_dir) / f'proj_grads_{proj_dim}.pkl'
    print('>> proj_grad_path', proj_grad_path)
    if not proj_grad_path.exists():
        proj_grads = val_projected_grads(vjp_lm_kw['val_batcher'], final_state,
                                         vjp_kw['psl_test'], vjp_lm_kw['n_val_ba'],
                                         proj_dim, rademacher_project_state, seed)
        with open(proj_grad_path, 'wb') as f:
            dill.dump(proj_grads, f)
    else:
        with open(proj_grad_path, 'rb') as f:
            proj_grads = dill.load(f)

    # now compute the singular values
    # proj_grads: (256, 8192)
    u, s, vh = np.linalg.svd(proj_grads.T, full_matrices=True)

    # take only the vectors corresponding to the top svd_dim singular values
    u = u[:, :svd_dim]
    ev = (s[:svd_dim]**2).sum() / (s**2).sum()
    print('Explained variance in top', svd_dim, 'singular values:', ev)

    # scale singular vectors by singular values
    # u = u * s[:svd_dim]

    # launch job to calculate 
    should_collect = os.environ.get("COLLECT") == '1'
    vjp_head_base = partial(pallas_rademacher_vjp_head, seed=seed, proj_dim=proj_dim)
    name = f'svd_{proj_dim}'
    svd_savedir = save_dir / name
    u = np.array(u).astype(np.float32)

    if not should_collect:
        def metagrad_for_singular_vector(idx):
            v = u[:, idx]
            this_vjp_head = jax.tree_util.Partial(vjp_head_base, proj_weighting=v)
            this_vjp_lm_kw = seed_to_vjp_lm_kw(0, random_projector, **wikitext_kw)
            this_vjp_lm_kw.update({
                'vjp_head': this_vjp_head,
            })

            ret = vjp_lm(**this_vjp_lm_kw)
            grad = grad_from_store(ret['deps'], ret['batch_indices'])
            return grad

        gres = {
            'exclude': 'deep-gpu-[10,11],deep-chungus-[1-5]',
            'gres': 'gpu:1',
            'cpus_per_task':1,
            'nodes': 1,
        }

        partition = 'h100'
        xs = [{'idx': i} for i in range(svd_dim)]
        slap(metagrad_for_singular_vector, xs, svd_savedir, gres=gres,
             partition=partition, block=False, job_name=name)
    elif should_collect:
        for kw, ret in collect(svd_savedir):
            idx = kw['idx']
            mg = ret['grad']

from slapreduce import collect

if __name__ == '__main__':
    out_p = Path('/mnt/xfs/home/engstrom/store/wikitext_svd/v0')
    out_p.mkdir(exist_ok=True, parents=True)
    full_calc_svd_lds(0, 0, out_p, 8192, 128)