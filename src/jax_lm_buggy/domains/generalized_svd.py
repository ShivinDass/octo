import jax
import jax.numpy as jnp
from pathlib import Path
import numpy as np
from slapreduce import slap
import dill
from metagradients.utils import make_shardings
from metagradients.vjp import async_iterator
from tqdm import tqdm
from .vjp_blocks import one_sample_vjp
from .generalized_project import train_model_at_location, DEFAULT_GRES, DEFAULT_PARTITION
from .projector_heads import rademacher_project_params

def val_projected_grads(to_vjp, final_state_path, proj_dim, seed, mat_path,
                        keep_only_index=None):
    dummy_head = lambda *args, **kwargs: None
    z = to_vjp(dummy_head, return_state=True, forward_only=True,
                         get_vjp_head_kw=True)
    psl_test, val_batcher, n_val_ba = [z[k] for k in ['per_sample_loss', 'val_batcher', 'val_its']]
    with open(final_state_path, 'rb') as f:
        final_state = dill.load(f)

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
                if keep_only_index is not None and this_idx[0] != keep_only_index:
                    continue

                this_x = x[sel]
                this_y = y[sel]
                sample = this_idx, (this_x, this_y)
                grad, primal = one_sample_vjp(sample, final_state, psl_test)
                proj_grad = rademacher_project_params(grad.params, seed, proj_dim)
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

    final_mat = jnp.concatenate(batch_projections, axis=0)
    final_mat = np.array(final_mat, dtype=np.float32)
    if keep_only_index is not None:
        assert final_mat.shape[0] == 1, final_mat.shape

    with open(mat_path, 'wb') as f:
        dill.dump(final_mat, f)

def basis_for(to_vjp, scratch_dir, proj_dim, seed, svd_dim, gres=DEFAULT_GRES,
              partition=DEFAULT_PARTITION, keep_only_index=None):
    toy_head = ('test_loss', {'test_index': 0})
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(exist_ok=True, parents=True)
    state_path = scratch_dir / 'state.dill'
    train_model_at_location(scratch_dir, to_vjp, toy_head, gres)
    if keep_only_index is None:
        mat_path = scratch_dir / 'proj_grads/proj_grads.pkl'
    else:
        mat_path = scratch_dir / f'proj_grads/proj_grads_keep{keep_only_index}.pkl'

    mat_path.parent.mkdir(exist_ok=True, parents=True)

    if not mat_path.exists():
        print('mat path does not exist', mat_path)
        kw = {
            'to_vjp': to_vjp,
            'final_state_path': state_path,
            'proj_dim': proj_dim,
            'seed': seed,
            'mat_path': mat_path,
            'keep_only_index': keep_only_index
        }

        this_dired = scratch_dir / 'calc_val_grads'
        slap(val_projected_grads, [kw], this_dired, gres=gres,
             partition=partition, block=True, job_name='calc_val_grads')

    with open(mat_path, 'rb') as f:
        proj_grads = dill.load(f)

    # save the basis
    basis_path = scratch_dir / 'basis.pkl'
    if not basis_path.exists():
        # proj_grads: (num_samples, proj_dim)
        if proj_grads.shape[0] > 1:
            u, s, vh = np.linalg.svd(proj_grads.T, full_matrices=True)
            assert u.shape == (proj_dim, proj_dim)

            u = u[:, :svd_dim].astype(np.float32)
            explained_variance = np.sum(s[:svd_dim]) / np.sum(s)
            print('>> explained_variance', explained_variance)

            with open(basis_path, 'wb') as f:
                dill.dump(u, f)
        else:
            u = proj_grads.T
            with open(basis_path, 'wb') as f:
                dill.dump(u, f)

    return [
        ('svd', {
            'basis_num': i,
            'basis_path': basis_path,
            'seed': seed,
            'proj_dim': proj_dim,
            'svd_dim': svd_dim}
        ) for i in range(svd_dim)
    ]




