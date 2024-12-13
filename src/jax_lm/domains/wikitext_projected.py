from functools import partial
import jax
import optax
import jax.numpy as jnp
from metagradients.vjp import replay_vjp
from .vjp_blocks import one_sample_vjp_head, sample_loss_vjp_head
from tqdm import tqdm
from slapreduce import slap
import numpy as np
import os
import dill
from pathlib import Path
from .wikitext_lds import MODEL_SEED, DATA_SEED, BS, \
    make_loaders_and_data_weights as wt_loader_maker, \
    model_maker as wt_model_maker, make_wikitext_optimizer
from .calculate_lds import make_vjp_skele, grad_from_store
from domains.vjp_lm import vjp_lm
from metagradients.utils import make_shardings
from .random_project_heads import random_project_vjp_head, random_projector
from .calculate_lds import lds_for_run

def seed_to_grad(*args, **kwargs):
    vjp_kw = seed_to_vjp_lm_kw(*args, **kwargs)
    ret = vjp_lm(**vjp_kw)
    grad = grad_from_store(ret['deps'], ret['batch_indices'])
    return grad

def seed_to_vjp_lm_kw(random_project_seed, random_projector, optimizer_maker,
                      model_maker, loader_and_data_weight_maker, bs, model_seed,
                      data_seed):
    random_projector = jax.tree_util.Partial(random_projector,
                                             seed=random_project_seed)
    vjp_head = jax.tree_util.Partial(random_project_vjp_head,
                                     projector=random_projector)
    vjp_skele = make_vjp_skele(bs)

    model, params = model_maker(model_seed)
    ret = loader_and_data_weight_maker(data_seed)
    train_batcher, val_batcher, data_weights, train_its, val_its = ret
    state0 = optimizer_maker(params, train_its)
    vjp_kw = dict(state=state0, vjp_head=vjp_head, vjp_skele=vjp_skele,
                  data_weights=data_weights, return_kw=False,
                  train_batcher=train_batcher, val_batcher=val_batcher,
                  model=model, n_train_ba=train_its, n_val_ba=val_its,
                  aux_datasets={}, forward_only=False)
    return vjp_kw

wikitext_kw = dict(
    optimizer_maker=make_wikitext_optimizer,
    model_maker=wt_model_maker,
    loader_and_data_weight_maker=wt_loader_maker,
    bs=BS,
    model_seed=MODEL_SEED,
    data_seed=DATA_SEED
)

wikitext_seed_to_grad = partial(seed_to_grad, **wikitext_kw)

def grad_for_seed_path(seed, saved_dired):
    p = Path(saved_dired) / f'grads/grad_{seed}.npy'
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def projected_state_vjps(seeds, seed_and_projector_to_vjp, projector, save_dir):
    grads = {}
    for seed in seeds:
        save_path = grad_for_seed_path(seed, save_dir)
        if save_path.exists():
            with open(save_path, 'rb') as f:
                grads[seed] = dill.load(f)
        else:
            grad = seed_and_projector_to_vjp(random_project_seed=seed,
                                             random_projector=projector)
            with open(save_path, 'wb') as f:
                dill.dump(grad, f)

            grads[seed] = grad

    return grads

# one seed:
def estimate_metagradients(vjp_heads, projected_metagrads_per_seed, state,
                           projector):
    metagradient_hats = []
    primals = []
    for vjp_head in tqdm(vjp_heads):
        mg_hat = None
        g, primal = vjp_head(state)
        orig_g_norm = optax.tree_utils.tree_l2_norm(g.params)
        all_proj_g = []
        for seed, mg in tqdm(projected_metagrads_per_seed.items()):
            proj_g = projector(g, seed)
            this_contrib = proj_g * mg
            if mg_hat is None:
                mg_hat = this_contrib
            else:
                mg_hat = jax.tree.map(jnp.add, mg_hat, this_contrib)

            all_proj_g.append(proj_g)

        mg_norm = optax.tree_utils.tree_l2_norm(all_proj_g)
        print('orig_g_norm', orig_g_norm, 'mg_norm', mg_norm)
        import ipdb; ipdb.set_trace()
        primals.append(primal)
        metagradient_hats.append(mg_hat)

    return metagradient_hats, primals


def full_calc_projected_lds(seeds, test_index, save_dir):
    # ensure all projected metagradients calculated
    one_seed = seeds[0]
    mgs_by_seed = projected_state_vjps(seeds,
                                       seed_and_projector_to_vjp=wikitext_seed_to_grad,
                                       projector=random_projector,
                                       save_dir=save_dir)


    # estimate metagradients
    # vjp_head = jax.tree_util.Partial(one_sample_vjp_head per_sample_loss)
    vjp_lm_kw = seed_to_vjp_lm_kw(0, random_projector, **wikitext_kw)
    vjp_lm_kw.update(dict(
        return_state=True,
        forward_only=True
    ))

    vjp_kw = vjp_lm(**(vjp_lm_kw | {'return_kw': True}))
    final_state_path = Path(save_dir) / 'final_state.pkl'
    print('final_state_path', final_state_path)
    if not final_state_path.exists():
        print('making final state')
        final_state = replay_vjp(**vjp_kw)['final_state']
        with open(final_state_path, 'wb') as f:
            dill.dump(final_state, f)
    else:
        print('loading final state')
        with open(final_state_path, 'rb') as f:
            final_state = dill.load(f)

    sharding, replicated_sharding = make_shardings()
    final_state = jax.device_put(final_state, replicated_sharding)

    val_batcher = vjp_lm_kw['val_batcher']
    val_batcher = jax.tree_util.Partial(val_batcher, sharding=sharding)
    vjp_head = jax.tree_util.Partial(one_sample_vjp_head,
                                     per_sample_loss=vjp_kw['psl_test'],
                                     val_batcher=val_batcher,
                                     val_its=vjp_lm_kw['n_val_ba'],
                                     test_index=test_index)
    # vjp_head = jax.tree_util.Partial(random_project_vjp_head,
    #                                  projector=partial(random_projector,
    #                                                    seed=one_seed),
    #                                  per_sample_loss=vjp_kw['psl_test'],
    #                                  val_batcher=val_batcher,
    #                                  val_its=vjp_lm_kw['n_val_ba'])
    # vjp_head = jax.tree_util.Partial(sample_loss_vjp_head,
    #                                  per_sample_loss=vjp_kw['psl_test'],
    #                                  val_batcher=val_batcher,
    #                                  val_its=vjp_lm_kw['n_val_ba'])

    mg_hats, primals = estimate_metagradients([vjp_head], mgs_by_seed, final_state,
                                              random_projector)
    mg_hat = mg_hats[0]
    primal = 0 # primals[0]
    data_weights = vjp_lm_kw['data_weights']
    vjp_lm_kw['vjp_head'] = vjp_head
    # import ipdb; ipdb.set_trace()

    return lds_for_run(primal, len(mg_hat), 0.1, 0, mg_hat, data_weights, vjp_lm_kw)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    PROJ_DIM = int(os.environ.get('PROJ_DIM', 128))
    NAME = os.environ['NAME']
    all_seeds = list(map(int, rng.integers(0, 2**16, size=PROJ_DIM)))
    seeds = [{'seeds': [i]} for i in all_seeds]
    save_dir = Path(f'/mnt/xfs/home/engstrom/store/wikitext_pgrads/')
    save_dir = save_dir / NAME

    if not os.environ.get('COLLECT', False):
        f = partial(projected_state_vjps,
                    seed_and_projector_to_vjp=wikitext_seed_to_grad,
                    projector=random_projector, save_dir=save_dir)

        gres = {
            'exclude': 'deep-gpu-[10,11],deep-chungus-[1-5]',
            'gres': 'gpu:1',
            'cpus_per_task':1,
            'nodes': 1,
        }

        partition = 'h100'

        slap(f, seeds, save_dir, gres=gres, partition=partition, block=False)
    else:
        full_calc_projected_lds(all_seeds, 0, save_dir)

