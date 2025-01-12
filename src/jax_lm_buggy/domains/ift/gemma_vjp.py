from types import SimpleNamespace
import jax
import numpy as np
import jax.numpy as jnp
import socket
from functools import partial
from metagradients.optimizers.interpolation import interp_from, interp_from_mom
from metagradients.optimizers.adam import make_adam_optimizer
from domains.vjp_blocks import sample_loss_vjp_head, example_loss_vjp_skeleton, \
    dstate_vjp_skeleton
from functools import cache
from metagradients.dataloading import naive_batch_maker
import os
from scipy.stats import spearmanr, pearsonr

from ..vjp_lm import vjp_lm
from ..calculate_lds import grad_from_store
from .instruction_ds_new import make_less_dataloaders, make_gemma_lora_model, \
    IFTREPLAYBatch

BUCKET_SIZE = 128

@cache
def make_skele_for_bs(bs, dstate_only=False):
    if not dstate_only:
        return partial(example_loss_vjp_skeleton, bs=bs)
    else:
        return partial(dstate_vjp_skeleton, bs=bs)

cfg = SimpleNamespace(
    bs=128,
    lr=0.0003,
    wd=1e-6,
    lr_warmup_its=40,
    b1=0.95,
    b2=0.975,
    min_lr_relative=1e-8,
    final_min_lr_relative=0.1,
    eps=1e-13,
    eps_sqrt=1e-13,
    selective_wd=True,
    dtype=jnp.float32,
    factored_lr_wd=True,
    anneal_type='linear',
    eps_steps=100,
    eps0=1e-10,
    epochs=1,
    mom_steps=10,
    mom0=0.5,
    minibatch_fraction=1.,
    train_frac=0.2,
    train_dataset_name='COMBO',
    val_dataset_name='MMLU',
    data_seed=16,
    model_seed=16,
    lora_d=128,
    lora_std=1.)

MAX_LENGTH = 2048
SLICE_BOUND = 513
MAX_VAL_LENGTH = 3072
OPT_WEIGHTS_LB = int(1e7)

def initialize_data_weights():
    data_weights = np.zeros(2 * OPT_WEIGHTS_LB, dtype=np.float32)
    data_weights[:OPT_WEIGHTS_LB] = 1.
    data_weights[OPT_WEIGHTS_LB:] = 0.
    return data_weights

def do_gemma_vjp(data_weights, return_state, forward_only, cfg, maxits,
                 bucket_size, include_test=False, return_kw=False,
                 return_final_kw=False):
    if maxits == -1:
        maxits = None

    # assert data_weights is not None
    assert data_weights is not None

    model_apply, params = make_gemma_lora_model(cfg.model_seed, cfg.lora_d,
                                                cfg.lora_std, cfg.lora_multiple)
    ret = make_less_dataloaders(train_dataset_name=cfg.train_dataset_name,
                                frac=cfg.train_frac,
                                val_dataset_name=cfg.val_dataset_name,
                                max_length=MAX_LENGTH, seed=cfg.data_seed,
                                bs=cfg.bs,
                                epochs=cfg.epochs,
                                minibatch_fraction=cfg.minibatch_fraction,
                                bucket_size=bucket_size,
                                max_val_length=MAX_VAL_LENGTH)

    batch_maker = partial(IFTREPLAYBatch, slice_bound=SLICE_BOUND,
                          data_weights=data_weights)

    def to_mgs_loader(loader, n_its):
        return partial(naive_batch_maker, get_batch=loader, minibs=4,
                       num_batches=n_its, batch_maker=batch_maker), n_its

    train_pair, val_pair, test_pair, valval_pair = ret

    train_loader, train_its = to_mgs_loader(*train_pair)
    val_loader, val_its = to_mgs_loader(*val_pair)
    test_loader, test_its = to_mgs_loader(*test_pair)
    valval_loader, valval_its = to_mgs_loader(*valval_pair)

    assert cfg.epochs == 1
    # n_datapoints = train_its * cfg.bs
    # if data_weights is None:
    #     data_weights = np.ones((n_datapoints,), dtype=np.float32)

    eps_schedule = jax.tree_util.Partial(interp_from, steps=cfg.eps_steps,
                                         eps0=cfg.eps0, eps_root0=cfg.eps0,
                                         space='geometric')
    mom_schedule = jax.tree_util.Partial(interp_from_mom, steps=cfg.mom_steps,
                                         mom0=cfg.mom0, mom1=1, space='linear')
    state = make_adam_optimizer(initial_params=params, train_its=train_its,
                                lr=cfg.lr, wd=cfg.wd,
                                pct_start=float(cfg.lr_warmup_its/train_its),
                                pct_final=1.0, b1=cfg.b1, b2=cfg.b2,
                                min_lr_relative=cfg.min_lr_relative,
                                final_min_lr_relative=cfg.final_min_lr_relative,
                                eps=cfg.eps, eps_sqrt=cfg.eps_sqrt,
                                selective_wd=cfg.selective_wd,
                                dtype=cfg.dtype, factored_lr_wd=cfg.factored_lr_wd,
                                anneal_type='linear', eps_schedule=eps_schedule,
                                mom_schedule=mom_schedule, per_param_lr=None,
                                reuse_optimizer=True)

    vjp_head = sample_loss_vjp_head
    vjp_skele = make_skele_for_bs(cfg.bs)
    train_its = min(maxits or train_its, train_its)
    aux_datasets = {
        'test_loss': (test_loader, test_its),
        'valval_loss': (valval_loader, valval_its)
    }

    if not include_test:
        aux_datasets.pop('test_loss')

    kw = dict(state=state, vjp_head=vjp_head, vjp_skele=vjp_skele,
              data_weights=data_weights, return_kw=False,
              train_batcher=train_loader, val_batcher=val_loader,
              model=model_apply, n_train_ba=train_its, n_val_ba=val_its,
              aux_datasets=aux_datasets, forward_only=forward_only,
              return_state=return_state, bs=cfg.bs)

    if return_final_kw:
        kw['return_kw'] = True

    if not return_kw:
        return vjp_lm(**kw)
    else:
        return kw

def main():
    DROP_FRAC = 0.1
    TRIALS = 8
    LDS_SEED = 0
    maxits = int(os.environ.get('MAXITS', -1))
    ret = do_gemma_vjp(None, False, False, cfg, maxits, BUCKET_SIZE)
    y0, deps = float(ret['primal']), ret['deps']
    batch_indices = ret['batch_indices']
    grad = grad_from_store(deps, batch_indices)
    n = grad.size
    print('>> Grad size', n)

    ys = [y0]
    y_hats = [y0]

    rng = np.random.default_rng(LDS_SEED)
    data_weights = np.ones((n,), dtype=np.float32)

    for _ in range(TRIALS):
        drop_indices = rng.choice(n, int(n * DROP_FRAC), replace=False)
        this_data_weights = np.copy(data_weights)
        this_data_weights[drop_indices] = 0.
        ret = do_gemma_vjp(this_data_weights, False, True, cfg, maxits, BUCKET_SIZE)
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

if __name__ == '__main__':
    main()
