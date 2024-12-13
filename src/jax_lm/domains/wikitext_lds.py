import jax
import jax.numpy as jnp
import socket
from domains.wikitext_model import get_pretrained_model as get_pretrained_model_wikitext
from functools import partial
from metagradients.optimizers.interpolation import interp_from, interp_from_mom
from metagradients.optimizers.adam import make_adam_optimizer
from functools import cache
from domains.wikitext_ds import loaders_and_order_for_seed
from metagradients.dataloading import naive_batch_maker
import os
from domains.calculate_lds import calculate_lds

TEST_SAMPLE = None
MAXITS = int(os.environ.get('MAXITS', 0))
DROP_FRAC = 0.1
LDS_SEED = 0
BS = 256
LR = 0.0001
WD = 1e-05
PCT_START = 0.25
PCT_FINAL = 1
B1 = 0.9
B2 = 0.95
MIN_LR_RELATIVE = 1e-06
FINAL_MIN_LR_RELATIVE = 0.1
EPS = 1.0000000000000001e-11
EPS_SQRT = 1.0000000000000001e-11
SELECTIVE_WD = True
DTYPE = jax.numpy.float32
FACTORED_LR_WD = True
ANNEAL_TYPE = 'linear'
EPS_SCHEDULE = jax.tree_util.Partial(interp_from, steps=200, eps0=1e-08,
                                        eps_root0=1e-08, space='geometric')
MOM_SCHEDULE = jax.tree_util.Partial(interp_from_mom, steps=25, mom0=0.85,
                                        mom1=1, space='linear')
PER_PARAM_LR = None
REUSE_OPTIMIZER = True
MODEL_SEED = 0
DATA_SEED = 0

@cache
def model_maker(seed):
    model, params = get_pretrained_model_wikitext('gpt2')
    def apply_model(params, x):
        return model.apply(params, x, deterministic=True)

    return jax.tree_util.Partial(apply_model), params

# missing only: initial_params=params, train_its=train_its, 
WIKITEXT_OPT_KW = dict(
    lr=LR, wd=WD, pct_start=PCT_START, pct_final=PCT_FINAL, b1=B1, b2=B2,
    min_lr_relative=MIN_LR_RELATIVE, eps=EPS, eps_sqrt=EPS_SQRT,
    final_min_lr_relative=FINAL_MIN_LR_RELATIVE,
    selective_wd=SELECTIVE_WD, dtype=DTYPE,
    factored_lr_wd=FACTORED_LR_WD, anneal_type=ANNEAL_TYPE,
    eps_schedule=EPS_SCHEDULE, mom_schedule=MOM_SCHEDULE,
    per_param_lr=PER_PARAM_LR, reuse_optimizer=REUSE_OPTIMIZER
)

def make_loaders_and_data_weights(data_seed):
    wikitext_loaders_for_seed = partial(loaders_and_order_for_seed, bs=BS,
                                        epochs=4)
    loaders = wikitext_loaders_for_seed(data_seed)
    (train_ba_fn, train_its), (val_ba_fn, val_its) = loaders
    train_its = min(MAXITS or train_its, train_its)
    assert train_its > 0

    # make dataweights
    num_datapoints = 0
    for it in range(train_its):
        ixs = train_ba_fn(it)[0]
        num_datapoints = int(max(num_datapoints, ixs.max() + 1))

    data_weights = jax.numpy.ones((num_datapoints,), dtype=jnp.float32)
 
    hostname = socket.gethostname()
    if any(f'deep-chungus-{k}.csail.mit.edu' in hostname for k in range(1, 6)):
        minibs = 8
    else:
        minibs = 16

    print(">> Hostname:", hostname, "Minibs:", minibs)

    train_batcher = partial(naive_batch_maker, get_batch=train_ba_fn,
                            minibs=minibs, num_batches=train_its)
    val_batcher = partial(naive_batch_maker, get_batch=val_ba_fn,
                          minibs=minibs, num_batches=val_its)
    return train_batcher, val_batcher, data_weights, train_its, val_its

def make_wikitext_optimizer(params, train_its):
    return make_adam_optimizer(initial_params=params, train_its=train_its,
                               **WIKITEXT_OPT_KW)

def main():
    ret = calculate_lds(MODEL_SEED, DATA_SEED, TEST_SAMPLE, BS, DROP_FRAC,
                        LDS_SEED, make_loaders_and_data_weights, model_maker,
                        make_wikitext_optimizer)
    print(ret)

if __name__ == '__main__':
    main()

