from metagradients.optimizers.adam import make_adam_optimizer
from metagradients.utils import set_dtype
import inspect
import jax.numpy as jnp
import jax
from functools import partial
from operator import getitem
from metagradients.utils import make_shardings
from metagradients.vjp import replay_vjp

def clean_jaxdict(d):
    to_write = {k: str(v) for k,v in d.items()}
    return to_write

@partial(jnp.vectorize, signature='(n),()->()')
def cross_entropy_loss(logits, label):
    def actual_loss(logits, label):
        logits = jax.nn.log_softmax(logits)
        loss = -getitem(logits, label)
        return loss

    def no_loss(logits, label):
        return jnp.zeros((), dtype=jnp.float32)

    return jax.lax.cond(label == -100, no_loss, actual_loss, logits, label)

@partial(jax.jit, static_argnames=['divisor'])
def lm_per_sample_loss(params, batch, model, data_weights, divisor=None):
    assert divisor == 1.0, divisor
    print('>> Compiling psl', batch[1][0].shape)
    idx, (x, y) = batch[:2]
    logits = model(params, x)
    losses = cross_entropy_loss(logits, y)
    nnz = jnp.sum(y != -100, axis=1)
    per_sample_mean = jnp.sum(losses, axis=1) / nnz
    if not (data_weights is None):
        these_data_weights = data_weights[idx]
        per_sample_mean = per_sample_mean * these_data_weights

    return per_sample_mean / divisor

def vjp_lm(*, state, vjp_head, vjp_skele, data_weights, return_kw,
           train_batcher, val_batcher, model, n_train_ba, n_val_ba,
           aux_datasets, forward_only, return_state=False):
    set_dtype('tf32', True)
    # curr_kw = inspect.currentframe().f_locals
    # curr_kw = clean_jaxdict(curr_kw)
    # del curr_kw['state']
    # print('>> Running okazaki_lm with:', curr_kw)

    sharding, replicated_sharding = make_shardings()

    data_weights = jax.device_put(data_weights, replicated_sharding)

    psl = jax.tree_util.Partial(lm_per_sample_loss, model=model)

    psl_test = jax.tree_util.Partial(psl, data_weights=None)
    psl_train = jax.tree_util.Partial(psl, data_weights=data_weights)

    sharding, replicated_sharding = make_shardings()
    val_batcher_head = jax.tree_util.Partial(val_batcher, sharding=sharding)

    vjp_head = jax.tree_util.Partial(vjp_head, per_sample_loss=psl_test,
                                     val_batcher=val_batcher_head,
                                     val_its=n_val_ba)
    vjp_skele = jax.tree_util.Partial(vjp_skele)

    kw = dict(state=state,
              train_batcher=train_batcher,
              val_batcher=val_batcher,
              train_its=n_train_ba,
              val_its=n_val_ba,
              psl_train=psl_train,
              psl_test=psl_test,
              vjp_skele=vjp_skele,
              vjp_head=vjp_head,
              segment_size=20,
              aux_datasets=aux_datasets,
              forward_only=forward_only,
              return_state=return_state)

    if return_kw:
        return kw

    return replay_vjp(**kw)
