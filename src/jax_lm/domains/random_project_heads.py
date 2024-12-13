import jax
from functools import partial
from jax.experimental.sparse import random_bcoo
import jax.numpy as jnp
from operator import mul
from functools import reduce

def rademacher(*args, **kwargs):
    dtype = kwargs.pop('dtype', jnp.float32)
    try:
        unif = jax.random.bernoulli(*args, **kwargs)
    except:
        import ipdb; ipdb.set_trace()
    return (2 * unif - 1).astype(dtype)

@partial(jax.jit, static_argnames=['d'])
def tree_project(seed, tree, seed2=None, d=1):
    key = jax.random.PRNGKey(seed)
    tree_dimensions = jax.tree.map(lambda x: x.size, tree)
    tree_d = jax.tree.reduce(lambda x, y: x + y, tree_dimensions, 0.)
    sparsity = 1/(tree_d**0.5)

    implicit_mode = seed2 is not None

    if implicit_mode:
        key2 = jax.random.PRNGKey(seed2)

    def random_project_and_reduce(cum, x):
        nonlocal key
        key = jax.random.split(key)[1]
        x_d = x.size
        indices_dtype = jnp.int32 #  if (x_d * d) < 2**31 else jnp.int64

        P = random_sparse_rademacher((d, x_d), key, sparsity, x.dtype,
                                     indices_dtype=indices_dtype)

        if implicit_mode:
            nonlocal key2
            key2 = jax.random.split(key2)[1]
            P2 = random_sparse_rademacher((d, x_d), key2, sparsity, x.dtype,
                                          indices_dtype=indices_dtype)
            this_product = (P * P2).sum(axis=1)
        else:
            x_flat = x.flatten()
            this_product = P @ x_flat

        assert this_product.size == d

        return cum + this_product

    ftree = []
    def filter_tree(p, x):
        p_str = str(p)
        not_opt_state = not ('opt_state' in p_str and any([x in p_str for x in ['jittable_opt_state', 'count']]))
        is_float = isinstance(x, jnp.ndarray) and x.dtype == jnp.float32

        if not_opt_state and is_float:
            # print('keeping in tree', p_str)
            # ftree.append(x)
            pass
        else:
            raise NotImplementedError

    jax.tree_util.tree_map_with_path(filter_tree, tree)

    return jax.tree.reduce(random_project_and_reduce, tree, jnp.zeros(d))

def random_sparse_rademacher(shape, key, sparsity, dtype, indices_dtype=jnp.int32):
    return random_bcoo(key, shape, dtype=dtype, indices_dtype=indices_dtype,
                       nse=sparsity, generator=rademacher)

@partial(jax.jit, static_argnames=['d'])
def tree_project_svd(seed, tree, seed2=None, d=1):
    raise NotImplementedError
    key = jax.random.PRNGKey(seed)
    tree_dimensions = jax.tree.map(lambda x: x.size, tree)
    tree_d = jax.tree.reduce(lambda x, y: x + y, tree_dimensions, 0.)
    sparsity = 1/(tree_d**0.5)

    implicit_mode = seed2 is not None

    if implicit_mode:
        key2 = jax.random.PRNGKey(seed2)

    def random_project_and_reduce(cum, x):
        nonlocal key
        key = jax.random.split(key)[1]
        x_d = x.size
        indices_dtype = jnp.int32

        P = random_sparse_rademacher_svd((d, x_d), key, sparsity, x.dtype,
                                     indices_dtype=indices_dtype)

        if implicit_mode:
            nonlocal key2
            key2 = jax.random.split(key2)[1]
            P2 = random_sparse_rademacher_svd((d, x_d), key2, sparsity, x.dtype,
                                          indices_dtype=indices_dtype)
            this_product = (P * P2).sum(axis=1)
        else:
            x_flat = x.flatten()
            this_product = P @ x_flat

        assert this_product.size == d

        return cum + this_product

    return jax.tree.reduce(random_project_and_reduce, tree, jnp.zeros(d))

def random_sparse_rademacher_svd(shape, key, sparsity, dtype, indices_dtype=jnp.int32):
    raise NotImplementedError
    num_els = int(reduce(mul, shape))
    print('>> num els', num_els)
    print('>> sparsity', sparsity)
    print('>> nz els', int(num_els * sparsity))
    print('>> shape', shape)
    return random_bcoo(key, shape, dtype=dtype, indices_dtype=jnp.int64,
                       nse=sparsity, generator=rademacher,
                       unique_indices=False, sorted_indices=True)

@partial(jax.jit, static_argnames=['d'])
def svd_project(state, seed, d):
    raise NotImplementedError
    inner_product = tree_project_svd(seed, state.params, d=d)
    scaling = (tree_project_svd(seed, state.params, d=d, seed2=seed))**0.5
    assert scaling.size == d, (scaling.shape, d)
    assert len(scaling.shape) == 1, scaling.shape
    assert inner_product.size == d, (inner_product.shape, d)
    assert len(inner_product.shape) == 1, inner_product.shape
    return inner_product / scaling

@jax.jit
def random_projector(state, seed):
    inner_product = tree_project(seed, state.params)
    scaling = (tree_project(seed, state.params, d=1, seed2=seed))**0.5
    return (inner_product / scaling).sum()

def random_project_vjp_head(state, projector, *, per_sample_loss, val_batcher,
                            val_its):
    primal, vjp_fn = jax.vjp(projector, state)
    g, = vjp_fn(jnp.ones_like(primal))
    return g, primal




