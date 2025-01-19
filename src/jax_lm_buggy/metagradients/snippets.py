######
per_sample_losses = get_all_val_losses(final_state, val_batcher,
                                       val_its, psl_test=psl_test)
def get_all_val_losses(state, val_batcher, val_its, *, psl_test):
    def minibatch_to_batchloss(minibatch, state, total_bs):
        losses = psl_test(state.params, minibatch, divisor=1.0)
        bsi = minibatch[2]
        full_losses = jnp.zeros((total_bs,), dtype=losses.dtype)
        full_losses = full_losses.at[bsi].set(losses)
        return full_losses

    per_batch_losser = partial(minibatch_to_batchloss, state=state)
    losses = []

    assert val_its > 0
    batches_iterator = async_iterator(val_batcher, 0, val_its, 'val')
    for _ in tqdm(range(val_its), desc='Calculating losses..'):
        batch, minibatches = next(batches_iterator)
        this_pbl = partial(per_batch_losser, total_bs=batch.bs)
        batch_losses = minibatch_func(this_pbl, minibatches)
        losses.append(batch_losses)

    losses = jnp.concatenate(losses)
    return losses


#######
from flax import struct

class MiniBatch(struct.PyTreeNode):
    indices: object = struct.field(pytree_node=True)
    x: object = struct.field(pytree_node=True)
    y: object = struct.field(pytree_node=True)

    def __len__(self):
        return len(self.indices)

########

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)

    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)

def tree_random_normal_like(rng_key, target, std):
    keys_tree = random_split_like_tree(rng_key, target)
    def mapper(l, k):
        if l.dtype == jax.float0 or l is None or l.dtype == jax.numpy.int32:
            return l

        return jax.random.normal(k, l.shape, l.dtype) * std

    return jax.tree_util.tree_map(mapper, target, keys_tree)

### PACKED SELECT
# TODO: do something more complicated here
def filter_data(get_train_batch, n_train_ba, data_weights, take_data_subset, epochs=1, minibs=None):
    if take_data_subset == 'all_nz':
        ixs, xs, ys = [], [], []
        for i in range(n_train_ba):
            ix, (x, y) = get_train_batch(i)
            mask = data_weights[ix] > 0.
            ixs.append(ix[mask])
            if not isinstance(x, HetSeqBatch):
                xs.append(x[mask])
                ys.append(y[mask])
            else:
                xs.extend(x.subselect(mask).xs)
                ys.extend(y.subselect(mask).xs)

    elif take_data_subset == 'pack':
        assert minibs is not None
        from .pack_data import pack_data
        return pack_data(get_train_batch, n_train_ba, data_weights, minibs)
    else:
        raise NotImplementedError(f'take_data_subset {take_data_subset} not implemented')

    is_het_seq = isinstance(x, HetSeqBatch)
    batch_size = len(x)

    ixs = jnp.concatenate(ixs, axis=0)
    if not is_het_seq:
        xs = jnp.concatenate(xs, axis=0)
        ys = jnp.concatenate(ys, axis=0)

    rng = np.random.default_rng(2398)
    all_order = []
    for _ in range(epochs):
        order = rng.permutation(ixs.shape[0])
        all_order.append(order)

    order = np.concatenate(all_order, axis=0)
    final_ixs = ixs[order]

    if not is_het_seq:
        final_xs = xs[order]
        final_ys = ys[order]
    else:
        final_xs = [xs[int(i)] for i in order]
        final_ys = [ys[int(i)] for i in order]

    new_n = final_ixs.shape[0]
    assert new_n % batch_size == 0

    def new_get_train_batch(i):
        assert i < new_n
        sel = slice(i * batch_size, (i+1) * batch_size)
        this_ixs = final_ixs[sel]

        if not is_het_seq:
            this_xs = final_xs[sel]
            this_ys = final_ys[sel]
        else:
            this_xs = HetSeqBatch(final_xs[sel], x.bucket_size, x.filler)
            this_ys = HetSeqBatch(final_ys[sel], y.bucket_size, y.filler)
            lengths = [len(x) for x in this_xs.xs]
            sorted_indices = np.argsort(lengths)
            this_ixs = this_ixs[sorted_indices]
            this_xs = this_xs.subselect(sorted_indices)
            this_ys = this_ys.subselect(sorted_indices)

        assert len(this_ixs) == batch_size
        return this_ixs, (this_xs, this_ys)

    return new_get_train_batch, int(new_n//batch_size)
### END PACKED SELECT


    make dataweights + psl
    if data_weights is None:
        data_weights = jnp.ones(num_datapoints, dtype=jnp.float32)

    if drop_cfx:
        data_weights = data_weights.at[drop_cfx].set(0)

#######

def dynamic_array_set(x, n, indices, values):
    assert isinstance(x, np.ndarray), type(x)
    assert isinstance(indices, np.ndarray), type(indices)
    assert isinstance(values, np.ndarray), type(values)
    assert isinstance(n, int), type(n)

    required_length = int(indices.max() + 1)
    min_grow_size = x.shape[0] * 2
    grow_length = max(required_length, min_grow_size)
    if required_length > x.shape[0]:
        print(">> Growing array from", len(x), "to", grow_length, 'given', required_length)
        new_x = np.zeros((grow_length,), dtype=x.dtype)
        new_x[:x.shape[0]] = x
        n = required_length
    else:
        new_x = x

    new_x[indices] += values
    return new_x, n

import jax
import jax.numpy as jnp
from functools import partial
import timeit

def project_single_tensor(tensor, key, block_size, out_dim):
    return _project_single_tensor(tensor, key, block_size, out_dim)

# @partial(jax.remat, static_argnums=(1, 2, 3))
def _project_single_tensor(tensor, key, block_size, out_dim):
    assert tensor.dtype == jnp.float32
    tensor = tensor.reshape(-1)
    num_elems = tensor.shape[0]
    num_iterates = (num_elems + block_size - 1) // block_size
    # keys = jax.random.split(key, num_iterates)
    # keys_list = list(keys)
    # slices = [tensor[i * block_size:(i + 1) * block_size] for i in range(num_iterates)]

    def body(i, val):
        key, acc, tensor = val
        key = jax.random.split(key)[1]
        s = i * block_size
        x = jax.lax.dynamic_slice(tensor, (s,), block_size)
        proj_shape = (out_dim, x.shape[0])
        random_rademacher = jax.random.rademacher(key, dtype=jnp.float32, shape=proj_shape)
        out = random_rademacher @ x
        return (key, acc + out, tensor)

    acc = jnp.zeros(out_dim, dtype=jnp.float32)
    _, acc, _ = jax.lax.fori_loop(0, num_iterates, body, (key, acc, tensor))
    return acc

    # @partial(jax.remat, static_argnums=(1,))
    # def project_one_slice(x, key):
    #     assert len(x.shape) == 1
    #     projection_shape = (out_dim, x.shape[0])
    #     random_rademacher = jax.random.rademacher(key,
    #                                               dtype=jnp.float32,
    #                                               shape=projection_shape)
    #     out = random_rademacher @ x
    #     assert out.shape == (out_dim,)
    #     return out

    # projected_slices = jax.tree.map(project_one_slice, slices, keys_list)
    # summed = jax.tree.reduce(jnp.add, projected_slices)
    return summed

curr_treedef = None
def check_treedef(x):
    global curr_treedef
    if curr_treedef is None:
        curr_treedef = x
    else:
        assert curr_treedef == x, (curr_treedef, x)

@partial(jax.jit, static_argnames=('block_size', 'out_dim'))
def project_pytree(pytree, seed, out_dim=1024, block_size=512):
    key = jax.random.PRNGKey(seed)
    flat, treedef = jax.tree_util.tree_flatten(pytree)
    check_treedef(treedef)
    keys = jax.random.split(key, len(flat))
    keys_list = list(keys)
    this_project_tensor = partial(project_single_tensor,
                                  block_size=block_size, out_dim=out_dim)
    projections = jax.tree.map(this_project_tensor, flat, keys_list)
    return jax.tree.reduce(jnp.add, projections)

def main():
    for block_size in [256, 512, 1024, 2048, 4096]:
        example_pytree = {
            'a': jnp.ones((20, 8192 * 8), dtype=jnp.float32),
            'b': jnp.ones((20, 8192 * 8), dtype=jnp.float32),
            'c': jnp.ones((20, 8192 * 8), dtype=jnp.float32),
        }

        project_pytree(example_pytree, 0, block_size=block_size)
        project_pytree(example_pytree, 0, block_size=block_size)
        print(block_size, timeit.timeit(lambda: jax.block_until_ready(project_pytree(example_pytree, 0, block_size=block_size)), number=10))

if __name__ == '__main__':
    main()

