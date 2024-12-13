import jax
import jax.numpy as jnp
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from functools import partial
from jax.experimental.pallas import gpu as plgpu
from operator import mul
from functools import reduce
import enum
jax.config.update('jax_default_matmul_precision', 'float32')

class Precision(enum.Enum):
    TF32_ROUND = 0
    TF32_TRUNC = 1
    FP32 = 2
    FP16 = 3
    BF16 = 4

def round_tf32(x):
    ASM = "cvt.rna.tf32.f32 $0, $1;"
    [result] = plgpu.elementwise_inline_asm(
            ASM,
            args=[x],
            constraints="=r, r",
            pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)]
    )
    return result

def tile_dot(a, b, precision):
    if precision == Precision.TF32_ROUND:
        ret = pl.dot(round_tf32(a), round_tf32(b), precision="tensorfloat32")
        return ret
    elif precision == Precision.TF32_TRUNC:
        return pl.dot(a.astype(jnp.float32), b.astype(jnp.float32), precision="tensorfloat32")
    elif precision == Precision.FP32:
        return pl.dot(a.astype(jnp.float32), b.astype(jnp.float32), precision="highest")
    elif precision == Precision.BF16:
        return pl.dot(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16))
    elif precision == Precision.FP16:
        return pl.dot(a.astype(jnp.float16), b.astype(jnp.float16))
    else:
        raise ValueError(f"Invalid precision: {precision}")

# x: uint32
def pcg_hash(x):
    state = x * jnp.array(747796405, dtype=jnp.uint32) + jnp.array(2891336453, dtype=jnp.uint32) 
    shiftone = jnp.array(28, dtype=jnp.uint32)
    shifttwo = jnp.array(4, dtype=jnp.uint32)
    mul = jnp.array(277803737, dtype=jnp.uint32)
    word = ((state >> ((state >> shiftone) + shifttwo)) ^ state) * mul
    shiftthree = jnp.array(22, dtype=jnp.uint32)
    return (word >> shiftthree) ^ word

def randint(seed, shape, maximum):
    num_slots = reduce(mul, shape)
    seeds = (seed).astype(jnp.uint32) + jnp.arange(0, num_slots, dtype=jnp.uint32)
    hashes = pcg_hash(seeds)
    return (hashes % maximum).reshape(shape)

def rademacher(seed, shape):
    ret = (randint(seed, shape, 2) * 2)
    newret = ret.astype(jnp.float32) - 1
    return newret

def seed_for(base_seed, row, col):
    return base_seed * 10000 + row * 1000000 + col * 1000000000

PROJ_BLOCKSIZE = 256
X_BLOCKSIZE = 64

def kernel(x_ref, seed_ref, y_ref, proj_blocksize, x_blocksize, input_d):
    row = pl.program_id(0)
    seed = seed_ref[...]
    acc = jnp.zeros((proj_blocksize,), jnp.float32)
    def loop_body(i, acc):
        curr_x_slice = pl.dslice(i * x_blocksize, x_blocksize)
        x_block = pl.load(x_ref, (curr_x_slice,))
        A_shape = (int(x_blocksize), int(proj_blocksize))
        this_seed = seed_for(seed, row, i)
        A_block = rademacher(this_seed, A_shape)
        # x_block: (x_blocksize,)
        # A_block: (x_blocksize, proj_blocksize)
        return acc + (x_block[:, None] * A_block).sum(axis=0)

    acc = jax.lax.fori_loop(0, input_d//x_blocksize, loop_body, acc)
    y_ref[...] = acc

from jax import core

def to_concrete(x, name):
    x = core.concrete_or_error(None, x, f"The {name} must be concrete.")
    if hasattr(x, 'to_concrete_value'):
        x = x.to_concrete_value()
    return x

@partial(jax.jit, static_argnames=["proj_d", "num_stages", "num_warps", "x_blocksize", "proj_blocksize"])
def random_Ax(x_input, seed, proj_d, num_stages, num_warps, x_blocksize=None, 
              proj_blocksize=None):
    # make seed concrete
    proj_d = to_concrete(proj_d, "proj_d")
    x_blocksize = to_concrete(x_blocksize, "x_blocksize")

    flat_input_ = x_input.reshape(-1)
    input_d = flat_input_.shape[0]

    assert input_d % x_blocksize == 0, (input_d, x_blocksize)

    grid = (proj_d // proj_blocksize,)
    x_blockspec = pl.BlockSpec((input_d,), lambda r: (0,)) 
    seed_blockspec = pl.BlockSpec(tuple(), lambda r: tuple())
    out_spec = pl.BlockSpec((proj_blocksize,), lambda r: (r,))
    out_shape = jax.ShapeDtypeStruct((proj_d,), jnp.float32)
    kernel_p = partial(kernel, proj_blocksize=int(proj_blocksize),
                       x_blocksize=int(x_blocksize), input_d=int(input_d))

    Ax = pl.pallas_call(kernel_p,
                        out_shape=out_shape,
                        grid=grid,
                        compiler_params=dict(
                            triton=dict(num_stages=num_stages,
                                        num_warps=num_warps),
                        ),
                        in_specs=[x_blockspec, seed_blockspec],
                        out_specs=out_spec,
                        )(flat_input_, seed)
    return Ax

@partial(jax.jit, static_argnames=["input_d", "num_stages", "num_warps",
                                   "proj_blocksize", "x_blocksize"])
def random_ATy(y_input, seed, input_d, num_stages=2, num_warps=2, proj_blocksize=None,
               x_blocksize=None):
    assert len(y_input.shape) == 1, y_input.shape
    proj_d = y_input.shape[0]
    x_blocksize = to_concrete(x_blocksize, "x_blocksize")
    proj_blocksize = to_concrete(proj_blocksize, "proj_blocksize")
    input_d = to_concrete(input_d, "input_d")

    assert proj_d % proj_blocksize == 0
    assert input_d % x_blocksize == 0, (input_d, x_blocksize)

    grid = (input_d // x_blocksize,)
    out_spec = pl.BlockSpec((x_blocksize,), lambda c: (c,)) 
    y_blockspec = pl.BlockSpec((proj_d,), lambda c: (0,))
    seed_blockspec = pl.BlockSpec(tuple(), lambda c: tuple())
    out_shape = jax.ShapeDtypeStruct((input_d,), jnp.float32)

    def kernel_pT(y_ref, seed_ref, x_ref):
        seed = seed_ref[...]
        col_i = pl.program_id(0)
        acc = jnp.zeros((x_blocksize,), jnp.float32)
        def loop_body(row_i, acc):
            curr_y_slice = pl.dslice(row_i * proj_blocksize, proj_blocksize)
            y_block = pl.load(y_ref, (curr_y_slice,))
            A_shape = (int(x_blocksize), int(proj_blocksize))
            this_seed = seed_for(seed, row_i, col_i)
            A_block = rademacher(this_seed, A_shape)
            acc_x = (y_block[None, :] * A_block).sum(axis=1)
            return acc + acc_x

        acc = jax.lax.fori_loop(0, proj_d//proj_blocksize, loop_body, acc)
        x_ref[...] = acc

    ATy = pl.pallas_call(kernel_pT,
                        out_shape=out_shape,
                        grid=grid,
                        compiler_params=dict(
                            triton=dict(num_stages=num_stages,
                                        num_warps=num_warps),
                        ),
                        in_specs=[y_blockspec, seed_blockspec],
                        out_specs=out_spec,
                        )(y_input, seed)
    return ATy

_NUM_STAGES = 1
_NUM_WARPS = 4
_PROJ_BLOCKSIZE = 16
_X_BLOCKSIZE = 256

@partial(jax.custom_vjp, nondiff_argnums=[2])
def rademacher_project_tensor(x, seed, proj_d):
    return project_fwd(x, seed, proj_d=proj_d)[0]

@partial(jax.jit, static_argnames=["proj_d"], static_argnums=(2,))
def project_fwd(x, seed, proj_d):
    proj_d = to_concrete(proj_d, "proj_d")
    x = x.reshape(-1)
    Ax = random_Ax(x, seed=seed, proj_d=proj_d, num_stages=_NUM_STAGES,
                   num_warps=_NUM_WARPS, x_blocksize=_X_BLOCKSIZE,
                   proj_blocksize=_PROJ_BLOCKSIZE)
    return Ax, (x, seed)

@jax.jit
def project_bwd(proj_d, saved, dy):
    x, seed = saved
    input_d = x.shape[0]
    ATdy = random_ATy(dy, seed=seed, input_d=input_d, num_stages=_NUM_STAGES,
                      num_warps=_NUM_WARPS, proj_blocksize=_PROJ_BLOCKSIZE,
                      x_blocksize=_X_BLOCKSIZE)
    return ATdy, None

rademacher_project_tensor.defvjp(project_fwd, project_bwd)

@partial(jax.jit, static_argnames=["d"])
def rademacher_project_state(params, seed, d):
    all_params = []
    def f(p, x):
        p_str = str(p)
        if not ('opt_state' in p_str and any([x in p_str for x in ['jittable_opt_state', 'count']])):
            num_els = x.size
            assert num_els % 128 == 0, (num_els, p_str)
            all_params.append(x)
        else:
            raise ValueError(f"no opt state allowed")

    jax.tree_util.tree_map_with_path(f, params)
    params_sorted_by_length = sorted(all_params, key=lambda x: x.size)
    final_params = []
    curr_params = []
    # 5 million
    max_d = 1_000_000
    for p in params_sorted_by_length:
        curr_params.append(p.reshape(-1))
        if sum([x.size for x in curr_params]) > max_d:
            if len(curr_params) > 1:
                catted = jnp.concatenate(curr_params)
            else:
                catted = curr_params[0]

            final_params.append(catted)
            curr_params = []
        
    if len(curr_params) > 0:
        catted = jnp.concatenate(curr_params)
        final_params.append(catted)
        curr_params = []

    res = rademacher_project(tuple(final_params), seed, d)
    return res

rademacher_project_params = rademacher_project_state

def pallas_rademacher_vjp_head(state, *, per_sample_loss, val_batcher,
                               val_its, seed, proj_dim, proj_weighting):
    def project_it(state, weighting):
        projected = rademacher_project_state(state, seed, proj_dim)
        assert projected.shape == (proj_dim,), projected.shape
        assert weighting.shape == (proj_dim,), weighting.shape
        weighted = projected * weighting
        return weighted.sum()

    project_it = jax.tree_util.Partial(project_it, weighting=proj_weighting)
    primal, vjp_fn = jax.vjp(project_it, state)
    g, = vjp_fn(jnp.ones_like(primal))
    return g, primal


@partial(jax.jit, static_argnames=["proj_d"])
def rademacher_project(state, seed, proj_d):
    # cum = jnp.zeros((d,), jnp.float32)
    proj_d = to_concrete(proj_d, "proj_d")
    i = 0
    # def f(cum, x):
    #     nonlocal i
    #     y = rademacher_project_tensor(x, seed + i, d)
    #     i += 1
    #     return y + cum
    def f_map(p, x):
        nonlocal i
        y = rademacher_project_tensor(x, seed + i, proj_d)
        i += 1
        return y

    mapped = jax.tree_util.tree_map_with_path(f_map, state)
    reduced = jax.tree.reduce(jnp.add, mapped)

    return reduced

def main(proj_blocksize, x_blocksize, num_stages, num_warps):
    # x = jnp.ones((2048,), jnp.float32)
    # seed = 0
    # proj_d = 512
    # num_stages = 2
    # num_warps = 2
    # ys = []
    # for i in range(32):
    #     x = jnp.zeros_like(x)
    #     x = x.at[i].set(1)
    #     Ax = random_Ax(x, seed, proj_d, num_stages, num_warps)
    #     ys.append(Ax)

    # A = jnp.stack(ys)
    # print(A)
    input_d = 1024 * 1024 * 4
    proj_d = 2048
    seed = 0

    # A from Ax
    ys = []
    for i in range(8):
        base_x = jnp.zeros((input_d,), jnp.float32)
        base_x = base_x.at[i].set(1)
        Ax = random_Ax(base_x, seed, proj_d, num_stages, num_warps, x_blocksize=x_blocksize, proj_blocksize=proj_blocksize)
        ys.append(Ax)

    A = jnp.stack(ys)

    # A from ATy
    # ATb_i = row i of A
    xs = []
    for i in range(8):
        base_y = jnp.zeros((proj_d,), jnp.float32)
        base_y = base_y.at[i].set(1)
        ATy = random_ATy(base_y, seed, input_d, num_stages, num_warps, proj_blocksize=proj_blocksize, x_blocksize=x_blocksize)
        xs.append(ATy)

    A2 = jnp.stack(xs).T
    jax.block_until_ready((A, A2))

def test_vjp_main():
    x = jnp.ones((2048,), jnp.float32)
    seed = 0
    proj_d = 512
    y = rademacher_project(x, seed, proj_d)
    A = []
    for i in range(2048):
        x = jnp.zeros_like(x)
        x = x.at[i].set(1)
        y = rademacher_project(x, seed, proj_d)
        A.append(y)
    A = jnp.stack(A).T
    print(A)

    key = jax.random.PRNGKey(0)
    test_x = jax.random.normal(key, (2048,))

    # try:
    #     assert jnp.allclose(A @ test_x, rademacher_project(test_x, seed, proj_d))
    # except:
    a1 = A @ test_x
    a2 = rademacher_project(test_x, seed, proj_d)
    print('fwd', (a1 - a2))
    print('fwd', jnp.absolute(a1 - a2).max())

    def f(x):
        return rademacher_project(x, seed, proj_d)

    test_y = jax.random.normal(key, (512,))
    primal, vjp = jax.vjp(f, x)
    dx, = vjp(test_y)

    # assert jnp.allclose(dx, A.T @ test_y)
    print('bwd', jnp.absolute(dx - A.T @ test_y).max())

    print('-' * 80)
    print(y)
    print('-' * 80)
    print(dx)
    print('-' * 80)
    print(y.shape, dx.shape, x.shape)

if __name__ == "__main__":
    test_vjp_main()

# def profile_main():
#     import timeit
#     optimal_by_input_d = {}
#     for input_d in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608]:
#         results = []
#         num_stages, num_warps = 1, 4
#         for proj_blocksize in [4, 8, 16, 32, 64]:
#             for x_blocksize in [32, 64, 128, 256, 512]:
#                 # try:
#                 print(f"num_stages: {num_stages}, num_warps: {num_warps}, proj_blocksize: {proj_blocksize}, x_blocksize: {x_blocksize}")
#                 # do with one warmup
#                 timeit.timeit(lambda: main(proj_blocksize, x_blocksize, num_stages, num_warps), number=1)
#                 res = (timeit.timeit(lambda: main(proj_blocksize, x_blocksize, num_stages, num_warps), number=2))
#                 print(res)
#                 results.append((num_stages, num_warps, proj_blocksize, x_blocksize, res))
#         results = sorted(results, key=lambda x: x[-1])
#         best = results[0]
#         optimal_by_input_d[input_d] = best[2:5]

#     import dill
#     with open("optimal_by_input_d.pkl", "wb") as f:
#         dill.dump(optimal_by_input_d, f)

# def random_ATx(x, seed, d, block_size, num_stages, num_warps):
#     flat_input = x.reshape(-1)
#     def kernel(X_ref, P_ref):
#         pass






