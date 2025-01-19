from functools import partial, cache
import jax
from .pallas_projector import rademacher_project_params

@partial(jax.jit, static_argnames=('proj_dim'))
def random_projector(params, seed, proj_dim, linear_combo):
    projected = rademacher_project_params(params, seed, proj_dim)
    ip = (projected @ linear_combo)
    assert ip.size == 1, ip.shape
    return ip.sum()

@cache
def random_projector_for_proj_dim(proj_dim):
    return partial(random_projector, proj_dim=proj_dim)
