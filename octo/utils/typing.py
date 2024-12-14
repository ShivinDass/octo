from typing import Any, Mapping, Sequence, Union

import jax

try:
    PRNGKey = jax.random.KeyArray
except:
    from jax._src.prng import PRNGKeyArray as _PRNGKeyArray
    PRNGKey = _PRNGKeyArray

PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
