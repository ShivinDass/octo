import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from .gemma_vjp import do_gemma_vjp
from metagradients.smoothness import agreement
import numpy as np
MAXITS = int(os.environ.get('MAXITS', 0))
FORWARD_ONLY = int(os.environ.get('FORWARD_ONLY', 1))
BUCKET_SIZE = int(os.environ.get('BUCKET_SIZE', 128))

def get_data_weights(cfg):
    vjkw = do_gemma_vjp(None, True, FORWARD_ONLY, cfg, MAXITS, BUCKET_SIZE,
                        False, True)
    ones_data_weights = vjkw['data_weights']
    return ones_data_weights

def smoothness_for_cfg(cfg):
    ones_data_weights = get_data_weights(cfg)
    def params_for_weights(w):
        ret = do_gemma_vjp(w, True, FORWARD_ONLY, cfg, MAXITS, BUCKET_SIZE,
                           False, False)
        final_params = ret['final_state'].params
        primal = float(ret['primal'])
        return final_params, primal

    fs0, primal = params_for_weights(ones_data_weights)

    rng = np.random.default_rng(0)
    n = len(ones_data_weights)
    dropped = rng.choice(n, int(n * 0.1), replace=False)

    eps1_data_weights = np.copy(ones_data_weights)
    eps1_data_weights[dropped] = 1 - 0.1
    fs1, _ = params_for_weights(eps1_data_weights)

    eps2_data_weights = np.copy(ones_data_weights)
    eps2_data_weights[dropped] = 1 - 0.2
    fs2, _ = params_for_weights(eps2_data_weights)

    smoothness = agreement(fs0, fs1, fs2)

    return {
        'smoothness': smoothness,
        'primal': primal
    }
