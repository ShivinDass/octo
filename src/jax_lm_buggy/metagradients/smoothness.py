import jax
import numpy as np

def agreement(model_base, model_eps, model_2eps):
    all_agreements = 0.
    all_n = 0.

    def agreement(base, eps1, eps2):
        diff1 = eps1 - base
        diff2 = eps2 - eps1

        diff1 = diff1.flatten()
        diff2 = diff2.flatten()

        agreement = (np.sign(diff1) == np.sign(diff2)).astype(np.float32)
        n = len(agreement)
        nonlocal all_agreements, all_n
        all_agreements += agreement.sum()
        all_n += n

    jax.tree.map(agreement, model_base, model_eps, model_2eps)
    return all_agreements / all_n