from types import SimpleNamespace
from itertools import product
from slapreduce import slap
import importlib
import os
from pathlib import Path
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

cfg = SimpleNamespace(
    bs=128,
    lr=0.0003,
    wd=1e-6,
    lr_warmup_its=40,
    b1=0.95,
    b2=0.975,
    min_lr_relative=1e-8,
    final_min_lr_relative=0.1,
    eps=1e-11,
    eps_sqrt=1e-11,
    selective_wd=True,
    dtype='float32',
    factored_lr_wd=True,
    anneal_type='linear',
    eps_steps=100,
    eps0=1e-11,
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

def make_cfgs():
    base_cfg = cfg
    b1_b2s = [(0.95, 0.975), (0.9, 0.95), (0.975, 0.9875)]
    eps0_epss = [(k, k) for k in [1e-9, 1e-11, 1e-13]]
    final_min_lr_relatives = [0.1, 0.2, 0.5]
    eps_stepss = [100]
    mom0s = [1]
    lora_stds = [1e2, 1e0]
    learning_rates = [0.000275, 0.0002, 0.00035, 0.000425]

    cfgs = []
    this_product = product(b1_b2s, eps0_epss, mom0s, lora_stds, learning_rates,
                           eps_stepss, final_min_lr_relatives)
    for (b1, b2), (eps0, eps), mom0, lora_std, lr, eps_steps, fmlrr in this_product:
        update = {
            'train_frac': 0.4,
            'b1': b1,
            'b2': b2,
            'eps_steps': eps_steps,
            'eps': eps,
            'eps_sqrt': eps,
            'eps0': eps0,
            'mom0': mom0,
            'lora_std': lora_std,
            'final_min_lr_relative': fmlrr,
            'lr': lr
        } 

        new_cfg_dict = (base_cfg.__dict__ | update)
        new_cfg = SimpleNamespace(**new_cfg_dict)
        cfgs.append({'cfg': new_cfg})

    return cfgs

def fn(cfg):
    # from .calculate_smoothness import smoothness_for_cfg
    mod = importlib.import_module('.calculate_smoothness', 'domains.ift')
    return mod.smoothness_for_cfg(cfg)

def main():
    cfgs = make_cfgs()
    gres = {
        'exclude': 'deep-gpu-[10,11],deep-chungus-[1-5]',
        'gres': 'gpu:1',
        'cpus_per_task':1,
        'nodes': 1,
    }

    name = os.environ['NAME']
    partition = 'h100'
    save_dir = Path('/mnt/xfs/home/engstrom/store/gemma_smoothness/') / name
    save_dir.mkdir(exist_ok=True, parents=True)
    slap(fn, cfgs, save_dir, gres=gres, partition=partition,
         block=False, job_name=name)

if __name__ == '__main__':
    main()