from types import SimpleNamespace
from itertools import product
from slapreduce import slap
import importlib
import os
from pathlib import Path
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def make_cfgs():
    maxlrs = [0.05] # 0.1, 0.05, 0.15]
    vjp_iterates = [-100]
    flr = [.5] 
    final_model_lrs = [1e-1]
    # 4 * 6
    root_epss = [1e-7] # , 1e-9, 1e-10] # , 1e-11, 1e-13]

    # b1b2s = [(0.95, 0.975), (0.975, 0.9875)]
    # b1 = os.environ['B1']
    # if b1 == '0.95':
    #     b1b2s = 
    # elif b1 == '0.975':
    # b1b2s = [(0.975, 0.9875)] # 
    b1b2s = [(0.95, 0.975), (0.975, 0.9875)] # 
    # else:
    #     raise ValueError('b1 must be 0.95 or 0.975')
    # TODO:
    # - check width again after
    # - then grid over number of datapoints

    print('>> THIS B1B2:', b1b2s)
     #learning_rates = [0.0003, 0.0002, 0.0004]
    learning_rates = [0.0008, 0.0006] #0.0012 * 1.75**k for k in range(-4, 2)]
    tdfs = [0.6]
    warmup_its = [60]
    widths = [128]
    stds = [1.]
    bss = [256]
    wds = [1e-5]
    LDSs = [False]
    sgps = [None]
    this_product = product(vjp_iterates, maxlrs, flr, root_epss, b1b2s,
                           learning_rates, wds, LDSs, bss, widths, stds, tdfs,
                           sgps, warmup_its, final_model_lrs)

    cfgs = []
    for vjp_iterate, maxlr, flr, root_eps, b1b2, learning_rate, wd, LDS, bs, width, std, tdf, start_grad_path, wit, fmlr in this_product:
        b1, b2 = b1b2
        this_lr = learning_rate
        cfgs.append({
            'strategy': 'random_descent',
            'init_frac_included': 1.0,
            'steps': 6,
            'model_lr': this_lr,
            'max_lr': maxlr,
            'lora_std': std,
            'final_lr': flr * maxlr,
            'final_model_lr': fmlr,
            'vjp_iterate': vjp_iterate,
            'train_data_frac': tdf,
            'task': 'MMLU',
            'data_seed': 16,
            'root_eps': root_eps,
            'b1': b1,
            'b2': b2,
            'wd': wd,
            'warmup_its': wit,
            'LDS': LDS,
            'bs': bs,
            'lora_width': width,
            'start_grad_path': start_grad_path
        })

    return cfgs

def fn(**kw):
    # from .calculate_smoothness import smoothness_for_cfg
    mod = importlib.import_module('.optimize_data', 'domains.ift')
    return mod.run_optimize(**kw)

def main():
    cfgs = make_cfgs()
    gres = {
        'exclude': 'deep-gpu-[10,11],deep-chungus-[1-5]',
        'gres': 'gpu:4',
        'cpus_per_task':4,
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