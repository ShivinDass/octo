import jax
jax.config.update('jax_platform_name', 'cpu')

from .kronfluence_utils import LanguageModelingTask
from .gpt.utils import load_model

import numpy as np
import tqdm
import sys
from pathlib import Path
import os
import torch as ch
import dill as pickle
import hashlib
from kronfluence.analyzer import Analyzer, prepare_model
from transformers import default_data_collator
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.arguments import FactorArguments, ScoreArguments
from domains.ift.instruction_ds_new import make_less_dataloaders, IFTREPLAYBatch
from .gemma.demo import restore_model as make_gemma_model
from functools import partial
from metagradients.dataloading import naive_batch_maker

def str_hash(s):
    s = str(s)
    return str(hashlib.md5(s.encode()).hexdigest())

def main():
    if len(sys.argv) < 3:
        print("Please provide input data path and params path as command line arguments.")
        sys.exit(1)

    input_data_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    with open(input_data_path, 'rb') as f:
        kwargs = pickle.load(f)

    hasher = str_hash(input_data_path)
    return _main(kwargs, output_path, hasher)

def main_gemma():
    args = ('COMBO', 0.6, 'MMLU', 2048, 16, 128, 1, 1.0, 128, 3072)
    ret = make_less_dataloaders(*args)
    train_pair, val_pair, test_pair, valval_pair = ret

    batch_maker = partial(IFTREPLAYBatch, slice_bound=513,
                          data_weights=np.ones(1_000_000, dtype=np.float32))

    def to_mgs_loader(loader, n_its):
        return partial(naive_batch_maker, get_batch=loader, minibs=4,
                       num_batches=n_its, batch_maker=batch_maker), n_its

    train_loader, train_its = to_mgs_loader(*train_pair)
    val_loader, val_its = to_mgs_loader(*val_pair)

    load_path = '/mnt/xfs/home/engstrom/store/gemma2b_params/lora_params_final.torch'
    params = ch.load(load_path, map_location='cuda')

    kwargs = {
        'train_ds': (train_loader, train_its),
        'val_ds': (val_loader, val_its),
        'test_indices': None,
        'domain': 'ift',
        'params': params
    }

    return _main(kwargs, Path('/tmp/gemma_out.pkl'), str_hash)

def _main(kwargs, output_path, hasher):
    # dict with:
    # - train_ds: tuple of train_loader_fn and n_train_iter
    # - val_ds: tuple of val_loader_fn and n_val_iter
    # - test_indices: indices to test set
    # - domain: 'wikitext' or 'cifar'
    # - params: model parameters WITHOUT state

    # Load input data

    if 'final_params' in kwargs.keys():
        kwargs['params'] = kwargs.pop('final_params')

    kwargs = {k: kwargs[k] for k in [
        'train_ds', 'val_ds', 'test_indices', 'domain', 'params'
    ]}

    kwargs['hasher'] = hasher
    kwargs['output_path'] = output_path
    kwargs['test_indices'] = None
    ret = setup_baseline(**kwargs)
    ds_train, ds_val, model, hasher, output_path, test_indices = ret
    from .trak import trak_matrix
    task = 'ift'
    if os.environ.get('KRONFLUENCE', '0') == '1':
        kronfluence_fn(ds_train, ds_val, model, hasher, output_path, test_indices, task)
    else:
        trak_matrix(model, [model.state_dict()], ds_train, ds_val, task, hasher)


def make_gpt2_model(params):
    tup = load_model('gpt2', block_size=1024, device='cuda')
    model, model_args = tup[0], tup[-1]
    for name, param in model.named_parameters():
        jparam = params['params']
        for subpart in name.split('.'):
            if subpart == 'h' or subpart == 'transformer':
                continue

            if subpart == 'weight':
                mappings = ['kernel', 'scale', 'embedding', 'weight']
                for m in mappings:
                    if m in jparam.keys():
                        subpart = m
                        assert sum([1 if m in jparam.keys() else 0 for m in mappings]) == 1
                        break

                if subpart == 'weight':
                    raise ValueError(f"Could not find a mapping for {name} given {jparam.keys()}")

            jparam = jparam[subpart]

        jparam = ch.from_numpy(jparam)
        if not jparam.shape == param.shape:
            if jparam.T.shape == param.shape:
                jparam = jparam.T

        unique_sides = set(map(int, jparam.shape))
        if len(unique_sides) != len(jparam.shape):
            jparam = jparam.T

        assert jparam.shape == param.shape, f"Shape mismatch: {jparam.shape} vs {param.shape}"
        assert isinstance(jparam, ch.Tensor)
        param.data = jparam

    return model

def lm_loss(logits, labels):
    '''
    xent loss for language modeling across samples
    -100 ignores token
    takes average token across only non -100 tokens
    '''
    flattened_labels = labels.view(-1)
    flattened_logits = logits.view(flattened_labels.shape[0], -1)

    # mask out -100 tokens
    mask = (flattened_labels != -100)
    out = ch.zeros(flattened_labels.shape[0], dtype=logits.dtype,
                   device=flattened_labels.device)
    losses = ch.nn.functional.cross_entropy(flattened_logits[mask],
                                            flattened_labels[mask],
                                            reduction='none')

    out[mask] = losses
    out = out.view(labels.shape)
    count_per_slot = mask.to(dtype=logits.dtype, non_blocking=True).view(labels.shape).sum(-1)

    avg_loss = out.sum(dim=1) / count_per_slot

    return avg_loss

def make_loss(domain):
    if domain in ['wikitext', 'ift']:
        return lm_loss

    raise ValueError(f"Domain {domain} not recognized.")

def setup_baseline(params, train_ds, val_ds, test_indices, domain,
                   output_path, hasher):
    train_loader_fn, n_train_iter = train_ds
    val_loader_fn, n_val_iter = val_ds
    del train_ds
    del val_ds
    model = make_model(params, domain).cuda()
    losser = make_loss(domain)

    total_loss = 0
    n_total = 0

    with ch.no_grad():
        batches = val_loader_fn(0, n_val_iter, sharding=None)
        for batch in batches:
            minibatches = batch.get_minibatches('val')
            # for i in tqdm.trange(n_val_iter):
            for mb in minibatches:
                idx, (x, y) = mb
                idx, (x, y) = np.array(idx), (np.array(x), np.array(y))
                x = ch.from_numpy(x).cuda().long()
                y = ch.from_numpy(y).cuda().long()
                with ch.no_grad():
                    logits, _ = model(x, y)
                    loss = losser(logits, y)
                    total_loss += loss.sum()
                    n_total += idx.shape[0]

        avg_loss = total_loss / n_total
        print('>> Avg loss', avg_loss)

    if domain == 'wikitext' or domain == 'ift':
        ds_train = make_dataset(train_loader_fn, n_train_iter)
        ds_val = make_dataset(val_loader_fn, n_val_iter)
    else:
        raise ValueError(f"Domain {domain} not recognized for dataset creation.")

    return ds_train, ds_val, model, hasher, output_path, test_indices

class GemmaTask(LanguageModelingTask):
    def tracked_modules(self):
        output = []
        num_layers = 18
        attributes = ["q_lora_a", "q_lora_b", "kv_lora_a", "kv_lora_b", "o_lora_a", "o_lora_b"]

        for layer in range(num_layers):
            for attr in attributes:
                output.append(f"model.layers.{layer}.attn.{attr}")

        return output

def kronfluence_fn(ds_train, ds_val, model, hasher, output_path, test_indices, task):
    if task == 'ift':
        from .gemma.model import convert_to_linear_einsums
        convert_to_linear_einsums(model)
        task = GemmaTask()
    else:
        task = LanguageModelingTask()

    for name, mod in model.named_modules():
        print(name)

    sys.exit(1)

    model = prepare_model(model=model, task=task)

    analyzer = Analyzer(
        analysis_name=f'{task}_{hasher}',
        model=model,
        task=task)

    def ddc(*args, **kwargs):
        # list of x,y pairs
        xsys, = args
        xs, ys = zip(*xsys)
        xs = ch.stack(xs).cuda()
        ys = ch.stack(ys).cuda()
        return (xs, ys)

    dataloader_kwargs = DataLoaderKwargs(collate_fn=ddc, num_workers=0)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    factor_args = FactorArguments(strategy='ekfac')

    if task == 'wikitext':
        bs = 16
    elif task == 'ift':
        bs = 2
    else:
        raise ValueError(f"Task {task} not recognized.")

    analyzer.fit_all_factors(
        factors_name='ekfac',
        dataset=ds_train,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=True,
        initial_per_device_batch_size_attempt=bs
    )

    try:
        rank = None
        score_args = ScoreArguments(query_gradient_rank=rank,
                                    query_gradient_svd_dtype=ch.float32)
        scores_name = 'ekfac_pairwise'
        factor_strategy = 'ekfac'
        analyzer.compute_pairwise_scores(
            scores_name=scores_name,
            score_args=score_args,
            factors_name=factor_strategy,
            query_dataset=ds_val,
            query_indices=test_indices,
            train_dataset=ds_train,
            per_device_query_batch_size=16,
            per_device_train_batch_size=16,
            overwrite_output_dir=True)

        scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
        try:
            scores = scores.cpu().detach().numpy()
        except:
            scores = np.array(scores)
        print(f">> Scores shape: {scores.shape}")

    except Exception as e:
        print('>> Error in computing pairwise scores:', e)
        import pdb; pdb.set_trace()

    import dill as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(scores, f)

def make_dataset(get_batch, num_batches):
    mapper = {}
    # for i in range(num_batches):
    batches = get_batch(0, num_batches, sharding=None)
    for batch in batches:
        minibatches = batch.get_minibatches('val')
        for mb in minibatches:
            idx, (x, y) = mb
            idx, (x, y) = np.array(idx), (np.array(x), np.array(y))
            idx = list(map(int, idx))
            for this_idx, this_x, this_y in zip(idx, x, y):
                mapper[this_idx] = (this_x, this_y)

    max_ix = max(mapper.keys())
    ret = []
    for i in range(max_ix + 1):
        ret.append(mapper[i])

    # Convert to TensorDataset
    longest_x = max(len(x) for x, _ in ret)
    longest_y = max(len(y) for _, y in ret)
    assert longest_x == longest_y
    print('MAX LENGTH', longest_x)

    def pad_it(x, pad_value):
        final_sample = ch.ones((longest_x,), device='cpu', dtype=ch.int64) * pad_value
        final_sample[:len(x)] = x.cpu()
        return final_sample

    x_data = ch.stack([pad_it(ch.from_numpy(x), 0).long() for x, _ in ret])
    y_data = ch.stack([pad_it(ch.from_numpy(y), -100).long() for _, y in ret])
    dataset = ch.utils.data.TensorDataset(x_data, y_data)
    return dataset

def make_model(params, domain):
    if domain == 'wikitext':
        return make_gpt2_model(params)
    elif domain == 'ift':
        return make_gemma_model(params)
    else:
        raise ValueError(f"Domain {domain} not recognized.")

if __name__ == '__main__':
    main_gemma()
