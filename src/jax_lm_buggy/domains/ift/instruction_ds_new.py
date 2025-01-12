from .instruction_ds import combo_dataset, _make_train_val, \
    make_tokenizer, make_dataset as ift_make_dataset
from .less.get_validation_dataset import get_dataset as less_get_val_dataset
import numpy as np
import jax
from pathlib import Path
import dill
cache_dir = Path('/mnt/xfs/home/engstrom/store/less_data/logan_cache/')
cache_dir.mkdir(exist_ok=True, parents=True)
from functools import cache, partial
import jax.numpy as jnp
from tqdm import tqdm

def make_val_dataset(task, max_length, partition, include_task_in_label):
    assert partition in ['val', 'test']
    assert max_length == 3072
    # specs = (task, max_length, partition, include_task_in_labell cl
    cache_path = cache_dir / f'{task}_{max_length}_{partition}_{include_task_in_label}.pkl'
    data_dir = '/mnt/xfs/home/engstrom/store/less_data'
    tokenizer = make_tokenizer()
    if not cache_path.exists():
        print('>> Making val dataset')
        ds = less_get_val_dataset(task, data_dir=data_dir, tokenizer=tokenizer,
                                  max_length=max_length,  use_chat_format=True,
                                  chat_format='tulu', partition=partition,
                                  include_task_in_label=include_task_in_label,
                                  max_response=1024)
        print('>> Writing to cache path', cache_path)
        with open(cache_path, 'wb') as f:
            dill.dump(ds, f)
    else:
        print('>> reading from cache path', cache_path)
        with open(cache_path, 'rb') as f:
            ds = dill.load(f)

    return ds

# enum for dataset type
import enum
class IFTrainSet(enum.Enum):
    COMBO = 1
    DOLLY = 2
    FLAN_V2 = 3
    OASST1 = 4
    COT = 5

class IFValSet(enum.Enum):
    TYDIQA = 1
    BBH = 2
    MMLU = 3

from metagradients.dataloading import naive_batch_maker, REPLAYBatch, REPLAYMinibatches

def fetch_one_minibatch(batch, s, minibs, bs):
    ixs, (x, y) = batch
    e = min(s + minibs, bs)
    sel = slice(s, e)
    curr_ixs, (curr_x, curr_y) = ixs[sel], (x[sel], y[sel])
    assert len(curr_x.shape) == 2, len(curr_x.shape)
    assert len(curr_y.shape) == 2, len(curr_y.shape)
    assert curr_x.dtype == np.int64, curr_x.dtype
    assert curr_y.dtype == np.int64, curr_y.dtype
    return (curr_ixs, (curr_x, curr_y)), e

def make_ift_iterator(batch, minibs, bs, slice_bound, sharding,
                      *, shear_eps0, data_weights):

    if shear_eps0:
        # premodify the batch
        # ixs: array of indices, x: hetseq, y: hetseq
        ixs, (x, y) = batch
        to_keep = np.where(data_weights[ixs] != 0)[0]
        ixs = ixs[to_keep]
        x = x.subselect(to_keep)
        y = y.subselect(to_keep)
        batch = ixs, (x, y)
        bs = len(ixs)

    if bs > 1024:
        print('>> BIG BATCH SIZE', bs)
        minibatch_tracker = tqdm(total=bs)
    else:
        minibatch_tracker = None

    if not bs in [128, 256]:
        print('>> UNUSUAL BATCH SIZE', bs)

    s = 0
    while True:
        # need to find the right (exclusive) e
        if shear_eps0:
            # check if we can increase minibs
            next_idxs = fetch_one_minibatch(batch, s, minibs, bs)[0][0]
            next_data_weights = data_weights[next_idxs]
            if np.all(next_data_weights == 0):
                this_minibs = minibs * 4
            else:
                this_minibs = minibs
        else:
            this_minibs = minibs

        # first attempt:
        this_minibatch, e = fetch_one_minibatch(batch, s, this_minibs, bs)
        this_length = this_minibatch[1][0].shape[1]
        half_minibatch, half_e = fetch_one_minibatch(batch, s, this_minibs//2, bs)
        half_length = half_minibatch[1][0].shape[1]

        if this_length > slice_bound or half_length < this_length:
            this_minibatch, e = half_minibatch, half_e
            this_length = half_length

        num_devices = len(sharding.mesh.device_ids) if hasattr(sharding, 'mesh') else 1
        if len(this_minibatch[0]) % num_devices != 0:
            this_minibatch, e = fetch_one_minibatch(batch, s, 1, bs)
            this_minibatch = jax.tree.map(jnp.array, this_minibatch)
            this_minibatch = jax.device_put(this_minibatch, jax.devices('gpu')[0])
        else:
            this_minibatch = jax.tree.map(jnp.array, this_minibatch)
            this_minibatch = jax.device_put(this_minibatch, sharding)

        this_minibs_size = len(this_minibatch[0])
        s = e

        if minibatch_tracker is not None:
            minibatch_tracker.update(this_minibs_size)
            minibatch_tracker.set_postfix({'ctx length': this_length})
            if e == bs:
                minibatch_tracker.close()

        yield this_minibatch

        if e == bs:
            break

class IFTREPLAYMinibatches(REPLAYMinibatches):
    def __init__(self, bs, batch, minibs, slice_bound, sharding, *,
                 data_weights, shear_eps0):
        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.slice_bound = slice_bound
        self.sharding = sharding
        self.shear_eps0 = shear_eps0
        self.data_weights = data_weights

    def __iter__(self):
        return make_ift_iterator(self.batch, self.minibs, self.bs,
                                 self.slice_bound, self.sharding,
                                 shear_eps0=self.shear_eps0,
                                 data_weights=self.data_weights)

class IFTREPLAYBatch(REPLAYBatch):
    def __init__(self, batch, minibs, sharding, slice_bound, *, data_weights):
        bs = len(batch[0])
        super().__init__(bs)
        self.batch = batch
        self.minibs = minibs
        self.slice_bound = slice_bound
        self.sharding = sharding
        self.data_weights = data_weights

    def get_minibatches(self, part):
        minibs = int({
            'train': 1,
            'val': 1,
            'meta': 0.5
        }[part] * self.minibs)

        is_not_meta = part != 'meta'

        return IFTREPLAYMinibatches(self.bs, self.batch, minibs,
                                    self.slice_bound, self.sharding,
                                    data_weights=self.data_weights,
                                    shear_eps0=is_not_meta)

# TYDIQA: 1 shot
# MMLU: 5 shot
# BBH: 3 shot
def make_instruction_dataset(*, trainset_type, valset_type, max_length,
                             frac_train=None):
    if trainset_type != IFTrainSet.COMBO:
        assert frac_train is None, (frac_train, trainset_type)
        ds_name = {
            IFTrainSet.DOLLY: 'dolly',
            IFTrainSet.FLAN_V2: 'flan_v2',
            IFTrainSet.OASST1: 'oasst1',
            IFTrainSet.COT: 'cot'
        }[trainset_type]

        train_ds = ift_make_dataset(dataset_name=ds_name, max_seq_length=max_length)

    elif trainset_type == IFTrainSet.COMBO:
        assert frac_train is not None
        train_ds = combo_dataset(frac=frac_train, msl=max_length, seed=0)

    task = {
        IFValSet.TYDIQA: 'tydiqa',
        IFValSet.BBH: 'bbh',
        IFValSet.MMLU: 'mmlu'
    }[valset_type]


    def make_val_style_set(partition_name, *, minibatch_fraction, split_cond=None,
                           split_frac=None, minibatch_seed=None, max_val_length):
        val_ds_hf = make_val_dataset(task, max_val_length, partition_name,
                                     include_task_in_label=True)
        if split_cond is not None:
            assert split_cond is not None and split_frac is not None
            rng = np.random.default_rng(0)
            unique_categories = list(np.unique(val_ds_hf['category']))
            unique_categories = sorted(unique_categories)
            n = len(val_ds_hf)
            all_indices = np.arange(n)
            take_mask = np.zeros_like(all_indices, dtype=bool)
            for cat in unique_categories:
                eligible_indices = all_indices[np.array(val_ds_hf['category']) == cat]
                if valset_type == IFValSet.BBH:
                    assert len(eligible_indices) == 3, len(eligible_indices)
                elif valset_type == IFValSet.MMLU:
                    assert len(eligible_indices) == 5, len(eligible_indices)
                else:
                    raise ValueError(f'unknown valset type {valset_type}')

                index_to_keep = rng.choice(eligible_indices)
                take_mask[index_to_keep] = True

            assert np.isclose(np.sum(take_mask) / n, split_frac, atol=1e-3), np.sum(take_mask) / n

            if split_cond:
                indices = all_indices[take_mask]
            else:
                indices = all_indices[~take_mask]

            val_ds_hf = val_ds_hf.select(list(map(int, indices)))
            print('FINAL VAL SET SIZE', len(val_ds_hf), 'FROM', n)

        if minibatch_fraction < 1:
            assert minibatch_fraction > 0 and minibatch_fraction <= 1
            if minibatch_seed is None:
                minibatch_seed = np.random.randint(0, 100000)
            else:
                assert isinstance(minibatch_seed, int)

            rng = np.random.default_rng(minibatch_seed)
            num_val = len(val_ds_hf)
            num_to_take = int(num_val * minibatch_fraction)
            indices = rng.choice(num_val, num_to_take, replace=False)
            val_ds_hf = val_ds_hf.select(indices)

        # convert to a list
        val_ds = []
        for ii, x in enumerate(val_ds_hf):
            val_ds.append((ii, {
                'input_ids': np.array(x['input_ids']),
                'labels': np.array(x['labels']),
                'attention_mask': np.array(x['attention_mask'])
            }))

        return val_ds

    def dataset_factory(bs, seed, epochs, minibatch_fraction, bucket_size,
                        max_val_length):
        test_ds = make_val_style_set('test', minibatch_fraction=1.,
                                     max_val_length=max_val_length)
        assert task in ['tydiqa', 'bbh', 'mmlu']
        if task != 'tydiqa':
            split_frac= 0.3333333333334 if task == 'bbh' else 0.2
            valval_ds = make_val_style_set('val', minibatch_fraction=1.,
                                           split_cond=True,
                                           split_frac=split_frac,
                                           max_val_length=max_val_length)
            val_ds = make_val_style_set('val',
                                        minibatch_fraction=minibatch_fraction,
                                        split_cond=False, split_frac=split_frac,
                                        max_val_length=max_val_length)
            if task == 'bbh':
                assert len(val_ds) == int(2 * len(valval_ds) * minibatch_fraction), (len(val_ds), len(valval_ds))
            elif task == 'mmlu':
                assert len(val_ds) == int(4 * len(valval_ds) * minibatch_fraction), (len(val_ds), len(valval_ds))
            else:
                raise ValueError(f'task {task} not supported')
        else:
            raise ValueError('TYDIQA not supported')

        train_ret = _make_train_val(train_ds, 1.0, bs, seed, epochs,
                                    bucket_size=bucket_size)
        (train_loader, n_train_it), _ = train_ret

        val_ret = _make_train_val(val_ds, 1.0, bs, None, 1,
                                  fractional_batches=True,
                                  bucket_size=bucket_size)
        (val_loader, n_val_it), _ = val_ret

        valval_ret = _make_train_val(valval_ds, 1.0, bs, None, 1,
                                     fractional_batches=True,
                                     bucket_size=bucket_size)
        (valval_loader, n_valval_it), _ = valval_ret

        test_ret = _make_train_val(test_ds, 1.0, bs, None, 1,
                                   fractional_batches=True,
                                   bucket_size=bucket_size)
        (test_loader, n_test_it), _ = test_ret

        return (train_loader, n_train_it), (val_loader, n_val_it), (test_loader, n_test_it), (valval_loader, n_valval_it)

    return dataset_factory

@cache
def make_gemma_lora_model(seed, lora_dim, lora_std=1.0, lora_multiple=1.0):
    from .gemma_utils import get_model
    (params, const_params), model_apply = get_model(jnp.float32, lora_dim, seed, '2b',
                                                    lora_std=lora_std,
                                                    which_is_zero='b',
                                                    lora_multiple=lora_multiple)

    def model_applier(params, x, const_params, cache=None, return_cache=False,
                        this_model_apply=None):
        ret, cache = this_model_apply((params, const_params), x, cache)

        if return_cache:
            return ret, cache
        else:
            return ret

    model_applier = jax.tree_util.Partial(model_applier,
                                          const_params=const_params,
                                          this_model_apply=jax.tree_util.Partial(model_apply))

    return model_applier, params

def make_less_dataloaders(train_dataset_name, frac, val_dataset_name,
                          max_length, seed, bs, epochs, minibatch_fraction,
                          bucket_size, max_val_length):
    train_dataset = IFTrainSet[train_dataset_name]
    val_dataset = IFValSet[val_dataset_name]

    kw = {
        'trainset_type': train_dataset,
        'valset_type': val_dataset,
        'max_length': max_length,
        'frac_train': frac if train_dataset == IFTrainSet['COMBO'] else None
    }

    loader_maker = make_instruction_dataset(**kw)
    return loader_maker(bs=bs, seed=seed, epochs=epochs,
                        minibatch_fraction=minibatch_fraction,
                        bucket_size=bucket_size,
                        max_val_length=max_val_length)

def inspect_batch(l):
    tokenizer = make_tokenizer()
    input_ids = l[1][0][:2]
    labels = l[1][1][:2]
    for zz in l[1][1].xs:
        assert zz[-1] == tokenizer.eos_token_id, zz

    print('>>> input ids')
    print([list(x) for x in input_ids])
    print('-' * 80)
    print('>>> labels')
    print([list(x) for x in labels])
    print('-' * 80)
    print('>>> decoded')
    print(tokenizer.decode(input_ids[0]))

if __name__ == '__main__':
    import sys
    arg = sys.argv[1]
    valset_type = {
        'tydiqa': IFValSet.TYDIQA,
        'bbh': IFValSet.BBH,
        'mmlu': IFValSet.MMLU
    }[arg]

    arg2 = sys.argv[2]
    trainset_type = {
        'dolly': IFTrainSet.DOLLY,
        'flan_v2': IFTrainSet.FLAN_V2,
        'oasst1': IFTrainSet.OASST1,
        'cot': IFTrainSet.COT,
        'combo': IFTrainSet.COMBO
    }[arg2]

    ft = 1 if trainset_type == IFTrainSet.COMBO else None

    factory = make_instruction_dataset(trainset_type=trainset_type,
                                       valset_type=valset_type, max_length=1024,
                                       frac_train=ft)

    (train_loader, n_train_it), (val_loader, n_val_it), (test_loader, n_test_it) = factory(256, 1, 1)

    # print('>> Doing a dry run of train')
    tokenizer = make_tokenizer()
    def print_it(loader, its):
        for i in tqdm(range(its)):
            l = loader(i)
            if i == 0 or i == n_val_it - 1:
                inspect_batch(l)

    print('*' * 80)
    print('*' * 80)
    print('*' * 80)
    print('VAL SET')

    print_it(val_loader, n_val_it)

    print('*' * 80)
    print('*' * 80)
    print('*' * 80)
    print('TEST SET')

    print_it(test_loader, n_test_it)
