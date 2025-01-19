from transformers import AutoTokenizer
from functools import partial
import numpy as np
from pathlib import Path
import datasets
from tqdm import tqdm
import numpy as np
from functools import cache
import os

def get_ts_paths():
    vs = os.environ.get('TOKENIZER', '8k')
    if vs != '32k':
        tok_path = f"/mnt/xfs/datasets/TinyStories-tokenizer-{vs}/tokenizer-pretrained"
    else:
        tok_path = f"/mnt/xfs/datasets/TinyStories-{vs}-logan/tokenizer-pretrained"

    val_path = f'/mnt/xfs/datasets/TinyStories-{vs}-logan/val'
    train_path = f'/mnt/xfs/datasets/TinyStories-{vs}-logan/train'
    assert Path(tok_path).exists(), tok_path
    assert Path(val_path).exists(), val_path
    assert Path(train_path).exists(), train_path
    emoji = '\U0001F514'
    print(f'>> ALERT {emoji}', tok_path)
    return tok_path, val_path, train_path

def _get_dataset(bs, train_path, val_path, make_labels=False):
    train_ds = datasets.load_from_disk(train_path)
    val_ds = datasets.load_from_disk(val_path)
    train_ds = train_ds.add_column('idx', np.arange(len(train_ds)))
    val_ds = val_ds.add_column('idx', np.arange(len(val_ds)))

    if len(train_ds) % bs != 0:
        shave = len(train_ds) % bs
        train_ds = train_ds.select(range(len(train_ds) - shave))
        # print('>> Shaved off', shave, 'examples from the training set')

    if len(val_ds) % bs != 0:
        shave = len(val_ds) % bs
        val_ds = val_ds.select(range(len(val_ds) - shave))
        # print('>> Shaved off', shave, 'examples from the validation set')

    # print('>> train shape', train_ds)
    # print('>> val shape', train_ds)
    assert 'input_ids' in train_ds.column_names
    if make_labels:
        assert not 'labels' in train_ds.column_names
        print('>> adding labels')
        def add_labels(ds, ds_path):
            ds_path = Path(ds_path)
            cache_path = ds_path.parent / f'{ds_path.name}_labeled_bs{bs}'
            if cache_path.exists():
                print('>> Loading cached dataset with labels')
                return datasets.load_from_disk(str(cache_path))

            print('>> cache not found')
            labels = [x['input_ids'][1:] for x in tqdm(ds)]
            input_ids = [x['input_ids'][:-1] for x in tqdm(ds)]
            new_ds = datasets.Dataset.from_dict({
                'input_ids': input_ids,
                'labels': labels,
                'idx': ds['idx']
            })

            print(f'>> saving dataset to disk @ {cache_path}')
            new_ds.save_to_disk(str(cache_path))
            return new_ds

        train_ds = add_labels(train_ds, train_path)
        val_ds = add_labels(val_ds, val_path)

    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # one_sample = train_ds[0]
    # iis = one_sample['input_ids']
    # labs = one_sample['labels']
    # print('>> detokenized')
    # print(tokenizer.decode(iis))
    # print(iis)
    # print(labs)

    assert 'labels' in train_ds.column_names
    return train_ds, val_ds

def get_wikitext(bs):
    print('>> Loading wikitext')
    train_ds_path = '/mnt/xfs/datasets/wikitext-tokenized/train'
    val_ds_path = '/mnt/xfs/datasets/wikitext-tokenized/val'
    return _get_dataset(bs, train_ds_path, val_ds_path)

def get_tinystories(bs):
    print('>> Loading tinystories')
    TS_VAL_DS_PATH, TS_TRAIN_DS_PATH = get_ts_paths()[1:]
    return _get_dataset(bs, TS_TRAIN_DS_PATH, TS_VAL_DS_PATH, make_labels=True)

def dataset_to_train_order(dataset, seed, epochs, bs):
    rng = np.random.default_rng(seed)
    orderings =[]
    ceil_epochs = int(np.ceil(epochs))
    for _ in range(ceil_epochs):
        ordering = rng.permutation(dataset.shape[0])
        orderings.append(ordering)

    ordering = np.concatenate(orderings)
    ds_size = int(dataset.shape[0] * epochs)
    ds_size = ds_size - (ds_size % bs)

    ordering = ordering[:ds_size]
    assert max(ordering) < dataset.shape[0]
    dataset = dataset.select(ordering)

    return dataset

def process_examples(examples, i, bs, drop):
    if drop is not None and i == drop[0] * bs:
        idxs = examples['idx']
        to_keep = [i for i in range(bs) if idxs[i] != drop[1]]
        assert len(to_keep) == bs - 1
        examples = examples.select(to_keep)

    def _to_arr(ls):
        return np.array(ls).astype(np.int32)

    idx = _to_arr(examples['idx'])
    iis = _to_arr(examples['input_ids'])
    labs = _to_arr(examples['labels'])
    return idx, (iis, labs)

def produce_fn(dataset, bs, this_process):
    def get_batch(i):
        s, e = i * bs, (i+1) * bs
        examples = dataset.select(range(s, e))
        idx, (iis, labs) = this_process(examples, i)
        return idx, (iis, labs)

    n_ba = int(len(dataset) // bs)
    assert n_ba == len(dataset) / bs
    return get_batch, n_ba

def produce_iterator(dataset, bs, drop=None, make_fn=False):
    assert drop is None or (isinstance(drop[0], int) and isinstance(drop[1], int))
    this_process = partial(process_examples, bs=bs, drop=drop)
    if make_fn:
        return produce_fn(dataset, bs, this_process)
    else:
        return _produce_iterator(dataset, bs, this_process)

def _produce_iterator(dataset, bs, this_process):
    for i in range(0, len(dataset), bs):
        examples = dataset.select(range(i, i+bs))
        idx, (iis, labs) = this_process(examples, i)
        yield idx, (iis, labs)

dataset_names = {
    'wikitext': get_wikitext,
    'tinystories': get_tinystories
}

@cache
def loaders_and_order_for_seed(seed, bs, epochs, make_fn=True, maxits=None):
    dataset_name = 'wikitext'
    train_ds, val_ds = dataset_names[dataset_name](bs)

    train_iterator_ds = dataset_to_train_order(train_ds, seed, epochs, bs)
    if maxits is not None:
        if not type(maxits) is int:
            maxits = int(len(train_ds)//bs * maxits)

    fn, num_its = produce_iterator(train_iterator_ds, bs, drop=None,
                                      make_fn=make_fn)
    assert isinstance(num_its, int)
    if maxits is not None:
        num_its = min(num_its, maxits)

    train_iterator = (fn, num_its)
    val_iterator = produce_iterator(val_ds, bs, drop=None, make_fn=make_fn)

    def cache_it(t):
        v, n = t
        assert type(n) == int
        assert callable(v)
        return cache(v), n

    # print('|> train iterator ds', train_iterator_ds)
    # print('|> train iterator', train_iterator)
    f = cache_it(val_iterator)
    return cache_it(train_iterator), f

@cache
def wikitext_tokenizer_maker():
    TS_TOKENIZER_PATH = get_ts_paths()[0]
    tokenizer = AutoTokenizer.from_pretrained(TS_TOKENIZER_PATH)
    return tokenizer
