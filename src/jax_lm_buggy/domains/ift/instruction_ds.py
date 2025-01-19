from datasets import load_dataset
import torch as ch
import dill as pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import cache, partial
# from .gemma_utils import make_tokenizer
from transformers import AutoTokenizer
import torch
# from .less.get_validation_dataset import get_dataset

@cache
def make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    return tokenizer

LESS_DATA_PATH = '/mnt/xfs/home/engstrom/store/less_data/'
import os

# TODO: add other datasets
DATASETS = {
    'dolly': os.path.join(LESS_DATA_PATH, 'train/processed/dolly/dolly_data.jsonl'),
    'flan_v2': os.path.join(LESS_DATA_PATH, 'train/processed/flan_v2/flan_v2_data.jsonl'),
    'oasst1': os.path.join(LESS_DATA_PATH, 'train/processed/oasst1/oasst1_data.jsonl'),
    'cot': os.path.join(LESS_DATA_PATH, 'train/processed/cot/cot_data.jsonl')
}

def concat_messages(messages, tokenizer):
    message_text = ""
    for i, message in enumerate(messages):
        # assert not done
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))

    if message_text[-1] == '\n':
        message_text = message_text[:-1]

    return message_text

def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length,
        truncation=True, add_special_tokens=False)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer),
                    return_tensors='pt', max_length=max_seq_length,
                    truncation=True, add_special_tokens=False
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    input_ids = input_ids.flatten()
    input_ids = ch.concat([ch.tensor([tokenizer.bos_token_id]), input_ids])
    labels = labels.flatten()
    labels = ch.concat([ch.tensor([-100]), labels])

    input_ids = input_ids[:max_seq_length + 1]
    labels = labels[:max_seq_length + 1]
    input_ids = input_ids[:-1]
    labels = labels[1:]
    attention_mask = torch.ones_like(input_ids)

    num_bos_token_ids = (input_ids == tokenizer.bos_token_id).sum()
    num_eos_token_ids = (input_ids == tokenizer.eos_token_id).sum()
    assert num_bos_token_ids == 1
    # ii_decoded =
    clipped = input_ids.shape[0] == max_seq_length
    one_message = num_eos_token_ids == 1 or clipped
    # assert one_message,  f'{num_eos_token_ids}; {input_ids} \n\n {labels} \n\n {attention_mask} \n\n ii_decoded: {tokenizer.decode(input_ids)} \n\n {input_ids.shape} vs {max_seq_length}'

    ret = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }

    ret = {k: v.numpy() for k, v in ret.items()}
    return ret

def get_dataset_cache(dataset_name, max_seq_length):
    import os
    username = os.environ.get('USER')
    cache_dir = Path(f'/mnt/xfs/home/{username}/store/less_data/logan_cache/')
    cache_path = cache_dir / f'{dataset_name}_{max_seq_length}_v3.pkl'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_path

def make_dataset(dataset_name, max_seq_length):
    cache_path = get_dataset_cache(dataset_name, max_seq_length)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                # print('>> DATA MAKING: Loading cache from ', cache_path)
                return pickle.load(f)
        except Exception as e:
            print('>> DATA MAKING Cache file is corrupted, deleting and re-creating at ', cache_path, e)
            Path(cache_path).unlink()

    print('>> DATA MAKING: Cache not found, creating at ', cache_path)

    p = DATASETS[dataset_name]
    df = pd.read_json(p, lines=True, orient='records')
    examples = df.to_dict(orient='records')
    tokenizer = make_tokenizer()

    encoder = partial(encode_with_messages_format, tokenizer=tokenizer,
                      max_seq_length=999999)
    encoded_raw = [encoder(example) for example in tqdm(examples)]

    def filt(x):
        return (x['labels'] != -100).any() and len(x['labels']) <= max_seq_length

    encoded = list(enumerate(filter(filt, encoded_raw)))
    print(encoded_raw[0])
    assert len(encoded) > 0
    print('>> Filtered out', len(encoded_raw) - len(encoded), 'examples of', len(encoded_raw), 'due to no label')

    with open(cache_path, 'wb') as f:
        pickle.dump(encoded, f)

    return encoded

import jax

@cache
def get_gpu_dev():
    return jax.devices('gpu')[0]

class HetSeqBatch():
    @classmethod
    def concat(cls, xs, sort=False):
        raise NotImplementedError('never call this')
        ls = []
        for x in xs:
            ls.extend(x.xs)
            assert x.bucket_size == xs[0].bucket_size
            assert x.filler == xs[0].filler, (x.filler, xs[0].filler)

        if sort:
            ls = sorted(ls, key=lambda x: x.shape[0])

        return cls(ls, xs[0].bucket_size, xs[0].filler)

    def __init__(self, xs, bucket_size, filler):
        if len(xs) > 0:
            assert len(xs[0].shape) == 1

        self.bucket_size = bucket_size
        self.xs = xs
        self.filler = filler

    def subselect(self, premask):
        indices_to_take = np.arange(len(self.xs))[premask]
        ls = [self.xs[int(i)] for i in indices_to_take]
        return HetSeqBatch(ls, self.bucket_size, self.filler)

    def __getitem__(self, sel):
        indices_to_take = np.arange(len(self.xs))[sel]
        ls = [self.xs[int(i)] for i in indices_to_take]
        max_len = max([x.shape[0] for x in ls])

        if max_len % self.bucket_size != 0:
            remainder = self.bucket_size - max_len % self.bucket_size
            max_len += remainder

        ret = self.filler((len(ls), max_len))
        for i, x in enumerate(ls):
            ret[i, :x.shape[0]] = x

        return ret

    def __repr__(self):
        return f'HetSeqBatch({self.xs})'

    def __len__(self):
        return len(self.xs)

INPUT_ID_FILLER = jax.tree_util.Partial(np.zeros, dtype=np.int64)
LABEL_FILLER = jax.tree_util.Partial(np.full, dtype=np.int64, fill_value=-100)

def lm_collector(ls, bucket_size):
    indices, ls = zip(*ls)
    indices = np.array(indices)

    input_id_filler = INPUT_ID_FILLER
    label_filler = LABEL_FILLER

    # ordering = np.argsort([x['input_ids'].shape[0] for x in ls], stable=True)
    ordering = sorted([(i, x['input_ids'].shape[0]) for i, x in enumerate(ls)], key=lambda x: x[1])
    ordering = [x[0] for x in ordering]
    indices = indices[ordering]
    ls = [ls[i] for i in ordering]
    input_ids = [x['input_ids'] for x in ls]
    labels = [x['labels'] for x in ls]

    input_ids = HetSeqBatch(input_ids, bucket_size, input_id_filler)
    labels = HetSeqBatch(labels, bucket_size, label_filler)
    return indices, (input_ids, labels)

def loader_from(ix, ds, indices, bs, bucket_size, collector=lm_collector):
    batch_indices = indices[ix * bs:(ix + 1) * bs]
    batch_data = [ds[i] for i in batch_indices]
    return collector(batch_data, bucket_size)

def _make_train_val(ds, train_frac, bs, seed, epochs, fractional_batches=False,
                    bucket_size=None):
    assert bucket_size is not None
    assert len(ds) > 0
    n = len(ds)
    if not fractional_batches:
        if n % bs != 0:
            shaving = n % bs
            n = n - shaving

    train_n = int(n * train_frac)
    if not fractional_batches:
        train_n = train_n - (train_n % bs)

    val_n = n - train_n

    if not fractional_batches:
        assert train_n % bs == 0
        assert val_n % bs == 0

    assert train_n + val_n == n

    if seed == None:
        seed = 0

    rng = np.random.default_rng(seed)
    train_ds = [(i, ds[i][1]) for i in range(train_n)]
    train_ixs = []
    for _ in range(epochs):
        epoch_ordering = [int(x) for x in rng.permutation(train_n)]
        train_ixs.extend(epoch_ordering)

    train_loader = partial(loader_from, ds=train_ds, indices=train_ixs, bs=bs,
                           bucket_size=bucket_size)
    n_train_it = int(np.ceil(len(train_ixs) / bs))

    if not fractional_batches:
        assert n_train_it * bs == len(train_ixs)

    if val_n > 0:
        raise ValueError('Validation not supported')
        val_ixs = list(range(val_n))
        val_ds = [(ix, ds[i][1]) for ix, i in enumerate(split_shuf_indices[train_n:])]
        assert len(val_ds) == len(val_ixs)
        val_loader = partial(loader_from, ds=val_ds, indices=val_ixs, bs=bs)
        n_val_it = len(val_ixs) // bs

        assert n_val_it * bs == len(val_ixs)
        return (train_loader, n_train_it), (val_loader, n_val_it)

    return (train_loader, n_train_it), None

def combo_dataset(frac=0.5, msl=512, seed=0):
    catted = []
    for name in DATASETS.keys():
        ds = make_dataset(name, msl)
        catted.extend(ds)
        # print('>> ds size of', name, 'is', len(ds))

    n = len(catted)
    rng = np.random.default_rng(seed)
    to_keep = rng.choice(catted, int(n * frac), replace=False)

    # relabel indices
    to_keep = [(i, x[1]) for i, x in enumerate(to_keep)]
    assert len(to_keep) == int(n * frac)

    return to_keep

if __name__ == '__main__':
    tokenizer = make_tokenizer()
    ds = combo_dataset()
    kw = {
        'ds': ds,
        'train_frac': 0.9,
        'bs': 32,
        'seed': 0,
        'epochs': 2
    }

    (train_loader, n_train_it), (val_loader, n_val_it) = _make_train_val(**kw)

    print('>> Doing a dry run of train')
    for i in tqdm(range(n_train_it)):
        l = train_loader(i)
        if i == 0 or i == n_val_it - 1:
            print(i, '@', l)

    print('>> Doing a dry run of val')
    for i in tqdm(range(n_val_it)):
        l = val_loader(i)
        if i == 0 or i == n_val_it - 1:
            print(i, '@', l)
