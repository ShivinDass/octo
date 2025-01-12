from .config import get_model_config
from .model import GemmaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import dill as pickle
import torch as ch
import jax
ch.backends.cuda.matmul.allow_tf32 = True
ch.backends.cudnn.allow_tf32 = True

def get_attention_mask_and_positions(example, pad_id):
    pad_mask = example != pad_id
    current_token_position = build_positions_from_mask(pad_mask)
    attention_mask = make_causal_attn_mask(pad_mask)
    return current_token_position, attention_mask

def make_causal_attn_mask(input_mask):
    # input_mask: bool, [B, T]
    seq_len = input_mask.shape[-1]
    # input_mask: [B, 1, T]
    attn_mask = input_mask[..., None, :]
    # causal mask: [T, T]
    causal_mask = ch.tril(ch.ones((seq_len, seq_len), dtype=ch.bool, device='cuda'))
    # Prefixes can be attended by all tokens
    # causal_mask: [1, T, T]
    causal_mask = causal_mask[None, ...]
    # attn_mask: [B, T, T]
    attn_mask = causal_mask * attn_mask
    return attn_mask

def build_positions_from_mask(input_mask):
    positions = ch.cumsum(input_mask[0], axis=-1)
    return positions - (positions >= 1).to(dtype=positions.dtype, device='cuda',
                                           non_blocking=True)

def convert_to_path(p):
    ls = []
    for i, key in enumerate(p):
        if key == 'params' and i == 0:
            ls.append('model')
        elif 'layer_' in key:
            this_layer = int(key.replace('layer_', ''))
            ls.extend(['layers', str(this_layer)])
        else:
            ls.append(key)

    candidate = '.'.join(ls)
    if candidate == 'model.embedder.input_embedding':
        return 'embedder.weight'

    return candidate

import numpy as np
from functools import partial

def path_to_mod(mod, p):
    for key in p.split('.'):
        mod = getattr(mod, key)

    return mod

def bufferize_matching(module, prefix, fn):
    for name, child_module in module.named_children():
        if child_module != module:
            this_prefix = prefix + [name]
            bufferize_matching(child_module, this_prefix, fn)

    nps = list(module.named_parameters(recurse=False))
    for name, param in nps:
        this_path = prefix + [name]
        this_path = '.'.join(this_path)
        if fn(this_path):
            # print('>> Bufferizing', this_path)
            delattr(module, name)
            module.register_buffer(name, param)

    # modules = list(module.modules())
    # module = modules[0]
    # named_params = list(module.named_parameters(recurse=False))
    # for name, param in named_params:
    #     if fn(name):
    #         print('>> Bufferizing', name)
    #         delattr(module, name)
    #         module.register_buffer(name, param)

    # for module in modules[1:]:
    #     bufferize_matching(module, fn)

def restore_model(lora_params):
    model = GemmaForCausalLM(get_model_config('2b'))
    print('>> STATEFULLY RESTORING MODEL')
    # with open(', 'rb') as f:
    #     fixed_params = pickle.load(f)
    fixed_params = ch.load('/mnt/xfs/home/engstrom/store/gemma2b_params/fixed_params_final.torch',
                           map_location='cuda')

    # now replace all the params in model with those of lora_params and fixed_params
    # state_dict = model.state_dict()

    # all_paths = set(state_dict.keys())

    # replace the params in the model with those from lora_params
    def replace_param(p, x, is_lora):
        # nonlocal all_paths
        assert not x is None
        assert not (bool('lora' in str(p).lower()) ^ bool(is_lora))

        fmt_path = [z.key for z in p]
        sd_tensor_path = convert_to_path(fmt_path)
        tensor_to_replace = path_to_mod(model, sd_tensor_path)
        old_data = tensor_to_replace.data
        # new_data = ch.from_numpy(x).to(non_blocking=True, device=tensor_to_replace.device)
        new_data = x.to(dtype=ch.bfloat16, non_blocking=True)
        assert old_data.shape == new_data.shape, f"Shape mismatch: {old_data.shape} vs {new_data.shape}"
        tensor_to_replace.data = new_data

        # ensure that we haven't previously replaced the tensor
        # assert sd_tensor_path in all_paths, f"Could not find {sd_tensor_path} in state_dict"
        # all_paths.remove(sd_tensor_path)

    jax.tree_util.tree_map_with_path(partial(replace_param, is_lora=False), fixed_params)
    jax.tree_util.tree_map_with_path(partial(replace_param, is_lora=True), jax.tree.map(ch.zeros_like, lora_params))

    # assert all_paths == {'freqs_cis'}, "Some parameters were not replaced: " + str(all_paths)
    print(">> SUCCESSFULLY RESTORED MODEL; loading from state_dict")
    # model.load_state_dict(state_dict)
    model.freqs_cis.data = model.freqs_cis.data.to(device=model.embedder.weight.device)
    bufferize_matching(model, [], lambda x: not ('lora' in x.lower()))
    # print pytorch params with paths
    # print('ALL PARAMS:')
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # print('END ALL PARAMS')

    return model

def main():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    input_ids = tokenizer.encode("""The authors started by extracting all Reddit post urls from the Reddit submissions dataset. These links were deduplicated, filtered to exclude non-html content, and then shuffled randomly. The links were then distributed to several machines in parallel for download, and all web pages were extracted using the newspaper python package. Using Facebook FastText, non-English web pages were filtered out.

Subsequently, near-duplicate documents were identified using local-sensitivity hashing (LSH). Documents were hashed into sets of 5-grams and all documents that had a similarity threshold of greater than 0.5 were removed. The the remaining documents were tokenized, and documents with fewer than 128 tokens were removed. This left 38GB of text data (40GB using SI units) from 8,013,769 documents.""", return_tensors='pt')
    pad_id = tokenizer.pad_token_id

    lora_params = ch.load('/mnt/xfs/home/engstrom/store/gemma2b_params/lora_params_final.torch', map_location='cuda')
    model = restore_model(lora_params)
    labels = input_ids.clone()
    labels = labels[:, 1:]
    input_ids = input_ids[:, :-1]

    input_ids = input_ids.to(model.embedder.weight.device)
    # positions = positions.to(model.embedder.weight.device)
    # attn_mask = attn_mask.to(model.embedder.weight.device)
    labels = labels.to(model.embedder.weight.device)
    logits = model(input_token_ids=input_ids, y=input_ids)[0] #, input_positions=positions,
                #    mask=attn_mask)[0]

    flat_logits = logits.view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    loss = ch.nn.functional.cross_entropy(flat_logits, flat_labels,
                                          reduction='mean')
    print('LOSS:', loss, loss.dtype)

    from .model import convert_to_linear_einsums
    convert_to_linear_einsums(model)

    new_logits = model(input_token_ids=input_ids, y=input_ids)[0] #, input_positions=positions,
    diff = ch.abs(new_logits - logits).max()
    print('DIFF:', diff)
    print('NEW LOSS', ch.nn.functional.cross_entropy(new_logits.view(-1, new_logits.shape[-1]),
                                                      flat_labels, reduction='mean'))


if __name__ == '__main__':
    main()