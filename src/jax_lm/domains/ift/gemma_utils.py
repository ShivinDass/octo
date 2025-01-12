from jax import numpy as jnp
from functools import partial
jnp.set_printoptions(threshold=1)

import os
import re
import jax
import jax.numpy as jnp
import tensorflow as tf
import hashlib
from gemma import params as params_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm
import kagglehub
from functools import cache
import numpy as np
from metagradients.dlpack import dlpack_blocking_gpu2cpu
import torch as ch

@cache
def get_tokenizer_ckpt_paths(variant):
    assert variant in ['7b', '2b', '8b', '2b-it']
    variant = '7b' if variant == '8b' else variant
    from pathlib import Path
    homedir = Path('/mnt/xfs/home/') / os.environ['USER']
    gemma_path = None
    for i in range(3):
        gemma_path_maybe = homedir / f'.cache/kagglehub/models/google/gemma/flax/{variant}/{i}'
        if gemma_path_maybe.exists():
            gemma_path = gemma_path_maybe
            break

    if gemma_path is None:
        gemma_path = kagglehub.model_download(f'google/gemma/flax/{variant}')

    ckpt_path = os.path.join(gemma_path, variant)
    tokenizer_path = os.path.join(gemma_path, 'tokenizer.model')
    return ckpt_path, tokenizer_path

class GemmaTokenizer:
  def __init__(self,
               spm_processor: spm.SentencePieceProcessor):
    self._spm_processor = spm_processor

  @property
  def pad_id(self) -> int:
    """Fast access to the pad ID."""
    return self._spm_processor.pad_id()

  def tokenize(self,
               example: str | bytes,
               prefix: str = '',
               suffix: str = '',
               add_eos: bool = False) -> jax.Array:
    """
    The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
               sequence.
    Returns:
      Tokens corresponding to the input string.
    """
    int_list = [self._spm_processor.bos_id()]
    int_list.extend(self._spm_processor.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self._spm_processor.eos_id())

    return jnp.array(int_list, dtype=jnp.int32)

  def tokenize_tf_op(self,
                     str_tensor: tf.Tensor,
                     prefix: str = '',
                     suffix: str = '',
                     add_eos: bool = True) -> tf.Tensor:
    """A TensorFlow operator for the tokenize function."""
    encoded = tf.numpy_function(
        self.tokenize,
        [str_tensor, prefix, suffix, add_eos],
        tf.int32)
    encoded.set_shape([None])
    return encoded

  def to_string(self, tokens: jax.Array) -> str:
    """Convert an array of tokens to a string."""
    return self._spm_processor.EncodeIds(tokens.tolist())

@cache
def make_tokenizer(variant='2b'):
    vocab = spm.SentencePieceProcessor()
    _, TOKENIZER_PATH = get_tokenizer_ckpt_paths(variant)
    vocab.Load(TOKENIZER_PATH)
    tokenizer = GemmaTokenizer(vocab)
    return tokenizer

def get_attention_mask_and_positions(example: jax.Array,
                                     pad_id : int,
                                     ) -> tuple[jax.Array, jax.Array]:
    """Builds the position and attention mask vectors from the given tokens."""
    pad_mask = example != pad_id
    current_token_position = transformer_lib.build_positions_from_mask(pad_mask)
    attention_mask = transformer_lib.make_causal_attn_mask(pad_mask)
    return current_token_position, attention_mask

def take_only_lora(path, x):
    str_path = '/'.join([k.key for k in path])
    if 'lora' in str_path:
        return x
    else:
        return 'none'

def get_param_size_in_bytes(params):
    sizes = [x.nbytes for x in jax.tree_flatten(params)[0]]
    return sum(sizes)

def compact_format(original):
    # Extract the keys from the original string using regex
    keys = re.findall(r"DictKey\(key='(.*?)'\)", original)
    # Join the keys with a " -> " separator
    return "->".join(keys)

def replace_param_subset(subset, superset):
    full_paths = jax.tree_util.tree_map_with_path(lambda path, _: str(path), superset)
    fixed_param_paths = jax.tree_util.tree_map_with_path(lambda path, _: str(path), subset)

    flat_values, treedef = jax.tree_util.tree_flatten(superset)
    flat_paths, _ = jax.tree_util.tree_flatten(full_paths)
    full_params_dict = dict(zip(flat_paths, flat_values))

    flat_fvalues, _ = jax.tree_util.tree_flatten(subset)
    flat_fpaths, _ = jax.tree_util.tree_flatten(fixed_param_paths)
    fixed_params_dict = dict(zip(flat_fpaths, flat_fvalues))
    fmt_str = '|'.join([compact_format(v) for v in full_params_dict.keys()])

    for k, v in fixed_params_dict.items():
        try:
            assert k in full_params_dict, (k, fmt_str)
        except Exception as e:
            print(e)

            for k in range(20):
                print(k, f'layer_{k}-' in fmt_str)
            import pdb; pdb.set_trace()

        full_params_dict[k] = v

    new_flat_values = [full_params_dict[k] for k in flat_paths]
    combined_tree = jax.tree_util.tree_unflatten(treedef, new_flat_values)
    return combined_tree


def str_hash(s):
    s = str(s)
    return eval('0x' + (hashlib.md5(s.encode())).hexdigest())

@jax.jit
def _trunc_normal_like(keyseed, a, std):
    key = jax.random.PRNGKey(keyseed)
    tnormal_samples = jax.random.truncated_normal(key, lower=-2.576,
                                                    upper=2.576,
                                                    shape=a.shape,
                                                    dtype=jnp.float32) * std
    return tnormal_samples

def get_model(casting, lora_dim, lora_seed, variant, which_is_zero='b',
              lora_std=1., *, lora_multiple):
    ckpt_path = get_tokenizer_ckpt_paths(variant)[0]
    params = params_lib.load_and_format_params(ckpt_path)
    if casting is not None:
        def cast_to_float32(x):
            if isinstance(x, jnp.ndarray) and x.dtype in [jnp.float16, jnp.bfloat16]:
                return x.astype(casting)

            return x

        params = jax.tree_util.tree_map(cast_to_float32, params)

    cfg = transformer_lib.TransformerConfig.from_params(params,
                                                        cache_size=4096,
                                                        lora_dim=lora_dim,
                                                        lora_multiple=lora_multiple)
    model = transformer_lib.Transformer(config=cfg)
    fixed_params = {'params': params['transformer']}

    if lora_dim:
        tokenizer = make_tokenizer(variant)
        from metagradients.utils import make_shardings
        _, replicated_sharding = make_shardings()

        input_tokens = tokenizer.tokenize("Hello, my dog is cute")[None, ...]
        tok_positions, attn_mask = get_attention_mask_and_positions(input_tokens,
                                                                    tokenizer.pad_id)

        rng = jax.random.PRNGKey(lora_seed)
        tup = rng, input_tokens, tok_positions, attn_mask
        tup = jax.device_put(tup, replicated_sharding)
        rng, input_tokens, tok_positions, attn_mask = tup

        full_params = jax.jit(model.init)(rng, input_tokens, tok_positions, None, attn_mask)
        full_params = jax.tree_util.tree_map_with_path(take_only_lora, full_params)
        params_lora_filt = jax.tree_util.tree_map(lambda x: None if x == 'none' else x,
                                                full_params)

        lora_size = get_param_size_in_bytes(params_lora_filt) / 1024 / 1024
        print('>> LORA param size in mb', lora_size)

    if lora_std != 0 and lora_dim:
        def reinitialize_lora(p, x):
            this_seed = lora_seed + (str_hash(str(x)) % 1000000)
            assert 'lora' in str(p)
            std_mul = lora_std
            this_status = 'b' if '_b' in str(p) else 'a'
            if this_status == 'a':
                try:
                    assert '_a' in str(p)
                except:
                    import pdb; pdb.set_trace()
            else:
                assert not '_a' in str(p)

            if which_is_zero == 'b':
                if this_status == 'b':
                    std = 0.
                else:
                    std = 1./x.shape[-2]
            else:
                if this_status == 'a':
                    std = 0.
                else:
                    std = 1./x.shape[-2]

            std *= std_mul
            try:
                res = _trunc_normal_like(this_seed, x, std)
            except:
                import pdb; pdb.set_trace()

            return res

        params_lora_filt = jax.tree_util.tree_map_with_path(reinitialize_lora, params_lora_filt)
    else:
        params_lora_filt = None

    def wrapped_model_apply(params, *args, model, **kwargs):
        lora_params, const_params = params
        if lora_dim:
            params_lora_merged = replace_param_subset(lora_params, full_params)
            merged_params = replace_param_subset(const_params, params_lora_merged)
            return model.apply(merged_params, *args, **kwargs)
        else:
            return model.apply(const_params, *args, **kwargs)

    this_model = partial(wrapped_model_apply, model=model)
    def model_applier(lora_params_const_params, input_ids, cache,
                      model_apply):
        tup = get_attention_mask_and_positions(input_ids, -100)
        tok_positions, attn_mask = tup
        ret = model_apply(lora_params_const_params, input_ids, tok_positions,
                          cache, attn_mask)
        return ret

    assert params_lora_filt is not None

    return (params_lora_filt, fixed_params), partial(model_applier, model_apply=this_model)


def write_params():
    def to_cpu(x, bfloat16=False):
        # x = jax.device_put(x, jax.devices('cpu')[0])
        print('>> transferred to cpu a tensor with shape', x.shape)
        x = dlpack_blocking_gpu2cpu(x)
        if bfloat16:
            print('to float32')
            x = (x.astype(np.float32))
            print('to torch')
            as_torch = ch.from_numpy(x)
            print('to bfloat16')
            return as_torch.to(dtype=ch.bfloat16)
        else:
            print('to float32')
            x = (x.astype(np.float32))
            print('to torch')
            as_torch = ch.from_numpy(x)
            return as_torch

    ret = get_model(jnp.float32, 128, 0, '2b', 'b', 1.0, lora_multiple=512)
    lora_params, fixed_params = ret[0]

    ch.save(jax.tree.map(to_cpu, lora_params), '/mnt/xfs/home/engstrom/store/gemma2b_params/lora_params_final.torch')
    ch.save(jax.tree.map(partial(to_cpu, bfloat16=True), fixed_params), '/mnt/xfs/home/engstrom/store/gemma2b_params/fixed_params_final.torch')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    input_ids = tokenizer.encode("Hello, my dog is super cute", return_tensors='np')



if __name__ == '__main__':
    write_params()