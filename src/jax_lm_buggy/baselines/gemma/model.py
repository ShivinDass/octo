# Copyright 2024 Google LLC
# FROM: https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model implementation."""

import json
import gc
import os
import torch
import torch as ch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union

from . import config as gemma_config
from . import tokenizer

class Sampler(nn.Module):
    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)

        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.scale = nn.Parameter(torch.zeros(dim))
        assert add_unit_offset

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.scale.float())
        else:
            output = output * self.scale.float()
        return output.type_as(x)


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        features = hidden_size
        hidden_dim = intermediate_size
        # self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        # self.up_proj = Linear(hidden_size, intermediate_size, quant)
        # self.down_proj = Linear(intermediate_size, hidden_size, quant)
        self.gating_einsum = ch.nn.Parameter(torch.zeros((2, features, hidden_dim)))
        self.linear = ch.nn.Parameter(torch.zeros((hidden_dim, features)))

    def forward(self, x):
        # gate = self.gate_proj(x)
        # gate = F.gelu(gate, approximate="tanh")
        # up = self.up_proj(x)
        # fuse = gate * up
        # outputs = self.down_proj(fuse)
        # return outputs
        ff_gate = x @ self.gating_einsum[0]
        gate_value = F.gelu(ff_gate, approximate='tanh')
        ff1 = x @ self.gating_einsum[1]
        activations = gate_value * ff1
        outputs = activations @ self.linear
        return outputs

def convert_to_linear_einsums(module):
    child_modules = list(module.named_children())
    for name, mod in child_modules:
        if mod != module:
            if isinstance(mod, Einsum):
                new_mod = LinearEinsum.from_einsum(mod)
                assert hasattr(module, name), (module, name)
                delattr(module, name)
                setattr(module, name, new_mod)
            else:
                convert_to_linear_einsums(mod)

class LinearEinsum(nn.Module):
    @classmethod
    def from_einsum(cls, einsum_instance):
        w = einsum_instance.w
        multiple = einsum_instance.multiple
        eqn = einsum_instance.eqn
        return cls(w, multiple, eqn, einsum_instance)

    def __init__(self, initial_parameter, multiple, eqn, orig_guy):
        super().__init__()
        self.multiple = multiple
        self.orig_guy = orig_guy
        self.eqn = eqn

        # CASE A
        if eqn in ['BTR,RD->BTD', 'BTR,CRD->CBTD']:
            # convert to CRD form
            should_squeeze_C = False
            if eqn == 'BTR,RD->BTD':
                initial_parameter = initial_parameter.unsqueeze(0)
                should_squeeze_C = True

            # assume BTR,CRD->CBTD
            C, R, D = initial_parameter.shape
            self.linear = nn.Linear(R, C * D, bias=False, device='cuda')
            # go from CRD to RCD to (R, C * D)
            reviewed_param = initial_parameter.permute(1, 0, 2).reshape(R, -1)
            assert self.linear.weight.shape == reviewed_param.t().shape, (self.linear.weight.shape, reviewed_param.shape)
            self.linear.weight.data = reviewed_param.t().clone()

            def fwd(x):
                B, T, R = x.shape
                out = self.linear(x)
                # out shape: (B, T, C * D)
                # permute to: (C, B, T, D)
                out = out.view(B, T, C, D).permute(2, 0, 1, 3)

                if should_squeeze_C:
                    out = out.squeeze(0)

                return out

        elif eqn in ['BTNH,NHD->BTD']:
            N, H, D = initial_parameter.shape
            self.linear = nn.Linear(N * H, D, bias=False, device='cuda')
            reviewed_param = initial_parameter.view(N * H, D)
            assert self.linear.weight.shape == reviewed_param.t().shape, (self.linear.weight.shape, reviewed_param.shape)
            self.linear.weight.data = reviewed_param.t().clone()

            def fwd(x):
                B, T, N, H = x.shape
                # res_ein = ch.einsum('BTNH,NHD->BTD', x, initial_parameter.float())
                x = x.reshape(B, T, N * H)
                res = self.linear(x)
                assert res.shape == (B, T, D)
                # print('>> Passed', ch.max(ch.absolute(res - res_ein)))
                # import pdb; pdb.set_trace()
                return res

        elif eqn in ['BTR,NRH->BTNH', 'BTR,CNRH->CBTNH']:
            # convert to CNRH form
            should_squeeze_C = False
            if eqn == 'BTR,NRH->BTNH':
                initial_parameter = initial_parameter.unsqueeze(0)
                should_squeeze_C = True

            # assume BTR,CNRH->CBTNH
            C, N, R, H = initial_parameter.shape
            # put in R, C * N * H form
            initial_parameter_reviewed = initial_parameter.permute(2, 0, 1, 3).reshape(R, -1)
            self.linear = nn.Linear(R, C * N * H, bias=False, device='cuda')
            assert self.linear.weight.shape == initial_parameter_reviewed.t().shape, (self.linear.weight.shape, initial_parameter_reviewed.shape)
            self.linear.weight.data = initial_parameter_reviewed.t().clone()

            def fwd(x):
                B, T, R = x.shape
                out = self.linear(x)
                # convert to CBTNH form
                out = out.view(B, T, C, N, H)
                out = out.permute(2, 0, 1, 3, 4)

                if should_squeeze_C:
                    out = out.squeeze(0)

                return out
        elif eqn == 'CBTR,CNRH->CBTNH':
            C, N, R, H = initial_parameter.shape
            linears = []
            for i in range(C):
                linear = nn.Linear(R, N * H, bias=False, device='cuda')
                # permute from NRH to RNH
                this_parameter = initial_parameter[i].permute(1, 0, 2)
                linear.weight.data = this_parameter.reshape(R, -1).t().clone()
                self.register_module(f'linear_{i}', linear)
                linears.append(linear)

            def fwd(x):
                C, B, T, R = x.shape
                outs = []
                for x_btr, linear in zip(x, linears):
                    out = linear(x_btr)
                    out = out.view(B, T, N, H)
                    outs.append(out)

                out = torch.stack(outs, dim=0)
                assert out.shape == (C, B, T, N, H)
                return out

        else:
            raise ValueError(f'Unknown eqn: {eqn}')

        self.fwd = fwd

    def forward(self, x):
        def do_it(x):
            if self.multiple is not None:
                x = x * self.multiple

            x = self.fwd(x)
            return x

        y1 = do_it(x)
        # y2 = self.orig_guy(x)
        # assert torch.allclose(y1, y2), (ch.max(ch.absolute(y1 - y2)), self.eqn)
        # print('>> Passed', self.eqn)
        return y1


class Einsum(nn.Module):
    def __init__(self, shape, multiple, eqn):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(shape))
        self.multiple = multiple
        self.eqn = eqn

    def forward(self, x):
        w = self.w
        if self.multiple is not None:
            w = w * self.multiple

        res = torch.einsum(self.eqn, x, w)
        return res

class GemmaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_logit_softcapping: Optional[float],
        query_pre_attn_scalar: Optional[int],
        head_dim: int,
        quant: bool,
        attn_type: gemma_config.AttentionType,
        sliding_window_size: Optional[int] = None,
        lora_dim=None,
        lora_multiple=None,
    ):
        super().__init__()
        assert lora_dim is not None and lora_multiple is not None, (lora_dim, lora_multiple)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        # self.qkv_proj = Linear(
        #     self.hidden_size,
        #     (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
        #     quant=quant)
        # self.o_proj = Linear(
        #     self.num_heads * self.head_dim,
        #     self.hidden_size,
        #     quant=quant)
        self.attn_vec_einsum = Einsum((self.num_heads, self.head_dim, self.hidden_size), None, 'BTNH,NHD->BTD')
        self.q_einsum = Einsum((self.num_heads, self.hidden_size, self.head_dim), None, 'BTR,NRH->BTNH')
        self.kv_einsum = Einsum((2, self.num_kv_heads, self.hidden_size, self.head_dim), None, 'BTR,CNRH->CBTNH')
        self.lora_dim = lora_dim
        # self.lora_multiple = lora_multiple

        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

        self.q_lora_a = Einsum((self.hidden_size, lora_dim), None, 'BTR,RD->BTD')
        self.q_lora_b = Einsum((self.num_heads, lora_dim, self.head_dim),
                               lora_multiple, 'BTR,NRH->BTNH')
        self.kv_lora_a = Einsum((2, self.hidden_size, lora_dim), None, 'BTR,CRD->CBTD')
        self.kv_lora_b = Einsum((2, self.num_kv_heads, self.lora_dim, self.head_dim),
                                lora_multiple, 'CBTR,CNRH->CBTNH')
        self.o_lora_a = Einsum((self.num_heads, self.head_dim, lora_dim), None, 'BTNH,NHD->BTD')
        self.o_lora_b = Einsum((self.lora_dim, self.hidden_size), lora_multiple, 'BTR,RD->BTD')

        assert self.num_heads != self.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        def qkv_from_hidden(hidden_states):
            # query_proj = self.q_einsum('BTD,NDH->BTNH', hidden_states)
            query_proj = self.q_einsum(hidden_states) # (C)
            # kv_proj = self.kv_einsum('BSD,CKDH->CBSKH', hidden_states)
            kv_proj = self.kv_einsum(hidden_states) # (C)

            reduced_x_q = self.q_lora_a(hidden_states) # (A)
            query_proj += self.q_lora_b(reduced_x_q)/self.lora_dim # (C)

            # reduced_x_kv = self.kv_lora_a('BSD,CDR->CBSR', hidden_states)
            reduced_x_kv = self.kv_lora_a(hidden_states) # (A)
            kv_proj += self.kv_lora_b(reduced_x_kv)/self.lora_dim # (C)
            key_proj, value_proj = kv_proj
            return query_proj, key_proj, value_proj

        xq, xk, xv = qkv_from_hidden(hidden_states)
        expected_xq = (batch_size, input_len, self.num_heads, self.head_dim)
        assert xq.shape == expected_xq, (xq.shape, expected_xq)

        expected_xk = (batch_size, input_len, self.num_kv_heads, self.head_dim)
        assert xk.shape == expected_xk, (xk.shape, expected_xk)

        expected_xv = (batch_size, input_len, self.num_kv_heads, self.head_dim)
        assert xv.shape == expected_xv, (xv.shape, expected_xv)

        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        key = xk
        value = xv
        q = xq

        if self.num_kv_heads != self.num_heads:
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        assert self.scaling == self.head_dim**-0.5
        q.mul_(self.scaling)

        encoded = do_attn(q, key, value, mask)
        attn_output = self.attn_vec_einsum(encoded) # (B)
        encoded_downscale = self.o_lora_a(encoded) # (B)
        encoded_lora = self.o_lora_b(encoded_downscale) # (A)
        attn_output += encoded_lora / self.lora_dim
        return attn_output

K_MASK = -2.3819763e38  # Set to a large negative number.
def do_attn(q, k, v, attn_mask):
    logits = ch.einsum('BTNH,BSNH->BTNS', q, k)
    if attn_mask is not None:
        padded_logits = ch.where(
            (ch.unsqueeze(attn_mask, -2)), logits, K_MASK
        )
    else:
        padded_logits = logits

    probs = ch.nn.functional.softmax(padded_logits, dim=-1)
    encoded = ch.einsum('BTNS,BSNH->BTNH', probs, v)
    return encoded


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=gemma_config.AttentionType.GLOBAL,
            lora_dim=config.lora_dim,
            lora_multiple=config.lora_multiple
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.pre_attention_norm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ffw_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.pre_attention_norm(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.pre_ffw_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        raise NotImplementedError('Gemma2DecoderLayer is not implemented.')
        super().__init__()
        self.attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=attn_type,
            sliding_window_size=config.sliding_window_size,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(config))
            elif config.architecture == gemma_config.Architecture.GEMMA_2:
                attn_type = (
                    config.attn_types[i]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        # kv_write_indices: torch.Tensor,
        # kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                # kv_write_indices=kv_write_indices,
                # kv_cache=kv_caches[i],
                mask=mask,
            )

            assert hidden_states.dtype == ch.bfloat16

        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config)
        self.sampler = Sampler(vocab_size, config)

        # Pre-compute rotary embedding table.
        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = precompute_freqs_cis(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta)
        self.register_buffer('freqs_cis', freqs_cis)
        self.pad_id = ch.tensor(self.tokenizer.pad_id).to(device='cuda')

    def forward(self, input_token_ids: torch.Tensor, y):
        from .demo import get_attention_mask_and_positions
        assert len(input_token_ids.shape) == 2, input_token_ids.shape
        input_positions, mask = get_attention_mask_and_positions(input_token_ids,
                                                                 self.pad_id)

        freqs_cis_flat = self.freqs_cis.index_select(0, input_positions.view(-1))
        freqs_cis = freqs_cis_flat.view(*input_positions.shape, -1)

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5,
                                  dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            # kv_write_indices=kv_write_indices,
            # kv_caches=kv_caches,
            mask=mask,
        )

        assert not self.embedder.quant

        embedder_weight = self.embedder.weight
        logits = torch.matmul(hidden_states, embedder_weight.t())
        # add an extra slot to match the wikitext model
        return logits, None

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results

    def load_weights(self, model_path: str):
        if os.path.isfile(model_path):
            self.load_state_dict(
                torch.load(
                    model_path, mmap=True, weights_only=True,
                )['model_state_dict'],
                strict=False,
            )
        else:
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict
                gc.collect()
