DeepSeekV3Config from target_model (https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py)
Reference model from llama3_eagle.py (https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/llama3_eagle.py)

import logging
import mathp

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, DeepSeekV3Config, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import DeepSeekV3Config

from ..utils import padding
from .base import Eagle3DraftModel

logger = logging.getLogger(__name__)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class DeepseekV3RotaryEmbedding(nn.Module):
   def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
       super().__init__()



       self.dim = dim
       self.max_position_embeddings = max_position_embeddings
       self.base = base
       inv_freq = 1.0 / (
           self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
       )
       self.register_buffer("inv_freq", inv_freq, persistent=False)



       # Build here to make `torch.jit.trace` work.
       self._set_cos_sin_cache(
           seq_len=max_position_embeddings,
           device=self.inv_freq.device,
           dtype=torch.get_default_dtype(),
       )
       self.max_seq_len_cached = None



   def _set_cos_sin_cache(self, seq_len, device, dtype):
       self.max_seq_len_cached = seq_len
       t = torch.arange(
           self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
       )



       freqs = torch.outer(t, self.inv_freq.to(t.device))
       # Different from paper, but it uses a different permutation in order to obtain the same calculation
       emb = torch.cat((freqs, freqs), dim=-1)
       self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
       self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)



   def forward(self, x, seq_len=None):
       # x: [bs, num_attention_heads, seq_len, head_size]
       if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
           self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)



       return (
           self.cos_cached[:seq_len].to(dtype=x.dtype),
           self.sin_cached[:seq_len].to(dtype=x.dtype),
       )




class DeepseekV3LinearScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
   """DeepseekV3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""



   def __init__(
       self,
       dim,
       max_position_embeddings=2048,
       base=10000,
       device=None,
       scaling_factor=1.0,
   ):
       self.scaling_factor = scaling_factor
       super().__init__(dim, max_position_embeddings, base, device)



   def _set_cos_sin_cache(self, seq_len, device, dtype):
       self.max_seq_len_cached = seq_len
       t = torch.arange(
           self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
       )
       t = t / self.scaling_factor



       freqs = torch.outer(t, self.inv_freq)
       # Different from paper, but it uses a different permutation in order to obtain the same calculation
       emb = torch.cat((freqs, freqs), dim=-1)
       self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
       self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)











class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
   """DeepseekV3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""



   def __init__(
       self,
       dim,
       max_position_embeddings=2048,
       base=10000,
       device=None,
       scaling_factor=1.0,
   ):
       self.scaling_factor = scaling_factor
       super().__init__(dim, max_position_embeddings, base, device)



   def _set_cos_sin_cache(self, seq_len, device, dtype):
       self.max_seq_len_cached = seq_len



       if seq_len > self.max_position_embeddings:
           base = self.base * (
               (self.scaling_factor * seq_len / self.max_position_embeddings)
               - (self.scaling_factor - 1)
           ) ** (self.dim / (self.dim - 2))
           inv_freq = 1.0 / (
               base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
           )
           self.register_buffer("inv_freq", inv_freq, persistent=False)



       t = torch.arange(
           self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
       )



       freqs = torch.outer(t, self.inv_freq)
       # Different from paper, but it uses a different permutation in order to obtain the same calculation
       emb = torch.cat((freqs, freqs), dim=-1)
       self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
       self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)










# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
   num_rotations, dim, base=10000, max_position_embeddings=2048
):
   return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
       2 * math.log(base)
   )






# Find dim range bounds based on rotations
def yarn_find_correction_range(
   low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
   low = math.floor(
       yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
   )
   high = math.ceil(
       yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
   )
   return max(low, 0), min(high, dim - 1)  # Clamp values just in case






def yarn_get_mscale(scale=1, mscale=1):
   if scale <= 1:
       return 1.0
   return 0.1 * mscale * math.log(scale) + 1.0






def yarn_linear_ramp_mask(min, max, dim):
   if min == max:
       max += 0.001  # Prevent singularity



   linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
   ramp_func = torch.clamp(linear_func, 0, 1)
   return ramp_func






class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):



   def __init__(
       self,
       dim,
       max_position_embeddings=2048,
       base=10000,
       device=None,
       scaling_factor=1.0,
       original_max_position_embeddings=4096,
       beta_fast=32,
       beta_slow=1,
       mscale=1,
       mscale_all_dim=0,
   ):
       self.scaling_factor = scaling_factor
       self.original_max_position_embeddings = original_max_position_embeddings
       self.beta_fast = beta_fast
       self.beta_slow = beta_slow
       self.mscale = mscale
       self.mscale_all_dim = mscale_all_dim
       super().__init__(dim, max_position_embeddings, base, device)



   def _set_cos_sin_cache(self, seq_len, device, dtype):
       self.max_seq_len_cached = seq_len
       dim = self.dim



       freq_extra = 1.0 / (
           self.base
           ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
       )
       freq_inter = 1.0 / (
           self.scaling_factor
           * self.base
           ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
       )



       low, high = yarn_find_correction_range(
           self.beta_fast,
           self.beta_slow,
           dim,
           self.base,
           self.original_max_position_embeddings,
       )
       inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
           device=device, dtype=torch.float32
       )
       inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
       self.register_buffer("inv_freq", inv_freq, persistent=False)



       t = torch.arange(seq_len, device=device, dtype=torch.float32)



       freqs = torch.outer(t, inv_freq)



       _mscale = float(
           yarn_get_mscale(self.scaling_factor, self.mscale)
           / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
       )



       emb = torch.cat((freqs, freqs), dim=-1)
       self.register_buffer(
           "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
       )
       self.register_buffer(
           "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
       )






# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
   """Rotates half the hidden dims of the input."""
   x1 = x[..., : x.shape[-1] // 2]
   x2 = x[..., x.shape[-1] // 2 :]
   return torch.cat((-x2, x1), dim=-1)






# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def  apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
   """Applies Rotary Position Embedding to the query and key tensors.


   Args:
       q (`torch.Tensor`): The query tensor.
       k (`torch.Tensor`): The key tensor.
       cos (`torch.Tensor`): The cosine part of the rotary embedding.
       sin (`torch.Tensor`): The sine part of the rotary embedding.
       position_ids (`torch.Tensor`):
           The position indices of the tokens corresponding to the query and key tensors. For example, this can be
           used to pass offsetted position ids when working with a KV-cache.
       unsqueeze_dim (`int`, *optional*, defaults to 1):
           The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
           sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
           that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
           k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
           cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
           the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
   Returns:
       `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
   """
   cos = cos[position_ids].unsqueeze(unsqueeze_dim)
   sin = sin[position_ids].unsqueeze(unsqueeze_dim)



   b, h, s, d = q.shape
   q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)



   b, h, s, d = k.shape
   k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)



   q_embed = (q * cos) + (rotate_half(q) * sin)
   k_embed = (k * cos) + (rotate_half(k) * sin)
   return q_embed, k_embed













class DeepseekV3Attention(nn.Module):
   """Multi-headed attention from 'Attention Is All You Need' paper"""



   def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
       super().__init__()
       self.config = config
       self.layer_idx = layer_idx
       if layer_idx is None:
           logger.warning_once(
               f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
               "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
               "when creating this class."
           )



       self.attention_dropout = config.attention_dropout
       self.hidden_size = config.hidden_size
       self.num_heads = config.num_attention_heads



       self.max_position_embeddings = config.max_position_embeddings
       self.rope_theta = config.rope_theta
       self.q_lora_rank = config.q_lora_rank
       self.qk_rope_head_dim = config.qk_rope_head_dim
       self.kv_lora_rank = config.kv_lora_rank
       self.v_head_dim = config.v_head_dim
       self.qk_nope_head_dim = config.qk_nope_head_dim
       self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim



       self.is_causal = True



       if self.q_lora_rank is None:
           self.q_proj = nn.Linear(
               self.hidden_size * 2, self.num_heads * self.q_head_dim, bias=False
           )
       else:
           self.q_a_proj = nn.Linear(
               self.hidden_size *2 , config.q_lora_rank, bias=config.attention_bias
           )
           self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
           self.q_b_proj = nn.Linear(
               config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
           )



       self.kv_a_proj_with_mqa = nn.Linear(
           self.hidden_size * 2,
           config.kv_lora_rank + config.qk_rope_head_dim,
           bias=config.attention_bias,
       )
       self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
       self.kv_b_proj = nn.Linear(
           config.kv_lora_rank,
           self.num_heads
           * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
           bias=False,
       )



       self.o_proj = nn.Linear(
           self.num_heads * self.v_head_dim,
           self.hidden_size,
           bias=config.attention_bias,
       )
       self._init_rope()



       self.softmax_scale = self.q_head_dim ** (-0.5)
       if self.config.rope_scaling is not None:
           mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
           scaling_factor = self.config.rope_scaling["factor"]
           if mscale_all_dim:
               mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
               self.softmax_scale = self.softmax_scale * mscale * mscale



   def _init_rope(self):
       if self.config.rope_scaling is None:
           self.rotary_emb = DeepseekV3RotaryEmbedding(
               self.qk_rope_head_dim,
               max_position_embeddings=self.max_position_embeddings,
               base=self.rope_theta,
           )
       else:
           scaling_type = self.config.rope_scaling["type"]
           scaling_factor = self.config.rope_scaling["factor"]
           if scaling_type == "linear":
               self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                   self.qk_rope_head_dim,
                   max_position_embeddings=self.max_position_embeddings,
                   scaling_factor=scaling_factor,
                   base=self.rope_theta,
               )
           elif scaling_type == "dynamic":
               self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                   self.qk_rope_head_dim,
                   max_position_embeddings=self.max_position_embeddings,
                   scaling_factor=scaling_factor,
                   base=self.rope_theta,
               )
           elif scaling_type == "yarn":
               kwargs = {
                   key: self.config.rope_scaling[key]
                   for key in [
                       "original_max_position_embeddings",
                       "beta_fast",
                       "beta_slow",
                       "mscale",
                       "mscale_all_dim",
                   ]
                   if key in self.config.rope_scaling
               }
               self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                   self.qk_rope_head_dim,
                   max_position_embeddings=self.max_position_embeddings,
                   scaling_factor=scaling_factor,
                   base=self.rope_theta,
                   **kwargs,
               )
           else:
               raise ValueError(f"Unknown RoPE scaling type {scaling_type}")



   def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
       return (
           tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
           .transpose(1, 2)
           .contiguous()
       )



   def forward(
       self,
       hidden_states: torch.Tensor,
       attention_mask: Optional[torch.Tensor] = None,
       position_ids: Optional[torch.LongTensor] = None,
       past_key_value: Optional[Cache] = None,
       output_attentions: bool = False,
       use_cache: bool = False,
       **kwargs,
   ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
       if "padding_mask" in kwargs:
           warnings.warn(
               "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
           )
       bsz, q_len, _ = hidden_states.size()



       if self.q_lora_rank is None:
           q = self.q_proj(hidden_states)
       else:
           q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
       q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
       q_nope, q_pe = torch.split(
           q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
       )



       compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
       compressed_kv, k_pe = torch.split(
           compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
       )
       k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
       kv = (
           self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
           .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
           .transpose(1, 2)
       )



       k_nope, value_states = torch.split(
           kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
       )
       kv_seq_len = value_states.shape[-2]
       if past_key_value is not None:
           if self.layer_idx is None:
               raise ValueError(
                   f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                   "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                   "with a layer index."
               )
           kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
       cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
       past_len = kv_seq_len - q_len
       pos_ids  = position_ids if past_len == 0 else (position_ids + past_len)





       q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, pos_ids)



       query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
       query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
       query_states[:, :, :, self.qk_nope_head_dim :] = q_pe



       key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
       key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
       key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
       if past_key_value is not None:
           cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
           key_states, value_states = past_key_value.update(
               key_states, value_states, self.layer_idx, cache_kwargs
           )



       attn_weights = (
           torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
       )



       if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
           raise ValueError(
               f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
               f" {attn_weights.size()}"
           )
       assert attention_mask is not None
       if attention_mask is not None:
           if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
               raise ValueError(
                   f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
               )
           attn_weights = attn_weights + attention_mask



       # upcast attention to fp32
       attn_weights = nn.functional.softmax(
           attn_weights, dim=-1, dtype=torch.float32
       ).to(query_states.dtype)
       attn_weights = nn.functional.dropout(
           attn_weights, p=self.attention_dropout, training=self.training
       )
       attn_output = torch.matmul(attn_weights, value_states)



       if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
           raise ValueError(
               f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
               f" {attn_output.size()}"
           )



       attn_output = attn_output.transpose(1, 2).contiguous()



       attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)



       attn_output = self.o_proj(attn_output)



       if not output_attentions:
           attn_weights = None



       return attn_output, attn_weights, past_key_value






class DeepseekV3MLP(nn.Module):
   def __init__(self, config, hidden_size=None, intermediate_size=None):
       super().__init__()
       self.config = config
       self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
       self.intermediate_size = (
           config.intermediate_size if intermediate_size is None else intermediate_size
       )



       self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
       self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
       self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
       self.act_fn = ACT2FN[config.hidden_act]



   def forward(self, x):
       down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
       return down_proj







class DeepseekV3RMSNorm(nn.Module):
   def __init__(self, hidden_size, eps=1e-6):
       """
       DeepseekV3RMSNorm is equivalent to T5LayerNorm
       """
       super().__init__()
       self.weight = nn.Parameter(torch.ones(hidden_size))
       self.variance_epsilon = eps



   def forward(self, hidden_states):
       input_dtype = hidden_states.dtype
       hidden_states = hidden_states.to(torch.float32)
       variance = hidden_states.pow(2).mean(-1, keepdim=True)
       hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
       return self.weight * hidden_states.to(input_dtype)












class MoEGate(nn.Module):
   def __init__(self, config):
       super().__init__()
       self.config = config
       self.top_k = config.num_experts_per_tok
       self.n_routed_experts = config.n_routed_experts
       self.routed_scaling_factor = config.routed_scaling_factor
       self.scoring_func = config.scoring_func
       self.topk_method = config.topk_method
       self.n_group = config.n_group
       self.topk_group = config.topk_group



       # topk selection algorithm
       self.norm_topk_prob = config.norm_topk_prob
       self.gating_dim = config.hidden_size
       self.weight = nn.Parameter(
           torch.empty((self.n_routed_experts, self.gating_dim))
       )
       if self.topk_method == "noaux_tc":
           self.e_score_correction_bias = nn.Parameter(
               torch.empty((self.n_routed_experts))
           )
       self.reset_parameters()



   def reset_parameters(self) -> None:
       import torch.nn.init as init



       init.kaiming_uniform_(self.weight, a=math.sqrt(5))



   def forward(self, hidden_states):
       bsz, seq_len, h = hidden_states.shape
       ### compute gating score
       hidden_states = hidden_states.view(-1, h)
       logits = F.linear(
           hidden_states.type(torch.float32), self.weight.type(torch.float32), None
       )
       if self.scoring_func == "sigmoid":
           scores = logits.sigmoid()
       else:
           raise NotImplementedError(
               f"insupportable scoring function for MoE gating: {self.scoring_func}"
           )



       ### select top-k experts
       if self.topk_method == "noaux_tc":
           assert not self.training
           scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
           group_scores = (
               scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
           )  # [n, n_group]
           group_idx = torch.topk(
               group_scores, k=self.topk_group, dim=-1, sorted=False
           )[
               1
           ]  # [n, top_k_group]
           group_mask = torch.zeros_like(group_scores)  # [n, n_group]
           group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
           score_mask = (
               group_mask.unsqueeze(-1)
               .expand(
                   bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
               )
               .reshape(bsz * seq_len, -1)
           )  # [n, e]
           tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
           _, topk_idx = torch.topk(
               tmp_scores, k=self.top_k, dim=-1, sorted=False
           )
           topk_weight = scores.gather(1, topk_idx)
       else:
           raise NotImplementedError(
               f"insupportable TopK function for MoE gating: {self.topk_method}"
           )



       ### norm gate to sum 1
       if self.top_k > 1 and self.norm_topk_prob:
           denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
           topk_weight = topk_weight / denominator
       topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor



       return topk_idx, topk_weight



class DeepseekV3MoE(nn.Module):
   """
   A mixed expert module containing shared experts.
   """



   def __init__(self, config):
       super().__init__()
       self.config = config
       self.num_experts_per_tok = config.num_experts_per_tok



       if hasattr(config, "ep_size") and config.ep_size > 1:
           assert config.ep_size == dist.get_world_size()
           self.ep_size = config.ep_size
           self.experts_per_rank = config.n_routed_experts // config.ep_size
           self.ep_rank = dist.get_rank()
           self.experts = nn.ModuleList(
               [
                   (
                       DeepseekV3MLP(
                           config, intermediate_size=config.moe_intermediate_size
                       )
                       if i >= self.ep_rank * self.experts_per_rank
                       and i < (self.ep_rank + 1) * self.experts_per_rank
                       else None
                   )
                   for i in range(config.n_routed_experts)
               ]
           )
       else:
           self.ep_size = 1
           self.experts_per_rank = config.n_routed_experts
           self.ep_rank = 0
           self.experts = nn.ModuleList(
               [
                   DeepseekV3MLP(
                       config, intermediate_size=config.moe_intermediate_size
                   )
                   for i in range(config.n_routed_experts)
               ]
           )
       self.gate = MoEGate(config)
       if config.n_shared_experts is not None:
           intermediate_size = config.moe_intermediate_size * config.n_shared_experts
           self.shared_experts = DeepseekV3MLP(
               config=config, intermediate_size=intermediate_size
           )



   def forward(self, hidden_states):
       identity = hidden_states
       orig_shape = hidden_states.shape
       topk_idx, topk_weight = self.gate(hidden_states)
       hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
       flat_topk_idx = topk_idx.view(-1)
       if not self.training:
           y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
       if self.config.n_shared_experts is not None:
           y = y + self.shared_experts(identity)
       return y



   @torch.no_grad()
   def moe_infer(self, x, topk_ids, topk_weight):
       cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
       cnts.scatter_(1, topk_ids, 1)
       tokens_per_expert = cnts.sum(dim=0)
       idxs = topk_ids.view(-1).argsort()
       sorted_tokens = x[idxs // topk_ids.shape[1]]
       sorted_tokens_shape = sorted_tokens.shape
       if self.ep_size > 1:
           tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
           tokens_per_expert_group = tokens_per_expert.new_empty(
               tokens_per_expert.shape[0]
           )
           dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
           output_splits = (
               tokens_per_expert_group.view(self.ep_size, -1)
               .sum(1)
               .cpu()
               .numpy()
               .tolist()
           )
           gathered_tokens = sorted_tokens.new_empty(
               tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
           )
           input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
           dist.all_to_all(
               list(gathered_tokens.split(output_splits)),
               list(sorted_tokens.split(input_split_sizes)),
           )
           tokens_per_expert_post_gather = tokens_per_expert_group.view(
               self.ep_size, self.experts_per_rank
           ).sum(dim=0)
           gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
           s = 0
           for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
               gatherd_idxs[s : s + k] = i % self.experts_per_rank
               s += k
           gatherd_idxs = gatherd_idxs.argsort()
           sorted_tokens = gathered_tokens[gatherd_idxs]
           tokens_per_expert = tokens_per_expert_post_gather
       tokens_per_expert = tokens_per_expert.cpu().numpy()



       outputs = []
       start_idx = 0
       for i, num_tokens in enumerate(tokens_per_expert):
           end_idx = start_idx + num_tokens
           if num_tokens == 0:
               continue
           expert = self.experts[i + self.ep_rank * self.experts_per_rank]
           tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
           expert_out = expert(tokens_for_this_expert)
           outputs.append(expert_out)
           start_idx = end_idx



       outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
       if self.ep_size > 1:
           new_x = torch.empty_like(outs)
           new_x[gatherd_idxs] = outs
           gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
           dist.all_to_all(
               list(gathered_tokens.split(input_split_sizes)),
               list(new_x.split(output_splits)),
           )
           outs = gathered_tokens



       new_x = torch.empty_like(outs)
       new_x[idxs] = outs
       final_out = (
           new_x.view(*topk_ids.shape, -1)
           .type(topk_weight.dtype)
           .mul_(topk_weight.unsqueeze(dim=-1))
           .sum(dim=1)
           .type(new_x.dtype)
       )
       return final_out



class DeepseekV3MoE(nn.Module):
   """
   A mixed expert module containing shared experts.
   """



   def __init__(self, config):
       super().__init__()
       self.config = config
       self.num_experts_per_tok = config.num_experts_per_tok



       if hasattr(config, "ep_size") and config.ep_size > 1:
           assert config.ep_size == dist.get_world_size()
           self.ep_size = config.ep_size
           self.experts_per_rank = config.n_routed_experts // config.ep_size
           self.ep_rank = dist.get_rank()
           self.experts = nn.ModuleList(
               [
                   (
                       DeepseekV3MLP(
                           config, intermediate_size=config.moe_intermediate_size
                       )
                       if i >= self.ep_rank * self.experts_per_rank
                       and i < (self.ep_rank + 1) * self.experts_per_rank
                       else None
                   )
                   for i in range(config.n_routed_experts)
               ]
           )
       else:
           self.ep_size = 1
           self.experts_per_rank = config.n_routed_experts
           self.ep_rank = 0
           self.experts = nn.ModuleList(
               [
                   DeepseekV3MLP(
                       config, intermediate_size=config.moe_intermediate_size
                   )
                   for i in range(config.n_routed_experts)
               ]
           )
       self.gate = MoEGate(config)
       if config.n_shared_experts is not None:
           intermediate_size = config.moe_intermediate_size * config.n_shared_experts
           self.shared_experts = DeepseekV3MLP(
               config=config, intermediate_size=intermediate_size
           )



   def forward(self, hidden_states):
       identity = hidden_states
       orig_shape = hidden_states.shape
       topk_idx, topk_weight = self.gate(hidden_states)
       hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
       flat_topk_idx = topk_idx.view(-1)
       if not self.training:
           y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
       if self.config.n_shared_experts is not None:
           y = y + self.shared_experts(identity)
       return y



   @torch.no_grad()
   def moe_infer(self, x, topk_ids, topk_weight):
       cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
       cnts.scatter_(1, topk_ids, 1)
       tokens_per_expert = cnts.sum(dim=0)
       idxs = topk_ids.view(-1).argsort()
       sorted_tokens = x[idxs // topk_ids.shape[1]]
       sorted_tokens_shape = sorted_tokens.shape
       if self.ep_size > 1:
           tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
           tokens_per_expert_group = tokens_per_expert.new_empty(
               tokens_per_expert.shape[0]
           )
           dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
           output_splits = (
               tokens_per_expert_group.view(self.ep_size, -1)
               .sum(1)
               .cpu()
               .numpy()
               .tolist()
           )
           gathered_tokens = sorted_tokens.new_empty(
               tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
           )
           input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
           dist.all_to_all(
               list(gathered_tokens.split(output_splits)),
               list(sorted_tokens.split(input_split_sizes)),
           )
           tokens_per_expert_post_gather = tokens_per_expert_group.view(
               self.ep_size, self.experts_per_rank
           ).sum(dim=0)
           gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
           s = 0
           for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
               gatherd_idxs[s : s + k] = i % self.experts_per_rank
               s += k
           gatherd_idxs = gatherd_idxs.argsort()
           sorted_tokens = gathered_tokens[gatherd_idxs]
           tokens_per_expert = tokens_per_expert_post_gather
       tokens_per_expert = tokens_per_expert.cpu().numpy()



       outputs = []
       start_idx = 0
       for i, num_tokens in enumerate(tokens_per_expert):
           end_idx = start_idx + num_tokens
           if num_tokens == 0:
               continue
           expert = self.experts[i + self.ep_rank * self.experts_per_rank]
           tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
           expert_out = expert(tokens_for_this_expert)
           outputs.append(expert_out)
           start_idx = end_idx



       outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
       if self.ep_size > 1:
           new_x = torch.empty_like(outs)
           new_x[gatherd_idxs] = outs
           gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
           dist.all_to_all(
               list(gathered_tokens.split(input_split_sizes)),
               list(new_x.split(output_splits)),
           )
           outs = gathered_tokens



       new_x = torch.empty_like(outs)
       new_x[idxs] = outs
       final_out = (
           new_x.view(*topk_ids.shape, -1)
           .type(topk_weight.dtype)
           .mul_(topk_weight.unsqueeze(dim=-1))
           .sum(dim=1)
           .type(new_x.dtype)
       )
       return final_out


Mixture-of-Experts: Different from the llama_eagle3 model, the MoE classes were included in the draft model so that the draft model’s hidden-state dynamics closely match the target’s, which allows the target to have a higher acceptance of tokens. MoE allows the model to generate different MLPs chosen by the gate to allow better scaling and more capacity without more per-token cost.























class DeepseekV3DecoderLayer(nn.Module):
   def __init__(self, config: DeepseekV3Config, layer_idx: int):
       super().__init__()
       self.hidden_size = config.hidden_size



       self.self_attn = ATTENTION_CLASSES[config._attn_implementation](
           config=config, layer_idx=layer_idx
       )



       self.mlp = (
           DeepseekV3MoE(config)
           if (
               config.n_routed_experts is not None
               and layer_idx >= config.first_k_dense_replace
               and layer_idx % config.moe_layer_freq == 0
           )
           else DeepseekV3MLP(config)
       )
       self.input_layernorm = DeepseekV3RMSNorm(
           config.hidden_size, eps=config.rms_norm_eps
       )
       self.post_attention_layernorm = DeepseekV3RMSNorm(
           config.hidden_size, eps=config.rms_norm_eps
       )



   def forward(
       Self,
       input_emb: torch.Tensor,
       hidden_states: torch.Tensor,
       cache_hidden: List[List[torch.Tensor]] = [],
       attention_mask: Optional[torch.Tensor] = None,
       position_ids: Optional[torch.LongTensor] = None,
       past_key_value: Optional[Tuple[torch.Tensor]] = None,
       output_attentions: Optional[bool] = False,
       use_cache: Optional[bool] = False,
       **kwargs,
   ) -> Tuple[
       torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
   ]:
       """
       Args:
           hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
           attention_mask (`torch.FloatTensor`, *optional*):
               attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
               query_sequence_length, key_sequence_length)` if default attention is used.
           output_attentions (`bool`, *optional*):
               Whether or not to return the attentions tensors of all attention layers. See `attentions` under
               returned tensors for more detail.
           use_cache (`bool`, *optional*):
               If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
               (see `past_key_values`).
           past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
       """
       if "padding_mask" in kwargs:
           warnings.warn(
               "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
           )
       residual = hidden_states



       hidden_states = self.input_layernorm(hidden_states)
       input_emb = self.input_layernorm(input_emb)
       hidden_states = torch.cat((input_emb, hidden_states), dim=-1)


       # Self Attention
       attn_out, _, _  = self.self_attn(
           hidden_states=hidden_states,
           attention_mask=attention_mask,
           position_ids=position_ids,
           cache_hidden=cache_hidden,
           past_key_value=None,
           output_attentions=False,
           use_cache=False,
           **kwargs,
       )
       hidden_states = residual + attn_out



       # Fully Connected
       residual = hidden_states
       hidden_states = self.post_attention_layernorm(hidden_states)
       hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states








class DeepSeekV3ForCausalLMEagle3(Eagle3DraftModel):

    config_class = DeepSeekV3Config

    def __init__(self, config, quant_config=None) -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens =nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer =DeepseekV3DecoderLayer(config, layer_idx=0)

        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = DeepSeekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # create vocab buffers
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        """
        Arguments:
            hidden_states (`torch.FloatTensor`): input to the layer, cat low, mid high hidden_states of shape `(batch, seq_len, hidden_states * 3)`
            input_ids (`torch.LongTensor`): input ids of shape `(batch, seq_len)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor`, *optional*): position ids of shape `(batch, seq_len)`
        """
        if ttt_length == 1:
            logger.info("using ttt_length 1, no need to cache hidden states")
            cache_hidden = None
        else:
            logger.info(f"using ttt_length {ttt_length}, caching hidden states")
            cache_hidden = [[], []]

        batch_size, seq_length, _ = hidden_states.size()

        # make position ids
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # make attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        # fc
        hidden_states = self.fc(hidden_states)
        hidden_states= self.midlayer(
  input_emb=inputs_embeds,
    hidden_states=hidden_states,
    cache_hidden=cache_hidden,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,

)

        # norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # eagle 3 requires hidden states from 3 layers
        assert hidden_states.size(-1) == self.config.hidden_size * 3
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def backbone(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            residual=None,
        )


















     
