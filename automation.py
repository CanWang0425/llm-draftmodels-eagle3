

#This scaffold generator builds on concepts from the SpecForge Eagle3 draft model framework, which provides the underlying architecture for speculative decoding with draft models.
#!/usr/bin/env python3
# gen_draft_scaffold.py
import argparse, importlib, os, textwrap

SCAFFOLD_TEMPLATE = """\
# Auto-generated scaffold for {model_name} Eagle3 draft model
# Structure follows your spec: helpers → rotary → attention (2H + offset) → decoder → draft model.
# All model-specific parts that can’t be inferred are left as callbacks in callbacks_{snake}.py.

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Target model imports (provided by CLI) --------
from {module} import {config_cls} as TargetConfig
from {module} import {attention_cls} as TargetAttention
from {module} import {mlp_cls} as TargetMLP
from {module} import {norm_cls} as TargetRMSNorm{maybe_decoder_import}

# -------- SpecForge base --------
from .base import Eagle3DraftModel

# =======================================================
# 1) Helper functions: masks, head repeat, (rotate/apply_rope if target lacks them)
# =======================================================
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    \"\"\"[bsz, T] -> [bsz, 1, T, T+past] causal mask\"\"\"
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    ar = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(ar < (ar + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    \"\"\"[bsz, S] -> [bsz, 1, T, S] padding mask (additive, -inf on pads)\"\"\"
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inv = 1.0 - expanded
    return inv.masked_fill(inv.to(torch.bool), torch.finfo(dtype).min)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    \"\"\"(b, n_kv, s, d) -> (b, n_kv*n_rep, s, d)\"\"\"
    b, n_kv, s, d = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return x.reshape(b, n_kv * n_rep, s, d)

# Optional: fallbacks if target doesn’t expose rotate_half/apply_rotary_pos_emb.
def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def default_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# =======================================================
# 2) Rotary embedding family — import your target’s classes instead if available
#    These are placeholders you can replace by editing callbacks or swapping imports.
# =======================================================
class SimpleRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv = 1.0 / (self.base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._set_cache(max_position_embeddings, device, torch.get_default_dtype())

    def _set_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)

# =======================================================
# 3) Callbacks: specialize QKV, positional math, MoE routing, etc.
# =======================================================
try:
    from .callbacks_{snake} import (
        eagle_qkv_from_2h,        # (hidden_2h, cfg) -> (q, k, v) in b,h,t,d or None triplet for wrapper mode
        eagle_apply_positions,    # (q, k, position_ids, past_len, rope_obj, apply_rope_fn) -> (q,k,new_pos_ids)
        eagle_after_attention,    # (attn_out) -> attn_out
        make_rotary,              # (cfg, head_dim) -> rope module
    )
except Exception:
    def eagle_qkv_from_2h(hidden_2h, cfg):
        return None, None, None
    def eagle_apply_positions(q, k, position_ids, past_len, rope_obj=None, apply_rope_fn=default_apply_rotary_pos_emb):
        # Default: add cache offset to position_ids if past_len>0
        return q, k, (position_ids if past_len == 0 else (position_ids + past_len))
    def eagle_after_attention(x): return x
    def make_rotary(cfg, head_dim): return SimpleRotaryEmbedding(head_dim, getattr(cfg, "max_position_embeddings", 2048), base=getattr(cfg, "rope_theta", 10000))

# =======================================================
# 4) General attention wrapper (Eagle logic): takes 2H, offsets positions, supports cache width
# =======================================================
class EagleAttentionWrapper(nn.Module):
    def __init__(self, cfg: TargetConfig):
        super().__init__()
        self.cfg = cfg
        # Pre-project 2H → H if we fall back to target attention path (safe default):
        self.pre_proj = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.attn = TargetAttention(cfg)
        # If callbacks choose to build true Q/K/V from 2H, we provide a minimal RoPE helper:
        # NOTE: Real models should use their own RotaryEmbedding classes via callbacks.make_rotary.
        self._rope = None

    def _ensure_rope(self, head_dim, sample_tensor):
        if self._rope is None:
            self._rope = make_rotary(self.cfg, head_dim).to(sample_tensor.device)

    def forward(self, hidden_2h, attention_mask=None, position_ids=None, cache_hidden=None, **kw):
        bsz, q_len, twoH = hidden_2h.shape
        assert twoH == self.cfg.hidden_size * 2, "expected concat(norm(emb), norm(hid)) → 2H"

        # total cached keys length (for offsetting positions)
        past_len = 0
        if cache_hidden is not None and len(cache_hidden) == 2 and len(cache_hidden[0]) > 0:
            past_len = sum(k.shape[-2] for k in cache_hidden[0])

        q, k, v = eagle_qkv_from_2h(hidden_2h, self.cfg)
        if q is None:
            # Wrapper path: project to H and let target attention do QKV/positional bias.
            x = self.pre_proj(hidden_2h)
            pos_ids = position_ids if past_len == 0 else (position_ids + past_len)
            out = self.attn(
                hidden_states=x,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            # Support both (out) and (out, attn, cache) signatures
            out = out[0] if isinstance(out, tuple) else out
            return eagle_after_attention(out)

        # Advanced path: true 2H→Q/K/V supplied by callback (must be b,h,t,d tensors).
        # Apply model-specific positional method with cache offset:
        head_dim = q.shape[-1]
        self._ensure_rope(head_dim, q)
        cos, sin = self._rope(q, seq_len=past_len + q_len)
        q, k, pos_ids = eagle_apply_positions(q, k, position_ids, past_len, rope_obj=(cos, sin), apply_rope_fn=default_apply_rotary_pos_emb)

        # If GQA/MQA is needed, do repeat_kv in the callback that builds q,k,v.

        # Minimal SDPA
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_probs, v)
        # (Optional) cache update can be performed by your outer loop (list-append).
        return eagle_after_attention(out)

# =======================================================
# 5) Decoder layer: two norms → concat(2H) → attention → resid → norm → MLP/MoE → resid
# =======================================================
class {model_name}DecoderLayer(nn.Module):
    def __init__(self, cfg: TargetConfig):
        super().__init__()
        self.cfg = cfg
        self.emb_norm = TargetRMSNorm(cfg.hidden_size, eps=getattr(cfg, "rms_norm_eps", 1e-6))
        self.hid_norm = TargetRMSNorm(cfg.hidden_size, eps=getattr(cfg, "rms_norm_eps", 1e-6))
        self.self_attn = EagleAttentionWrapper(cfg)
        self.post_attn_norm = TargetRMSNorm(cfg.hidden_size, eps=getattr(cfg, "rms_norm_eps", 1e-6))
        {mlp_or_moe}

    def forward(self, input_emb: torch.Tensor, hidden_states: torch.Tensor, attention_mask=None, position_ids=None, cache_hidden=None):
        # 2H concat
        x2h = torch.cat([self.emb_norm(input_emb), self.hid_norm(hidden_states)], dim=-1)
        # Attention + resid
        attn_out = self.self_attn(x2h, attention_mask=attention_mask, position_ids=position_ids, cache_hidden=cache_hidden)
        x = hidden_states + attn_out
        # MLP/MoE + resid
        x2 = self.post_attn_norm(x)
        x2 = self.mlp(x2)
        return x + x2

# =======================================================
# 6) Draft model: embeds → 3H→H projector → decoder → norm → lm_head (+ vocab maps)
# =======================================================
class {model_name}ForCausalLMEagle3(Eagle3DraftModel):
    config_class = TargetConfig

    def __init__(self, cfg: TargetConfig, quant_config=None):
        super().__init__(cfg)
        self.config = cfg
        self.vocab_size = cfg.vocab_size
        self.draft_vocab_size = getattr(cfg, "draft_vocab_size", cfg.vocab_size)

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, getattr(cfg, "pad_token_id", 0))
        self.proj_3h_to_h = nn.Linear(cfg.hidden_size * 3, cfg.hidden_size, bias=False)
        self.midlayer = {model_name}DecoderLayer(cfg)
        self.final_norm = TargetRMSNorm(cfg.hidden_size, eps=getattr(cfg, "rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(cfg.hidden_size, self.draft_vocab_size, bias=False)

        # Non-trainable vocab maps
        self.register_buffer("t2d", torch.zeros(self.vocab_size, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))

    # --- masks+positions to mirror target behavior ---
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_kv_len):
        combined = None
        if input_shape[-1] > 1:
            combined = _make_causal_mask(input_shape, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_kv_len)
        if attention_mask is not None:
            expanded = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined = expanded if combined is None else expanded + combined
        return combined

    # --- training/inference entry ---
    def forward(self, hidden_states: torch.Tensor, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, ttt_length: int = 1):
        # cache style: if ttt_length>1, caller should maintain list-cache; here we only pass it through.
        cache_hidden = None if ttt_length == 1 else [[], []]

        bsz, seqlen, _ = hidden_states.size()
        device = hidden_states.device
        position_ids = torch.arange(0, seqlen, dtype=torch.long, device=device)[None, :]

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seqlen), dtype=torch.bool, device=device)
        attn_mask = self._prepare_decoder_attention_mask(attention_mask, (bsz, seqlen), hidden_states, past_kv_len=0)

        x = self.proj_3h_to_h(hidden_states)  # 3H → H
        x = self.midlayer(inputs_embeds, x, attention_mask=attn_mask, position_ids=position_ids, cache_hidden=cache_hidden)
        x = self.final_norm(x)
        return x

    # --- SPEC loop hooks ---
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(-1) == self.config.hidden_size * 3, "expected concat(low,mid,high)"
        return self.proj_3h_to_h(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden_states))

    def backbone(self, input_embeds, hidden_states, cache_hidden, attention_mask, position_ids, use_cache=True):
        return self.midlayer(input_embeds, hidden_states, attention_mask, position_ids, cache_hidden)







"""

CALLBACKS_TEMPLATE = """\
# callbacks_{snake}.py — model-specific hooks for {model_name}
# Fill these only if the defaults don’t match your target; kept minimal on purpose.

import torch
import torch.nn as nn

# Optional: if you want TRUE 2H->Q/K/V inside attention, implement below and return (q,k,v)
# Expected shapes: (b,h,t,d). You can also do GQA/MQA (repeat_kv) here if your model requires it.
def eagle_qkv_from_2h(hidden_2h: torch.Tensor, cfg):
    return None, None, None  # return triplet to enable advanced path, keep None to use wrapper pre-proj

# Apply the target’s positional method with cache offset. You receive either a rope_obj (cos,sin)
# plus a function apply_rope_fn(q,k,cos,sin,position_ids,unsqueeze_dim=1), or you can ignore them
# if your target uses ALiBi or something else.
def eagle_apply_positions(q, k, position_ids, past_len, rope_obj=None, apply_rope_fn=None):
    if rope_obj is None or apply_rope_fn is None:
        # fallback: just offset positions, let target attention handle positions internally
        return q, k, (position_ids if past_len == 0 else (position_ids + past_len))
    cos, sin = rope_obj
    pos_ids = position_ids if past_len == 0 else (position_ids + past_len)
    q_rot, k_rot = apply_rope_fn(q, k, cos, sin, pos_ids, unsqueeze_dim=1)
    return q_rot, k_rot, pos_ids

# Post-attention tweak (e.g., output projection or residual scaling) if your target does anything special.
def eagle_after_attention(x):
    return x

# Provide a RotaryEmbedding module for your target (or return an instantiated target class)
def make_rotary(cfg, head_dim):
    # Example: swap to your target’s RotaryEmbedding if available
    # from {module} import DeepseekV3RotaryEmbedding
    # return DeepseekV3RotaryEmbedding(head_dim, max_position_embeddings=cfg.max_position_embeddings, base=getattr(cfg, "rope_theta", 10000))
    from .{snake}_eagle3 import SimpleRotaryEmbedding
    return SimpleRotaryEmbedding(head_dim, getattr(cfg, "max_position_embeddings", 2048), base=getattr(cfg, "rope_theta", 10000))
"""

def snake_case(name: str) -> str:
    return "".join(("_" + c.lower() if c.isupper() else c) for c in name).lstrip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="Python module that defines the target (e.g., modeling_deepseek)")
    ap.add_argument("--config-class", required=True, help="Config class name in module")
    ap.add_argument("--attention-class", required=True, help="Attention class name in module")
    ap.add_argument("--mlp-class", required=True, help="MLP/FFN class name in module")
    ap.add_argument("--norm-class", required=True, help="RMSNorm/LayerNorm class name in module")
    ap.add_argument("--decoder-class", default="", help="(Optional) decoder layer class in module — not required for scaffold")
    ap.add_argument("--model-name", required=True, help="Draft model prefix (e.g., DeepseekV3)")
    ap.add_argument("--use-moe", action="store_true", help="If the target uses MoE in some layers")
    ap.add_argument("--outdir", default="specforge/modeling/draft", help="Where to write the scaffold files")
    args = ap.parse_args()

    # Validate module import early (nice error if typos)
    try:
        importlib.import_module(args.module)
    except Exception as e:
        raise SystemExit(f"Could not import module '{args.module}': {e}")

    snake = snake_case(args.model_name)

    mlp_or_moe = "self.mlp = TargetMLP(cfg)"
    if args.use_moe:
        mlp_or_moe = textwrap.dedent(f"""\
            # If your target uses MoE, import and wire its MoE block here (or keep TargetMLP for dense layers).
            # Example:
            # from {args.module} import DeepseekV3MoE as TargetMoE
            # self.mlp = TargetMoE(cfg)
            self.mlp = TargetMLP(cfg)
        """)

    maybe_decoder_import = ""
    if args.decoder_class:
        maybe_decoder_import = f"\\nfrom {args.module} import {args.decoder_class} as TargetDecoder"

    os.makedirs(args.outdir, exist_ok=True)
    draft_path = os.path.join(args.outdir, f"{snake}_eagle3.py")
    callbacks_path = os.path.join(args.outdir, f"callbacks_{snake}.py")

    with open(draft_path, "w", encoding="utf-8") as f:
        f.write(SCAFFOLD_TEMPLATE.format(
            model_name=args.model_name,
            snake=snake,
            module=args.module,
            config_cls=args.config_class,
            attention_cls=args.attention_class,
            mlp_cls=args.mlp_class,
            norm_cls=args.norm_class,
            maybe_decoder_import=maybe_decoder_import,
            mlp_or_moe=mlp_or_moe,
        ))

    with open(callbacks_path, "w", encoding="utf-8") as f:
        f.write(CALLBACKS_TEMPLATE.format(
            model_name=args.model_name,
            module=args.module,
            snake=snake,
        ))

    print(f"Wrote {draft_path}")
    print(f"Wrote {callbacks_path}")
    print("Next steps:")
    print("  • Fill callbacks for QKV, pos embeddings, and MoE if your target differs.")
    print("  • Register the draft in your AutoEagle3DraftModel mapping or runtime registry.")
    print("  • Add a config JSON for training (configs/{snake}_eagle.json).")

if __name__ == "__main__":
    main()








python gen_draft_scaffold.py \
  --module modeling_deepseek \
  --config-class DeepseekV3Config \
  --attention-class DeepseekV3Attention \
  --mlp-class DeepseekV3MLP \
  --norm-class DeepseekV3RMSNorm \
  --model-name DeepseekV3 \
  --use-moe
