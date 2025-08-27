# llm-draftmodels-eagle3

**Generalized framework for building Eagle3-style draft models from arbitrary target LLMs.**  
Includes helper functions, decoder modifications, caching logic, and examples adapting DeepSeekV3 into the SpecForge Eagle3 training pipeline.

---

## Overview

This is a report on how users will be able to build and train a draft model based on a different target model (can be customized or derived from Hugging Face).  

It is separated into three sections:  
1. The generalized structure of a draft model file  
2. The detailed (non-code) logic of building a draft model class  
3. The detailed code sections that will need to be imported or changed from a target model  

It is derived based on the example of fitting the **DeepSeekV3 target model** from Hugging Face into the **SpecForge llama3_eagle training pipeline**.  

- DeepSeekV3 reference: [modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py)  
- DeepSeekV3-eagle3 draft model: [Google Doc link](https://docs.google.com/document/d/1FBoB8sDmWACiZEzSU6XJ4jLPz7ZnQmcucwPRMdfkKa8/edit?usp=sharing) (see `deepseekV3_eagle3.py` tab)  

---

## General Structure of a Draft Model File

- **Helper functions:**  
  - Position functions: `rotate_half`, `apply_rotary_pos_emb` (or ALiBi/YARN math)  
  - Masking functions: `make_causal_mask`, `_expand_mask`, `ensure_mask_width`, `stack_kv_cache`  
  - Scaling/embedding functions: NTK/linear scaling helpers, `repeat_kv`  

- **[Model]RotaryEmbedding Classes**  
  - e.g., `LinearScaling`, `DynamicNTKScaling`, `Yarn`  

- **Attention class:**  
  - `class [Model]Attention(nn.Module)`  
  - Eagle changes: Q/K/V from 2H (concat of normed emb + normed hidden), RoPE/ALiBi applied, cache-aware, SDPA path preferred  

- **MLP class:** `class [Model]V3MLP(nn.Module)`  
- **RMSNorm:** `class [Model]RMSNorm(nn.Module)`  
- **MoE classes (if applicable)**  
- **Model decoder layer:**  
  - Two norms (emb_norm, hid_norm) → concat to 2H → attention → residual → post-attn norm → MLP → residual  

- **Draft Model class (executed during training loop):**  
  - embeddings → fc (3H → H) → decoder layer(s) → final norm → lm_head (to draft vocab) → optional vocab maps (`t2d`, `d2t`)  

---

## Building a Draft Model Class

`class [TargetModel]ForCausalLMEagle3(Eagle3DraftModel):`

### Initialize building blocks
1. **Embeddings**  
2. **Eagle-target model decoder layer** (imported from target model)  
3. **Projector (3x → 1x)** for concatenated low/mid/high hidden states  
4. **Final normalization** (same type as target model)  
5. **lm_head** (linear layer: hidden → draft vocab size)  
6. **Vocab maps** (`t2d`, `d2t`) stored as non-trainable tensors  

### Masking & positions
- `_prepare_decoder_attention_mask`: combines causal + padding masks from target  

### Forward loop
- Choose cache method:  
  - 1 token → no cache  
  - >1 tokens → Eagle lightweight cache (reuse K/V)  
- Project 3× features → 1× feature  
- Run Eagle decoder:  
  - Normalize input/hidden separately, concat (2H)  
  - Apply attention + positions/embeddings  
  - Residual connection → MLP block → residual  
- Apply normalization & return hidden states  

**Why faster?**  
- Cache avoids recomputation  
- Draft model runs fewer layers  

### Wrapper functions
- `embed_input_ids(ids)` → embeddings  
- `project_hidden_states(3×)` → fc projector  
- `compute_logits(h)` → norm → lm_head → logits  
- `backbone` → forward of single decoder layer  

---

## Detailed Code Notes

- *Pink highlights*: still in progress  
- *Orange/yellow*: must be customized per model  

### Steps:
1. Create `[Model]_eagle.py` under `specforge/modeling/draft/`  
2. Use `llama_eagle3` as reference  
3. Keep helper functions (`_make_causal_mask`, `_expand_mask`, `repeat_kv`)  
4. Reuse/override `rotate_half` and `apply_rotary_pos_emb` if target model doesn’t have them  
5. Import/modify target model classes:  
   - RotaryEmbedding  
   - Attention (double hidden size → 2H for QKV)  
   - MLP  
   - RMSNorm  
   - MoE (if needed)  
   - DecoderLayer  
6. Modify forward loop with Eagle logic (see included code in this repo/report)  

---

## Auto Model Mapping

Update `specforge/modeling/auto.py`:  

```python
class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
        [Model]: [Model]ForCausalLMEagle3,  # NEW
    }

class AutoDraftModelConfig:
    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        "[Model]ForCausalLMEagle3": [Model]Config,  # NEW
    }

