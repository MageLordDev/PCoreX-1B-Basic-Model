"""
Hugging Face `transformers` ekosistemine tam uyumlu, TPU (torch_xla / accelerate)
üzerinde eğitilebilir causal dil modeli.

- `TrEnForCausalLM`, `PreTrainedModel` ve `GenerationMixin`'den türer:
  save_pretrained / from_pretrained / push_to_hub / Trainer / accelerate / PEFT
  hepsiyle doğrudan uyumludur.
- `AutoConfig` ve `AutoModelForCausalLM` ile kayıt edilir, böylece
  `AutoModelForCausalLM.from_pretrained("./tr_en_llm")` çalışır.
- KV-cache (past_key_values) destekler -> `model.generate()` kullanılabilir.
- Gradient checkpointing destekler (`model.gradient_checkpointing_enable()`).
- Eager attention kullanır (TPU/XLA ile en sorunsuz çalışan yol); istenirse
  `attn_implementation="sdpa"` GPU'da otomatik hızlanma sağlar.

Eğitim (TPU örneği, siz çalıştıracaksınız):

    # accelerate ile (önerilen, HF Trainer da içeride bunu kullanır)
    accelerate config   # TPU seçin
    accelerate launch train.py

    # ya da doğrudan HF Trainer:
    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(output_dir="out", tpu_num_cores=8, ...)
    trainer = Trainer(model=model, args=args, train_dataset=...)
    trainer.train()
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN

from configuration_tr_en import TrEnConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE - pozisyon bilgisini attention içine döndürerek gömer."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        # position_ids: (B, T)
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # cos/sin: (B, T, head_dim) -> (B, 1, T, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class TrEnAttention(nn.Module):
    def __init__(self, config: TrEnConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = config.dropout
        self.rotary = RotaryEmbedding(self.head_dim, base=config.rope_theta)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(position_ids, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v) if use_cache else None

        attn_mask = None
        is_causal = past_key_value is None
        if attention_mask is not None:
            # attention_mask: (B, 1, T_q, T_kv) additive mask (0 / -inf), zaten causal içerir
            attn_mask = attention_mask
            is_causal = False

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), present


class TrEnMLP(nn.Module):
    def __init__(self, config: TrEnConfig):
        super().__init__()
        hidden = int(config.ffn_mult * config.n_embd * 2 / 3)
        hidden = 256 * ((hidden + 255) // 256)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TrEnBlock(nn.Module):
    def __init__(self, config: TrEnConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = TrEnAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = TrEnMLP(config)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        attn_out, present = self.attn(
            self.ln1(x), attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class TrEnPreTrainedModel(PreTrainedModel):
    config_class = TrEnConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TrEnBlock"]
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


class TrEnModel(TrEnPreTrainedModel):
    """Gövde: embedding + transformer blokları (lm_head yok)."""

    def __init__(self, config: TrEnConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TrEnBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _build_causal_mask(attention_mask, seq_len, past_len, dtype, device):
        # (B, T) padding mask -> (B, 1, T, T+past) additive causal+padding mask
        total_len = seq_len + past_len
        causal = torch.full((seq_len, total_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        causal = torch.triu(causal, diagonal=1 + past_len)
        causal = causal[None, None, :, :]
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
            causal = causal + pad
        return causal

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_hidden_states=False,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, T, _ = inputs_embeds.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)

        causal_mask = self._build_causal_mask(attention_mask, T, past_len, inputs_embeds.dtype, inputs_embeds.device)

        x = self.drop(inputs_embeds)
        all_hidden = () if output_hidden_states else None
        new_past = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden = all_hidden + (x,)
            layer_past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                x, present = torch.utils.checkpoint.checkpoint(
                    layer, x, causal_mask, position_ids, layer_past, use_cache, use_reentrant=False,
                )
            else:
                x, present = layer(x, attention_mask=causal_mask, position_ids=position_ids,
                                    past_key_value=layer_past, use_cache=use_cache)
            if use_cache:
                new_past.append(present)

        x = self.norm(x)
        if output_hidden_states:
            all_hidden = all_hidden + (x,)

        return x, new_past, all_hidden


class TrEnForCausalLM(TrEnPreTrainedModel, GenerationMixin):
    """AutoModelForCausalLM ile kayıt edilecek üst sınıf (lm_head dahil)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TrEnConfig):
        super().__init__(config)
        self.model = TrEnModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        if config.tie_weights:
            self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else False

        hidden_states, new_past, all_hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past,
            hidden_states=all_hidden,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel()
        return n


# --- Auto sınıflarına kayıt: AutoConfig / AutoModelForCausalLM ile çalışsın ---
TrEnConfig.register_for_auto_class()
TrEnForCausalLM.register_for_auto_class("AutoModelForCausalLM")


if __name__ == "__main__":
    config = TrEnConfig()
    model = TrEnForCausalLM(config)
    total = model.num_params()
    print(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, vocab={config.vocab_size}")
    print(f"Toplam parametre sayısı: {total:,} (~{total/1e6:.1f}M)")

    x = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids=x, labels=x)
    print(f"logits shape: {tuple(out.logits.shape)}, loss (rastgele init): {out.loss.item():.3f}")

    # HF standart kaydetme/yükleme testi
    model.save_pretrained("./tr_en_llm_ckpt")
    config.save_pretrained("./tr_en_llm_ckpt")
    print("save_pretrained tamam -> ./tr_en_llm_ckpt")
