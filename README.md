# TR-EN 1B — Hugging Face + TPU Compatible Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **1.0B-parameter** (1,000,341,504 weights, 1,000,417,280 with RMSNorm), decoder-only
(GPT/LLaMA-style) language model architecture for Turkish + English. It is a full
**Hugging Face `transformers` implementation**: built on `PreTrainedModel`, registered
for `AutoModelForCausalLM`, and compatible with `save_pretrained` / `push_to_hub` /
`Trainer` / `accelerate` / **TPU (torch_xla)**.

The weights are randomly initialized: this repository provides the architecture,
tokenizer pipeline and a training skeleton, **not a trained model**.

The architecture is identical to the 196M variant (see [`README_196M.md`](README_196M.md));
only depth, width and vocabulary size were scaled up.

## Files

| File | Description |
|---|---|
| `configuration_tr_en.py` | `TrEnConfig(PretrainedConfig)` — model hyperparameters (defaults = 1B) |
| `modeling_tr_en.py` | `TrEnForCausalLM(PreTrainedModel, GenerationMixin)` — the architecture |
| `tokenizer.py` | Trains the TR+EN byte-level BPE tokenizer (raw `tokenizers` output) |
| `convert_tokenizer.py` | Converts the raw tokenizer into a standard HF tokenizer folder |
| `train.py` | Training skeleton built on HF `Trainer` (same code on TPU/GPU) |
| `requirements-gpu.txt` | Version-pinned dependencies for CUDA GPUs |
| `requirements-tpu.txt` | Version-pinned dependencies for TPU (`torch_xla`) |
| `LICENSE` | MIT License |

## Architecture

| Setting | Value |
|---|---|
| Type | Decoder-only causal LM |
| Layers (`n_layer`) | 18 |
| Hidden size (`n_embd`) | 2048 |
| Attention heads (`n_head`) | 16 (head_dim = 128) |
| FFN hidden (SwiGLU) | 5632 |
| Context length (`block_size`) | 2048 |
| Vocabulary size | 36,864 (byte-level BPE) |
| Positional encoding | RoPE (`rope_theta = 10000`) |
| Normalization | RMSNorm (pre-norm) |
| MLP | SwiGLU, no biases |
| Embeddings | Input embedding tied with `lm_head` |
| Attention | Multi-head, `scaled_dot_product_attention`, KV-cache |
| Extras | Gradient checkpointing, `generate()` support |

Parameter count:
```
total = vocab*n_embd + n_layer * (4*n_embd² + 3*n_embd*ffn_hidden)
      = 36864*2048 + 18 * (4*2048² + 3*2048*5632)
      = 1,000,341,504   (1,000,417,280 including RMSNorm weights)
```

To scale to another size, change `n_layer / n_embd / vocab_size` in
`configuration_tr_en.py`; the model code does not need to change.

## Requirements

### Software (pinned)

`torch_xla` must match the `torch` version exactly, so there are two separate pinned files:

| Environment | Install |
|---|---|
| GPU (CUDA) | `pip install -r requirements-gpu.txt` |
| TPU | `pip install -r requirements-tpu.txt` |

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.2 |
| tokenizers | 0.19.1 |
| datasets | 2.21.0 |
| accelerate | 0.34.2 |
| torch_xla *(TPU only)* | 2.4.0 — **must equal the torch version** |

If you use a different combination (newer TPU runtime, different Python or CUDA version):
1. Keep `torch` and `torch_xla` on the **same** version.
2. If you change `transformers`, re-check that `CausalLMOutputWithPast` and
   `GenerationMixin.prepare_inputs_for_generation` still match the signatures used in
   `modeling_tr_en.py`.
3. Make sure your Python version is supported by the `torch_xla` wheel you pick
   (usually 3.10–3.11).

### Hardware (approximate)

| Use case | Memory |
|---|---|
| Inference, fp16/bf16 | ~2 GB weights + ~0.3 GB KV-cache at 2048 tokens |
| Training, mixed precision + AdamW | ~16 GB for weights/gradients/optimizer states, plus activations |
| Training, pure bf16 (TPU) | ~12 GB, plus activations |

A Chinchilla-style budget for this size is on the order of ~20B tokens
(≈1.2e20 FLOPs).

**Memory strategy matters at 1B:**
- **16 GB GPUs (e.g. Kaggle T4):** T4 does not support bf16, so use fp16. Full AdamW
  state does not fit on a single 16 GB card; use an 8-bit optimizer, DeepSpeed ZeRO-2/3
  or FSDP.
- **TPU v3-8 (16 GB HBM per core):** a full replica per core leaves little room for
  activations, so use FSDP / SPMD sharding.
- `train.py` is a plain `Trainer` skeleton with gradient checkpointing enabled. It does
  **not** configure FSDP/ZeRO, so add the memory strategy that fits your hardware.

## Usage

### 1) Train the tokenizer
```bash
python tokenizer.py --files data_tr.txt data_en.txt --vocab_size 36864 --out tr_en_tokenizer.json --test
```
`data_tr.txt` / `data_en.txt` are plain-text corpora, one sentence or paragraph per line.

### 2) Convert the tokenizer to HF format
```bash
python convert_tokenizer.py --tokenizer_json tr_en_tokenizer.json --out ./tr_en_llm_ckpt
```

### 3) Sanity-check the architecture (optional)
```bash
python modeling_tr_en.py
# prints parameter count, random-init loss and runs a save_pretrained test
```

### 4) Train
```bash
accelerate config       # choose TPU or GPU
accelerate launch train.py \
    --tokenizer_dir ./tr_en_llm_ckpt \
    --train_file data_tr_en.txt \
    --output_dir ./tr_en_llm_out \
    --bf16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4
```
Use `bf16` on TPU / Ampere+ GPUs; on T4 use fp16 instead. Batch size, learning rate and
number of steps depend on your data and hardware.

### 5) Load after training
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("./tr_en_llm_out")
model = AutoModelForCausalLM.from_pretrained("./tr_en_llm_out", trust_remote_code=True)

out = model.generate(**tok("Merhaba,", return_tensors="pt"), max_new_tokens=30)
print(tok.decode(out[0]))
```

### 6) Push to the Hub (optional)
```python
model.push_to_hub("your-username/tr-en-gpt-1b")
tok.push_to_hub("your-username/tr-en-gpt-1b")
```

## Tokenizer design (Turkish + English)

- **Byte-level BPE** (GPT-2/GPT-4 style): handles Turkish agglutinative morphology well
  and never produces unknown characters.
- **NFC normalization:** keeps İ, ı, ş, ğ, ç, ö, ü consistent.
- **No lowercasing:** İ/i and I/ı are distinct in Turkish, so case is preserved.
- **Special tokens:** `<pad> <unk> <bos> <eos> <sep> <mask>` and
  `<|user|> <|assistant|> <|system|>` for chat formatting.

## Status and limitations

- The model is **untrained**; `generate()` will produce random tokens until you train it.
- The code was only syntax-checked (`py_compile`) where it was written (no torch, GPU or
  TPU available there). Run `python modeling_tr_en.py` on your own environment first.
- `train.py` is a starting point, not a tuned training recipe.

## License

This project is released under the [MIT License](LICENSE).

Copyright (c) 2026 Beytullah (MageLordDev / EnderArcane Studio)
