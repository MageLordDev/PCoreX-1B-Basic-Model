"""
Hugging Face uyumlu config sınıfı.
AutoConfig ile kayıt edilir, save_pretrained/from_pretrained otomatik çalışır.
"""

from transformers import PretrainedConfig


class TrEnConfig(PretrainedConfig):
    model_type = "tr_en_gpt"

    def __init__(
        self,
        vocab_size: int = 36864,
        block_size: int = 2048,
        n_layer: int = 18,
        n_head: int = 16,
        n_embd: int = 2048,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
        tie_weights: bool = True,
        rope_theta: float = 10000.0,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_position_embeddings = block_size  # HF ekosistemiyle uyumluluk için alias
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.ffn_mult = ffn_mult
        self.dropout = dropout
        self.bias = bias
        self.tie_weights = tie_weights
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_weights,
            **kwargs,
        )
