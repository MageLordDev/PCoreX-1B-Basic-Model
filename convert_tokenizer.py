"""
tokenizer.py ile eğitilen ham `tokenizers` çıktısını (tr_en_tokenizer.json),
Hugging Face `AutoTokenizer.from_pretrained(...)` ile doğrudan yüklenebilecek
standart bir HF tokenizer klasörüne çevirir.

Kullanım:
    python convert_tokenizer.py --tokenizer_json tr_en_tokenizer.json --out ./tr_en_llm_ckpt
"""

import argparse

from transformers import PreTrainedTokenizerFast


def convert(tokenizer_json_path: str, out_dir: str, model_max_length: int = 2048):
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        sep_token="<sep>",
        mask_token="<mask>",
        additional_special_tokens=["<|user|>", "<|assistant|>", "<|system|>"],
        model_max_length=model_max_length,
    )
    tok.save_pretrained(out_dir)
    print(f"HF tokenizer kaydedildi: {out_dir}")
    print(f"Örnek: {tok('Merhaba dünya, hello world').input_ids}")
    return tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_json", type=str, required=True)
    parser.add_argument("--out", type=str, default="./tr_en_llm_ckpt")
    parser.add_argument("--model_max_length", type=int, default=2048)
    args = parser.parse_args()
    convert(args.tokenizer_json, args.out, args.model_max_length)
