"""
Hazır eğitim iskeleti — TPU (torch_xla / accelerate) ve GPU'da aynı kodla çalışır.
Kendi veri setinizi/hiperparametrelerinizi girip doğrudan çalıştırabilirsiniz.

TPU'da çalıştırma:
    pip install torch torch_xla transformers datasets accelerate
    accelerate config              # "TPU" seçin, çekirdek sayısını girin
    accelerate launch train.py --train_file data.txt

GPU'da çalıştırma (aynı script):
    accelerate launch train.py --train_file data.txt

Not: Bu dosya bir iskelettir; veri seti yolu, batch size, learning rate vb.
kendi eğitim planınıza göre ayarlanmalı. Eğitimin kendisi sizin sorumluluğunuzda.
"""

import argparse

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from configuration_tr_en import TrEnConfig
from modeling_tr_en import TrEnForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, default="./tr_en_llm_ckpt",
                         help="convert_tokenizer.py çıktısı HF tokenizer klasörü")
    parser.add_argument("--train_file", type=str, required=True, help="txt/json/jsonl korpus")
    parser.add_argument("--output_dir", type=str, default="./tr_en_llm_out")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--bf16", action="store_true", help="TPU/A100 için bf16 önerilir")
    parser.add_argument("--tpu_num_cores", type=int, default=None,
                         help="Eski HF sürümlerinde TPU çekirdek sayısı; accelerate kullanıyorsanız gerek yok")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # --- Model: sıfırdan (config'ten) başlat ---
    config = TrEnConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = TrEnForCausalLM(config)
    print(f"Model parametre sayısı: {model.num_params()/1e6:.1f}M")

    # --- Veri seti ---
    raw = load_dataset("text", data_files={"train": args.train_file})

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // args.block_size) * args.block_size
        result = {
            "input_ids": [concatenated[i:i + args.block_size] for i in range(0, total_len, args.block_size)]
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to=["tensorboard"],
        tpu_num_cores=args.tpu_num_cores,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Eğitim bitti, model kaydedildi: {args.output_dir}")


if __name__ == "__main__":
    main()
