"""
Türkçe + İngilizce için BPE tokenizer eğitim scripti (HuggingFace `tokenizers` kütüphanesi).

Kurulum:
    pip install tokenizers

Kullanım:
    python tokenizer.py --files data_tr.txt data_en.txt --vocab_size 32000 --out tr_en_tokenizer.json

Notlar (Türkçe için önemli detaylar):
- NFC unicode normalizasyonu kullanılır (İ/ı/ş/ğ/ç/ö/ü gibi karakterler bozulmasın diye).
- Byte-level BPE seçildi: hem Türkçe'nin zengin çekim eklerini (agglutinative yapı)
  alt-kelime (subword) parçalarına iyi bölebiliyor, hem de OOV (bilinmeyen karakter) sorunu yaşanmıyor.
- lowercase yapılmıyor çünkü Türkçe'de "İstanbul/istanbul" ve "I/ı/İ/i" ayrımı kritik;
  case-folding kelime anlamını bozabilir (İ <-> i, I <-> ı farkı İngilizce'den farklı çalışır).
- Özel tokenlar hem TR hem EN kullanım senaryolarını (chat, kod, doldurma) kapsayacak şekilde eklendi.
"""

import argparse

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<mask>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    # NFC normalizasyonu: Türkçe aksanlı/nokta karakterlerinin (İ, ı, ş, ğ, ç, ö, ü)
    # tutarlı tek bir unicode formunda kalmasını sağlar.
    tok.normalizer = normalizers.Sequence([NFC()])
    # Byte-level pre-tokenizer: hem TR hem EN metni kayıpsız şekilde byte'lara ayırır,
    # bilinmeyen karakter problemi (OOV) oluşmaz.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B <eos>",
        special_tokens=[("<bos>", 2), ("<eos>", 3)],
    )
    return tok


def train(files, vocab_size, out_path, min_frequency=2):
    tok = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        # Türkçe agglutinative yapı için ek/kelime köklerinin ayrı token olarak
        # öğrenilebilmesi adına alfabeyi baştan tüm byte'larla başlatıyoruz.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train(files, trainer)
    tok.save(out_path)
    print(f"Tokenizer kaydedildi: {out_path}")
    print(f"Gerçek vocab boyutu: {tok.get_vocab_size()}")
    return tok


def quick_test(tok_path):
    tok = Tokenizer.from_file(tok_path)
    samples = [
        "Merhaba dünya, bugün hava çok güzel ve İstanbul'da yürüyüş yapıyorum.",
        "Hello world, this is a quick tokenizer sanity check for the model.",
        "Yapay zeka modelleri büyük miktarda veriyle eğitiliyor.",
    ]
    for s in samples:
        enc = tok.encode(s)
        print(f"\nMetin: {s}")
        print(f"Token sayısı: {len(enc.ids)}")
        print(f"Tokenlar: {enc.tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="TR ve EN düz metin dosyaları (.txt)")
    parser.add_argument("--vocab_size", type=int, default=36864)
    parser.add_argument("--out", type=str, default="tr_en_tokenizer.json")
    parser.add_argument("--test", action="store_true", help="Eğitimden sonra örnek cümlelerle test et")
    args = parser.parse_args()

    train(args.files, args.vocab_size, args.out)
    if args.test:
        quick_test(args.out)
