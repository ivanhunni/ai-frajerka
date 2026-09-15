from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus.txt"
OUTPUT = ROOT / "tokenizer.json"

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]

def train():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=1000, min_frequency=1, special_tokens=SPECIAL_TOKENS)
    tokenizer.train([str(CORPUS)], trainer)
    tokenizer.save(str(OUTPUT))
    print(f"Tokenizer uložený do {OUTPUT}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train()
