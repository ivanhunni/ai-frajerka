from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train():
    # Inicializácia BPE tokenizera
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Nastavenie trenéra a špeciálnych tokenov
    trainer = BpeTrainer(
        vocab_size=1000,
        min_frequency=1,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # Trénovanie na texte zo súboru corpus.txt
    files = ["corpus.txt"]
    tokenizer.train(files, trainer)

    # Uloženie naučeného slovníka a pravidiel do tokenizer.json
    tokenizer.save("tokenizer.json")
    print("Tokenizer bol úspešne vytrénovaný a uložený do 'tokenizer.json'.")

if __name__ == "__main__":
    train()