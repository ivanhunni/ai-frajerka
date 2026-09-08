import os
import glob
import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer_path: str, max_seq_len: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.tokens = []

        # Načítanie všetkých .json a .jsonl súborov v zadanom adresári
        filepaths = glob.glob(os.path.join(data_dir, "**/*.json*"), recursive=True)
        
        texts = []
        for fp in filepaths:
            with open(fp, "r", encoding="utf-8") as f:
                if fp.endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            texts.append(data.get("text", ""))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            texts.append(item.get("text", ""))
                    elif isinstance(data, dict):
                        texts.append(data.get("text", ""))

        # Tokenizácia spojeného textu
        full_text = " ".join(texts)
        encoded = self.tokenizer.encode(full_text)
        self.tokens = encoded.ids

    def __len__(self):
        # Počet sekvencií s dĺžkou max_seq_len
        return max(0, len(self.tokens) - self.max_seq_len)

    def __getitem__(self, idx: int):
        # Vstup (x) a cieľ (y) posunutý o 1 token dopredu pre autoregresný tréning
        chunk = self.tokens[idx : idx + self.max_seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y