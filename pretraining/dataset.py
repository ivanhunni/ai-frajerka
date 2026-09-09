import glob
import json
import os
import torch
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, seq_length: int = 512):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = []

        # Načítanie všetkých textových a JSON súborov zo zložky pretraining
        file_paths = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True) + \
                     glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)

        full_text = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith(".json"):
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            full_text.append(item.get("text", ""))
                    elif isinstance(data, dict):
                        full_text.append(data.get("text", ""))
                else:
                    full_text.append(f.read())

        # Tokenizácia spojeného textu
        combined_text = "\n".join(full_text)
        encoded = self.tokenizer.encode(combined_text)
        self.tokens = torch.tensor(encoded.ids if hasattr(encoded, 'ids') else encoded, dtype=torch.long)

    def __len__(self):
        if len(self.tokens) <= self.seq_length:
            return 0
        return (len(self.tokens) - 1) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1:end_idx + 1]

        return x, y