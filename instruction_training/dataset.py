# instruction_training/dataset.py
import json
import glob
import os
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data_dirs, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        # Načítanie všetkých .json súborov zo zadaných priečinkov
        for data_dir in data_dirs:
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            for file_path in json_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.samples.extend(data)
                        elif isinstance(data, dict):
                            self.samples.append(data)
                    except json.JSONDecodeError:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Očakáva formát: {"instruction": "...", "input": "...", "output": "..."}
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # Skladanie promptu
        if input_text:
            user_prompt = f"### Inštrukcia:\n{instruction}\n\n### Vstup:\n{input_text}\n\n### Odpoveď:\n"
        else:
            user_prompt = f"### Inštrukcia:\n{instruction}\n\n### Odpoveď:\n"

        full_text = user_prompt + output_text

        # Tokenizácia
        prompt_ids = self.tokenizer.encode(user_prompt).ids
        full_ids = self.tokenizer.encode(full_text).ids

        # Orezanie na max_len
        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        # Vytvorenie masky pre loss (-100 povie PyTorch CrossEntropy, aby daný token ignoroval)
        # Nechceme platiť stratu za generovanie inštrukcie, iba za generovanie odpovede!
        labels = list(target_ids)
        prompt_len = len(prompt_ids) - 1
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # Padding (doplnenie na max_len ak je treba, tu implementované dynamicky)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_instruction_fn(batch):
    """ Dynamický padder pre batche """
    max_len = max(len(item["input_ids"]) for item in batch)
    
    pad_id = 0  # Predpokladáme pad_id = 0
    input_ids_padded = []
    labels_padded = []

    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - len(inp)

        inp_p = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
        lbl_p = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids_padded.append(inp_p)
        labels_padded.append(lbl_p)

    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded)
    }