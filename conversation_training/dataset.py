# conversation_training/dataset.py
import json
import glob
import os
import torch
from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, conversation_dirs, personality_dirs, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        # 1. Načítanie konverzačných dát
        for c_dir in conversation_dirs:
            for file_path in glob.glob(os.path.join(c_dir, "*.json*")):
                self._load_file(file_path)

        # 2. Načítanie osobných dát (casual, romantic, atď.)
        for p_dir in personality_dirs:
            for file_path in glob.glob(os.path.join(p_dir, "*.json*")):
                self._load_file(file_path)

    def _load_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.samples.extend(data)
                elif isinstance(data, dict):
                    self.samples.append(data)
        except (json.JSONDecodeError, Exception):
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Očakáva sa štruktúra:
        # {
        #   "system": "Si milá a starostlivá priateľka.",
        #   "messages": [
        #       {"role": "user", "content": "Ahoj, ako sa máš?"},
        #       {"role": "assistant", "content": "Ahoj! Mám sa super, myslela som na teba."}
        #   ]
        # }
        system_prompt = item.get("system", "Si inteligentná a podporujúca AI priateľka.")
        messages = item.get("messages", [])

        # Skladanie konverzačného kontextu a maskovanie
        full_input_ids = []
        labels = []

        # Vloženie systémového promptu
        sys_formatted = f"<|system|>\n{system_prompt}\n"
        sys_ids = self.tokenizer.encode(sys_formatted).ids
        full_input_ids.extend(sys_ids)
        labels.extend([-100] * len(sys_ids))  # Systémový prompt sa netrénuje

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                formatted = f"<|user|>\n{content}\n"
                ids = self.tokenizer.encode(formatted).ids
                full_input_ids.extend(ids)
                labels.extend([-100] * len(ids))  # Správy používateľa ignorujeme
            else:
                formatted = f"<|assistant|>\n{content}\n"
                ids = self.tokenizer.encode(formatted).ids
                full_input_ids.extend(ids)
                labels.extend(ids)  # Odpovede AI sa učíme generovať

        # Orezanie na max_len
        if len(full_input_ids) > self.max_len:
            full_input_ids = full_input_ids[:self.max_len]
            labels = labels[:self.max_len]

        input_ids = full_input_ids[:-1]
        target_labels = labels[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_labels, dtype=torch.long)
        }

def collate_conversation_fn(batch):
    """ Dynamické zarovnanie (padding) pre konverzačné batche """
    max_len = max(len(item["input_ids"]) for item in batch)
    
    pad_id = 0
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