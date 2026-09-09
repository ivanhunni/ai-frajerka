# alignment/dataset.py
import json
import glob
import os
import torch
from torch.utils.data import Dataset

class DPODataset(Dataset):
    def __init__(self, alignment_dirs, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        for a_dir in alignment_dirs:
            for file_path in glob.glob(os.path.join(a_dir, "*.json*")):
                self._load_file(file_path)

    def _load_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.samples.extend(data)
                elif isinstance(data, dict):
                    self.samples.append(data)
        except Exception:
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Očakávaná štruktúra:
        # {
        #   "prompt": "### Inštrukcia:\nPrečo si na mňa taká chladná?\n\n### Odpoveď:\n",
        #   "chosen": "Prepáč, nemyslela som to tak. Mala som len ťažký deň, poďme sa porozprávať.",
        #   "rejected": "To je tvoj problém. Nestaraj sa."
        # }
        prompt = item.get("prompt", "")
        chosen_text = item.get("chosen", item.get("output", ""))
        rejected_text = item.get("rejected", "Neviem.")

        chosen_full = prompt + chosen_text
        rejected_full = prompt + rejected_text

        chosen_ids = self.tokenizer.encode(chosen_full)[:self.max_len]
        rejected_ids = self.tokenizer.encode(rejected_full)[:self.max_len]
        prompt_ids = self.tokenizer.encode(prompt)[:self.max_len]

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "prompt_len": len(prompt_ids)
        }

def collate_dpo_fn(batch):
    max_chosen_len = max(len(item["chosen_ids"]) for item in batch)
    max_rejected_len = max(len(item["rejected_ids"]) for item in batch)
    
    pad_id = 0
    chosen_padded, rejected_padded = [], []
    chosen_masks, rejected_masks = [], []

    for item in batch:
        c_ids = item["chosen_ids"]
        r_ids = item["rejected_ids"]
        p_len = item["prompt_len"]

        # Padding pre chosen
        c_pad = max_chosen_len - len(c_ids)
        c_p = torch.cat([c_ids, torch.full((c_pad,), pad_id, dtype=torch.long)])
        c_m = torch.cat([torch.zeros(p_len), torch.ones(len(c_ids) - p_len), torch.zeros(c_pad)])

        # Padding pre rejected
        r_pad = max_rejected_len - len(r_ids)
        r_p = torch.cat([r_ids, torch.full((r_pad,), pad_id, dtype=torch.long)])
        r_m = torch.cat([torch.zeros(p_len), torch.ones(len(r_ids) - p_len), torch.zeros(r_pad)])

        chosen_padded.append(c_p)
        rejected_padded.append(r_p)
        chosen_masks.append(c_m)
        rejected_masks.append(r_m)

    return {
        "chosen_ids": torch.stack(chosen_padded),
        "rejected_ids": torch.stack(rejected_padded),
        "chosen_mask": torch.stack(chosen_masks),
        "rejected_mask": torch.stack(rejected_masks)
    }