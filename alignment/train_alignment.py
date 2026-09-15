# alignment/train_alignment.py
import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.transformer import Transformer
from tokenizers import Tokenizer
from model.config import TransformerConfig
from alignment.config import AlignmentConfig
from alignment.dataset import DPODataset, collate_dpo_fn

def compute_log_probs(logits, labels, mask):
    # Log-probabilities pre jednotlivé tokeny
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_log_probs * mask).sum(dim=-1)

def train_dpo_alignment():
    cfg = AlignmentConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie pre Alignment (DPO): {device}")

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    # 1. Načítanie Trénovaného Modelu (Policy)
    print("Načítavam konverzačný model...")
    policy_model = Transformer(TransformerConfig(vocab_size=tokenizer.get_vocab_size(), max_seq_len=cfg.max_seq_len))
    if os.path.exists(cfg.conversation_checkpoint_path):
        checkpoint = torch.load(cfg.conversation_checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        policy_model.load_state_dict(state_dict)
    policy_model.to(device)

    # 2. Vytvorenie Zmrazeného Referenčného Modelu (Reference Model)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3. Dáta a Optimizer
    dataset = DPODataset(alignment_dirs=cfg.alignment_dirs, tokenizer=tokenizer, max_len=cfg.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_dpo_fn)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.output_dir, exist_ok=True)
    policy_model.train()

    print(f"Spúšťam Alignment (DPO) na {len(dataset)} vzorkách...")
    global_step = 0

    for epoch in range(cfg.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            optimizer.zero_grad()

            # Pass cez Policy Model
            policy_chosen_logits = policy_model(chosen_ids)
            policy_rejected_logits = policy_model(rejected_ids)

            # Pass cez Reference Model (bez gradientov)
            with torch.no_grad():
                ref_chosen_logits = ref_model(chosen_ids)
                ref_rejected_logits = ref_model(rejected_ids)

            # Výpočet log-pravdepodobností
            pi_logps_chosen = compute_log_probs(policy_chosen_logits[:, :-1, :], chosen_ids[:, 1:], chosen_mask[:, 1:])
            pi_logps_rejected = compute_log_probs(policy_rejected_logits[:, :-1, :], rejected_ids[:, 1:], rejected_mask[:, 1:])
            
            ref_logps_chosen = compute_log_probs(ref_chosen_logits[:, :-1, :], chosen_ids[:, 1:], chosen_mask[:, 1:])
            ref_logps_rejected = compute_log_probs(ref_rejected_logits[:, :-1, :], rejected_ids[:, 1:], rejected_mask[:, 1:])

            # DPO Loss
            pi_logratios = pi_logps_chosen - pi_logps_rejected
            ref_logratios = ref_logps_chosen - ref_logps_rejected
            
            loss = -F.logsigmoid(cfg.beta * (pi_logratios - ref_logratios)).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                print(f"Epoch: {epoch+1}/{cfg.epochs} | Step: {step} | DPO Loss: {loss.item():.4f}")

            if global_step % cfg.save_every_steps == 0:
                save_path = os.path.join(cfg.output_dir, f"aligned_checkpoint_step_{global_step}.pt")
                torch.save({"model_state_dict": policy_model.state_dict()}, save_path)

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"--- Priemerná DPO Loss za epochu {epoch+1}: {avg_loss:.4f} ---")

    # Finálne uloženie
    final_path = os.path.join(cfg.output_dir, "final_aligned_model.pt")
    torch.save(policy_model.state_dict(), final_path)
    print(f"Alignment dokončený! Zarovnaný model je uložený v: {final_path}")

if __name__ == "__main__":
    train_dpo_alignment()