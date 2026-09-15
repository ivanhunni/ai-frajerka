# instruction_training/train_instruction.py
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from model.transformer import Transformer  # Predpokladaný názov vášho modelu
from tokenizers import Tokenizer
from model.config import TransformerConfig        # Predpokladaný názov vášho tokenizeru
from instruction_training.config import InstructionTrainingConfig
from instruction_training.dataset import InstructionDataset, collate_instruction_fn

def train_instruction_tuning():
    cfg = InstructionTrainingConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}")

    # 1. Načítanie Tokenizeru
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    # 2. Načítanie Predtrénovaného Modelu
    print("Načítavam predtrénovaný model...")
    model = Transformer(TransformerConfig(vocab_size=tokenizer.get_vocab_size(), max_seq_len=cfg.max_seq_len))
    if os.path.exists(cfg.pretrained_checkpoint_path):
        checkpoint = torch.load(cfg.pretrained_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint úspešne načítaný.")
    else:
        print(f"Upozornenie: Checkpoint {cfg.pretrained_checkpoint_path} nebol nájdený. Trénujem od nuly.")
    
    model.to(device)

    # 3. Dáta
    dataset = InstructionDataset(data_dirs=cfg.data_dirs, tokenizer=tokenizer, max_len=cfg.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_instruction_fn)

    # 4. Optimizer a Stratová funkcia (Loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = CrossEntropyLoss(ignore_index=-100)  # Maskovanie -100 tokenov

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.train()

    # 5. Trénovací cyklus
    print("Spúšťam Instruction Fine-Tuning...")
    global_step = 0

    for epoch in range(cfg.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids)
            
            # Výpočet Loss (Flatten pre CrossEntropyLoss)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                print(f"Epoch: {epoch+1}/{cfg.epochs} | Step: {step} | Loss: {loss.item():.4f}")

            # Uloženie checkpointu
            if global_step % cfg.save_every_steps == 0:
                save_path = os.path.join(cfg.output_dir, f"sft_checkpoint_step_{global_step}.pt")
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_path)
                print(f"Saved checkpoint to {save_path}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Priemerná Loss za epochu {epoch+1}: {avg_loss:.4f} ---")

    # Finálne uloženie
    final_path = os.path.join(cfg.output_dir, "final_instruction_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Trénovanie dokončené! Finálny model uložený v {final_path}")

if __name__ == "__main__":
    train_instruction_tuning()