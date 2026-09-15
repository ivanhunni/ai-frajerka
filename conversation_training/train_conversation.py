# conversation_training/train_conversation.py
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from model.transformer import Transformer  # Názov vášho modelu
from tokenizers import Tokenizer
from model.config import TransformerConfig
from conversation_training.config import ConversationTrainingConfig
from conversation_training.dataset import ConversationDataset, collate_conversation_fn

def train_conversation_tuning():
    cfg = ConversationTrainingConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie pre Conversation Training: {device}")

    # 1. Načítanie Tokenizeru
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    # 2. Načítanie Predchádzajúceho Modelu
    print("Načítavam checkpoint pre konverzačný tréning...")
    model = Transformer(TransformerConfig(vocab_size=tokenizer.get_vocab_size(), max_seq_len=cfg.max_seq_len))
    if os.path.exists(cfg.base_checkpoint_path):
        checkpoint = torch.load(cfg.base_checkpoint_path, map_location=device)
        # Podpora pre celú štruktúru checkpointu aj samotný state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint úspešne načítaný.")
    else:
        print(f"Upozornenie: {cfg.base_checkpoint_path} nebol nájdený. Trénujem bez predchádzajúceho stavu.")
    
    model.to(device)

    # 3. Dáta
    dataset = ConversationDataset(
        conversation_dirs=cfg.conversation_dirs,
        personality_dirs=cfg.personality_dirs,
        tokenizer=tokenizer,
        max_len=cfg.max_seq_len
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_conversation_fn)

    # 4. Optimizer a Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = CrossEntropyLoss(ignore_index=-100)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.train()

    # 5. Trénovací cyklus
    print(f"Spúšťam Conversation Training pre {len(dataset)} konverzácií...")
    global_step = 0

    for epoch in range(cfg.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            
            logits = model(input_ids)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                print(f"Epoch: {epoch+1}/{cfg.epochs} | Step: {step} | Loss: {loss.item():.4f}")

            if global_step % cfg.save_every_steps == 0:
                save_path = os.path.join(cfg.output_dir, f"conv_checkpoint_step_{global_step}.pt")
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_path)
                print(f"Uložený checkpoint: {save_path}")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"--- Priemerná Loss za epochu {epoch+1}: {avg_loss:.4f} ---")

    # Finálne uloženie
    final_path = os.path.join(cfg.output_dir, "final_conversation_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Konverzačný tréning dokončený! Model bol uložený v: {final_path}")

if __name__ == "__main__":
    train_conversation_tuning()