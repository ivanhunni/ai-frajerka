import torch
from torch.utils.data import DataLoader
from model import Transformer, TransformerConfig
from training import TextDataset, CrossEntropyLossWithIgnore, create_optimizer, SaveCheckpoint

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používa sa zariadenie: {device}")

    # 1. Konfigurácia a Model
    config = TransformerConfig(vocab_size=1000, max_seq_len=64, d_model=128, n_heads=4, n_layers=4)
    model = Transformer(config).to(device)

    # 2. Dataset a DataLoader (načíta dáta z tvojho priečinka dataset/)
    dataset = TextDataset(data_dir="dataset", tokenizer_path="tokenizer/tokenizer.json", max_seq_len=config.max_seq_len)
    if len(dataset) == 0:
        print("Dataset je prázdny! Skontrolujte súbory v dataset/ alebo vytrénujte tokenizer.")
        return

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Optimizer a Loss
    optimizer = create_optimizer(model, lr=1e-3)
    criterion = CrossEntropyLossWithIgnore(ignore_index=config.pad_token_id)

    # 4. Tréningový cyklus
    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epocha {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # 5. Uloženie modelu
    SaveCheckpoint(model, optimizer, epochs, avg_loss, "checkpoint.pt")

if __name__ == "__main__":
    train()