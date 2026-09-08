import os
import torch

def SaveCheckpoint(model, optimizer, epoch: int, loss: float, filepath: str = "checkpoint.pt"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint bol uložený do '{filepath}'.")

def LoadCheckpoint(filepath: str, model, optimizer=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint na adrese '{filepath}' neexistuje.")

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint načítaný z epochy {checkpoint.get('epoch', 'neznáma')}.")
    return checkpoint.get("epoch", 0), checkpoint.get("loss", 0.0)