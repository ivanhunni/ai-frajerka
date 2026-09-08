import torch
import torch.nn as nn

class CrossEntropyLossWithIgnore(nn.Module):
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Preusporiadanie tvaru pre PyTorch CrossEntropyLoss: [batch_size * seq_len, vocab_size]
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        return self.criterion(logits, targets)