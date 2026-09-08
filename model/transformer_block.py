import torch
import torch.nn as nn
from .config import TransformerConfig
from .attention import MultiHeadAttention
from .mlp import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Residual connections s LayerNorm (Pre-LN štruktúra)
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x