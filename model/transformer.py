import torch
import torch.nn as nn
from .config import TransformerConfig
from .embeddings import Embeddings
from .transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()
        self.config = config or TransformerConfig()
        self.embeddings = Embeddings(self.config)
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.norm = nn.LayerNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        mask = self.generate_causal_mask(seq_len, x.device)

        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        
        logits = self.head(x)
        return logits