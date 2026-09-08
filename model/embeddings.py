import torch
import torch.nn as nn
from .config import TransformerConfig

class Embeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        token_embeddings = self.tok_embed(x)
        position_embeddings = self.pos_embed(positions)
        
        return self.dropout(token_embeddings + position_embeddings)