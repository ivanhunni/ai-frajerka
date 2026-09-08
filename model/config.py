from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int = 5000
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    pad_token_id: int = 0