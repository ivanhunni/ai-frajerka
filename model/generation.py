import torch
import torch.nn.functional as F
from .transformer import Transformer

@torch.no_grad()
def generate(
    model: Transformer, 
    idx: torch.Tensor, 
    max_new_tokens: int, 
    temperature: float = 1.0, 
    top_k: int = None
) -> torch.Tensor:
    """
    Funkcia na postupnú generáciu nových tokenov (Autoregressive inference).
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Orezanie kontextu, ak presahuje max_seq_len
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]
        
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx