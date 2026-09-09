# inference/sampler.py
import torch
import torch.nn.functional as F

def sample_logits(logits, temperature=0.7, top_k=40, top_p=0.9):
    """
    Aplikuje Temperature, Top-K a Top-P (Nucleus) filter na logity
    a vráti index vybraného tokenu.
    """
    # 1. Aplikovanie Temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Ak je temperature 0, vyberie sa najpravdepodobnejší token (Greedy search)
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Top-K Orezanie
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    # 3. Top-P (Nucleus) Orezanie
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Odstránenie tokenov s kumulatívnou pravdepodobnosťou nad top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Posun vpravo, aby sme si ponechali aj prvý token nad prahom
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = -float('Inf')

    # 4. Výpočet pravdepodobností a výber tokenu
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token