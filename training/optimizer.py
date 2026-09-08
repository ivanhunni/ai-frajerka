import torch
import torch.nn as nn
from torch.optim import AdamW

def create_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.01):
    # Oddelenie parametrov: weight decay neaplikujeme na bias a LayerNorm
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(optim_groups, lr=lr)