from dataclasses import dataclass
from typing import Optional


@dataclass
class PretrainingConfig:
    # Data paths
    data_dir: str = "dataset/pretraining"
    tokenizer_path: str = "tokenizer/tokenizer.json"
    output_dir: str = "checkpoints/pretraining"

    # Training Hyperparameters
    batch_size: int = 16
    sequence_length: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    save_every_steps: int = 5000

    # Optimization
    grad_clip: float = 1.0
    device: str = "cuda"  # "cuda", "mps", or "cpu"