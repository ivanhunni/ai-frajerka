# alignment/config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class AlignmentConfig:
    # Cesty k modelom
    conversation_checkpoint_path: str = "checkpoints/conversation_model/final_conversation_model.pt"
    output_dir: str = "checkpoints/aligned_model"
    
    # Cesty k dátam
    alignment_dirs: List[str] = field(default_factory=lambda: [
        "dataset/evaluation/conversation_tests",
        "dataset/evaluation/personality_tests"
    ])

    # DPO Hyperparametre
    beta: float = 0.1             # Koeficient kontroly odchýlky od referenčného modelu
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 5e-7   # Extra nízka LR pre DPO
    weight_decay: float = 0.01
    max_seq_len: int = 512
    
    # HW & Nastavenia
    device: str = "cuda"
    save_every_steps: int = 200