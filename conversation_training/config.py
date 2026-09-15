# conversation_training/config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class ConversationTrainingConfig:
    # Cesty k dátam a checkpointom
    # Môžete nadviazať na model z instruction trainingu alebo base pretrained model
    base_checkpoint_path: str = "checkpoints/instruction_model/final_instruction_model.pt"
    output_dir: str = "checkpoints/conversation_model"
    
    conversation_dirs: List[str] = field(default_factory=lambda: [
        "dataset/instruction/conversation"
    ])
    personality_dirs: List[str] = field(default_factory=lambda: [
        "dataset/personality"
    ])

    # Parametre trénovania
    epochs: int = 5
    batch_size: int = 2
    learning_rate: float = 1e-5  # Nižšia učiaca rýchlosť pre jemné dolaďovanie dialógu
    weight_decay: float = 0.01
    max_seq_len: int = 512       # Dlhší kontext pre uloženie histórie správ
    
    # HW & Nastavenia
    device: str = "cuda"
    save_every_steps: int = 300