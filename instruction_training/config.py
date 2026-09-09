# instruction_training/config.py
from dataclasses import dataclass

@dataclass
class InstructionTrainingConfig:
    # Cesty k dátam a checkpointom
    pretrained_checkpoint_path: str = "checkpoints/pretrained_model.pt"
    output_dir: str = "checkpoints/instruction_model"
    data_dirs: list = None

    # Parametre trénovania (SFT býva kratšie a s nižšou learning rate)
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_len: int = 512
    
    # HW & Nastavenia
    device: str = "cuda"  # "cuda" alebo "cpu"
    save_every_steps: int = 500

    def __post_init__(self):
        if self.data_dirs is None:
            self.data_dirs = [
                "dataset/instruction/conversation",
                "dataset/instruction/question_answer",
                "dataset/instruction/reasoning"
            ]