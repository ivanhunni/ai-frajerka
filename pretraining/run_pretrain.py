import torch
from tokenizers import Tokenizer
from model.config import ModelConfig
from model.transformer import Transformer
from pretraining.config import PretrainingConfig
from pretraining.dataset import PretrainingDataset
from pretraining.trainer import Pretrainer


def main():
    # 1. Konfigurácie
    pretrain_cfg = PretrainingConfig()
    model_cfg = ModelConfig()

    # Nastavenie zariadenia (CUDA / MPS / CPU)
    if torch.cuda.is_available():
        pretrain_cfg.device = "cuda"
    elif torch.backends.mps.is_available():
        pretrain_cfg.device = "mps"
    else:
        pretrain_cfg.device = "cpu"

    print(f"Using device: {pretrain_cfg.device}")

    # 2. Načítanie Tokenizeru
    tokenizer = Tokenizer.from_file(pretrain_cfg.tokenizer_path)

    # 3. Inicializácia modelu a dát
    model = Transformer(model_cfg)
    dataset = PretrainingDataset(
        data_dir=pretrain_cfg.data_dir,
        tokenizer=tokenizer,
        seq_length=pretrain_cfg.sequence_length
    )

    # 4. Spustenie pretréningu
    trainer = Pretrainer(model=model, dataset=dataset, config=pretrain_cfg)
    trainer.train()


if __name__ == "__main__":
    main()