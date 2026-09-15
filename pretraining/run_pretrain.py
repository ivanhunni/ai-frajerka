import torch
from tokenizers import Tokenizer
from model.config import TransformerConfig
from model.transformer import Transformer
from pretraining.config import PretrainingConfig
from pretraining.dataset import PretrainingDataset
from pretraining.trainer import Pretrainer

def main():
    pretrain_cfg = PretrainingConfig()
    if torch.cuda.is_available():
        pretrain_cfg.device = "cuda"
    elif torch.backends.mps.is_available():
        pretrain_cfg.device = "mps"
    else:
        pretrain_cfg.device = "cpu"

    tokenizer = Tokenizer.from_file(pretrain_cfg.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    model_cfg = TransformerConfig(vocab_size=vocab_size, max_seq_len=pretrain_cfg.sequence_length)
    model = Transformer(model_cfg)
    dataset = PretrainingDataset(pretrain_cfg.data_dir, tokenizer, pretrain_cfg.sequence_length)
    if len(dataset) == 0:
        raise RuntimeError("Pretraining dataset je prázdny alebo kratší než sequence_length.")
    print(f"Using device: {pretrain_cfg.device}; vocab_size={vocab_size}; samples={len(dataset)}")
    Pretrainer(model, dataset, pretrain_cfg).train()

if __name__ == "__main__":
    main()
