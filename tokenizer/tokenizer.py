from pathlib import Path
from tokenizers import Tokenizer

class CustomTokenizer:
    def __init__(self, model_path="tokenizer/tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(model_path)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("[PAD]") or 0

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]):
        return self.tokenizer.decode(ids)
