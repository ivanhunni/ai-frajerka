import os
import torch
from tokenizers import Tokenizer
from model.config import TransformerConfig
from model.transformer import Transformer
from inference.sampler import sample_logits

class InferenceEngine:
    def __init__(self, model_path="checkpoints/aligned_model/final_aligned_model.pt",
                 tokenizer_path="tokenizer/tokenizer.json", device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.model = Transformer(TransformerConfig(vocab_size=self.tokenizer.get_vocab_size(), max_seq_len=512))

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model neexistuje: {model_path}. Najprv spusti tréningové kroky alebo nastav správny checkpoint."
            )
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def generate_reply(self, messages, system_prompt="Si milá a starostlivá AI priateľka.",
                       max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9):
        prompt = f"<|system|>\n{system_prompt}\n"
        for msg in messages:
            role = msg.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            prompt += f"<|{role}|>\n{msg.get('content', '')}\n"
        prompt += "<|assistant|>\n"

        ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = []
        max_context = self.model.config.max_seq_len

        for _ in range(max_new_tokens):
            context = input_tensor[:, -max_context:]
            logits = self.model(context)[:, -1, :]
            next_token = sample_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            token_id = next_token.item()
            generated.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            text = self.tokenizer.decode(generated)
            if "<|user|>" in text or "<|system|>" in text or "<|end|>" in text:
                break

        return self.tokenizer.decode(generated).replace("<|end|>", "").strip()
