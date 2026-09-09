# inference/engine.py
import os
import torch
from model.transformer import Transformer
from tokenizer.tokenizer import Tokenizer
from inference.sampler import sample_logits

class InferenceEngine:
    def __init__(self, model_path="checkpoints/aligned_model/final_aligned_model.pt", 
                 tokenizer_path="tokenizer/tokenizer.json", device=None):
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Inference Engine beží na: {self.device}")

        # Načítanie Tokenizeru
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Načítanie Modelu
        self.model = Transformer()
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            print("Model pre inference bol úspešne načítaný.")
        else:
            print(f"Upozornenie: Soubor {model_path} nebol nájdený! Model funguje s náhodnými váhami.")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_reply(self, messages, system_prompt="Si milá a starostlivá AI priateľka.", 
                       max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9):
        """
        Vygeneruje odpoveď na základe konverzačnej histórie.
        """
        # Skladanie promptu
        formatted_prompt = f"<|system|>\n{system_prompt}\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_prompt += f"<|{role}|>\n{content}\n"
        
        formatted_prompt += "<|assistant|>\n"

        # Tokenizácia
        input_ids = self.tokenizer.encode(formatted_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = []

        # Generovací cyklus
        for _ in range(max_new_tokens):
            # Orezanie kontextu ak je dlhší než kapacita modelu (napr. max 1024 tokenov)
            context_tensor = input_tensor[:, -1024:]
            
            logits = self.model(context_tensor)
            next_token_logits = logits[:, -1, :]

            # Výber nasledujúceho tokenu
            next_token = sample_logits(next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            # Stop podmienka (ak model vygeneruje end-of-sequence token alebo začiatok novej roly)
            token_id = next_token.item()
            generated_ids.append(token_id)
            
            # Pridanie nového tokenu do vstupu
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

            # Dekódovanie priebežného textu pre detekciu stop slov
            decoded_text = self.tokenizer.decode(generated_ids)
            if "<|user|>" in decoded_text or "<|system|>" in decoded_text or "<|end|>" in decoded_text:
                break

        # Vyčistenie výstupného textu
        reply = self.tokenizer.decode(generated_ids)
        reply = reply.split("<|user|>")[0].split("<|system|>")[0].replace("<|end|>", "").strip()
        
        return reply