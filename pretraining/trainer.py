import os
import torch
from torch.utils.data import DataLoader
from pretraining.config import PretrainingConfig


class Pretrainer:
    def __init__(self, model, dataset: torch.utils.data.Dataset, config: PretrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        step = 0
        os.makedirs(self.config.output_dir, exist_ok=True)

        while step < self.config.max_steps:
            for x, y in self.dataloader:
                if step >= self.config.max_steps:
                    break

                x = x.to(self.config.device)
                y = y.to(self.config.device)

                self.optimizer.zero_grad()
                logits = self.model(x)

                # Reshape pre výpočet Cross Entropy stratovej funkcie
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                step += 1

                if step % 100 == 0:
                    print(f"Step {step}/{self.config.max_steps} | Loss: {loss.item():.4f}")

                if step % self.config.save_every_steps == 0:
                    checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_step_{step}.pt")
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

        # Uloženie finálneho modelu
        final_path = os.path.join(self.config.output_dir, "pretrained_model.pt")
        torch.save(self.model.state_dict(), final_path)
        print(f"Pretraining complete. Model saved to {final_path}")