import lightning
import torch

from yagpt.model import YaGPTConfig, YaGPT
from yagpt.tokenizer import AbstractTokenizer


class YaGPTWrapper(lightning.LightningModule):
    def __init__(
            self,
            config: YaGPTConfig,
            lr: float = 1e-3,
            scheduler_t0: int = 300,
            scheduler_t_mult: int = 1,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = YaGPT(config)
        self.lr = lr
        self.scheduler_t0 = scheduler_t0
        self.scheduler_t_mult = scheduler_t_mult
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        y_pred = logits.view(-1, logits.size(-1))
        y_target = y.view(-1)
        loss = torch.nn.functional.cross_entropy(y_pred, y_target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def generate_text(
            self,
            prompt: str,
            tokenizer: AbstractTokenizer,
            n_steps: int = 200,
            temperature: float = 1.5,
            top_k: int = 5,
    ) -> str:
        tokens = tokenizer.encode(prompt)
        batch = torch.tensor(tokens).unsqueeze(0).long()
        for token in self.model.generate_text(batch, n_steps, temperature=temperature, top_k=top_k):
            predicted_token = tokenizer.decode([token])
            yield predicted_token
