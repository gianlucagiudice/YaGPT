import lightning
import torch

from yagpt.model import YaGPTConfig, YaGPT


class YaGPTWrapper(lightning.LightningModule):
    def __init__(self, config: YaGPTConfig, lr: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = YaGPT(config)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        logits = logits.transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(logits, y)
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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
