from typing import Optional

import fire
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from yagpt.utils import model_factory
from yagpt.callback import TrainingGenerationCallback
from yagpt.dataset import DivinaCommediaDataset, collate_fn
from yagpt.model import YaGPTWrapper


def main(
        dataset_path: str,
        batch_size: int = 16,
        d_model: int = 512,
        seq_len: int = 192,
        n_heads: int = 8,
        n_layers: int = 12,
        train_ratio: float = 0.9,
        dropout: float = 0.1,
        n_epochs: int = 10,
        val_check_interval: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        log_every_n_steps: int = 5,
):
    # Load datasets
    train_dataset = DivinaCommediaDataset(dataset_path, seq_len, 'train', train_ratio=train_ratio)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=True, num_workers=8, persistent_workers=True
    )

    val_dataset = DivinaCommediaDataset(dataset_path, seq_len, 'val', train_ratio=train_ratio)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=False, num_workers=8, persistent_workers=True
    )

    vocab_size = train_dataset.vocab_size
    model: YaGPTWrapper = model_factory(d_model, seq_len, n_heads, n_layers, dropout, vocab_size)

    # Train model
    trainer = lightning.Trainer(
        max_epochs=n_epochs,
        log_every_n_steps=log_every_n_steps,
        logger=lightning.pytorch.loggers.WandbLogger(project="YaGPT", log_model='all'),
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        callbacks=[
            lightning.pytorch.callbacks.ModelCheckpoint(dirpath='checkpoints', monitor='val_loss', mode='min'),
            TrainingGenerationCallback(n_samples=4, autoregressive_steps=32),
            lightning.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        ],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    fire.Fire(main)
