from typing import Optional

import fire
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from yagpt.callback import TrainingGenerationCallback
from yagpt.dataset import DivinaCommediaDataset, collate_fn
from yagpt.model import YaGPTWrapper, YaGPTConfig


def train(
        # Dataset
        dataset_path: str,
        # Model parameters
        batch_size: int = 16,
        d_model: int = 512,
        seq_len: int = 192,
        n_heads: int = 8,
        n_layers: int = 12,
        train_ratio: float = 0.9,
        dropout: float = 0.2,
        # Training parameters
        max_epochs: int = 10,
        max_steps: int = -1,
        accelerator: str = 'auto',
        val_check_interval: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        log_every_n_steps: int = 5,
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 3,
        # Optimizer parameters
        lr: float = 1e-3,
        scheduler_t0: int = 300,
        scheduler_t_mult: int = 1,
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

    model_config = YaGPTConfig(
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_model * 4,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        vocab_size=train_dataset.vocab_size
    )

    model = YaGPTWrapper(model_config, lr=lr, scheduler_t0=scheduler_t0, scheduler_t_mult=scheduler_t_mult)

    # Train model
    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        logger=lightning.pytorch.loggers.WandbLogger(project="YaGPT", log_model='all'),
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        callbacks=[
            ModelCheckpoint(dirpath='checkpoints', monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min'),
            LearningRateMonitor(logging_interval='step'),
            TrainingGenerationCallback(n_samples=4, autoregressive_steps=32),
        ],
        accelerator=accelerator,
        gradient_clip_val=gradient_clip_val,
        detect_anomaly=True
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    fire.Fire(train)
