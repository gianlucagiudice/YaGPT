from typing import Optional, Union, Tuple

import fire
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from yagpt.callback import TrainingGenerationCallback
from yagpt.dataset import YaDataset
from yagpt.model import YaGPTWrapper, YaGPTConfig
from yagpt.tokenizer import AbstractTokenizer, tokenizer_factory


def init_dataset(dataset_dir_path: str, batch_size: int, seq_len: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = YaDataset(dataset_dir_path, 'train', seq_len)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=8, persistent_workers=True
    )

    val_dataset = YaDataset(dataset_dir_path, 'val', seq_len)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=8, persistent_workers=True
    )
    return train_loader, val_loader


def init_tokenizer(tokenizer_name: str, tokenizer_path: Optional[str], dataset_dir_path: str) -> AbstractTokenizer:
    if tokenizer_path is None:
        import os
        for file in os.listdir(dataset_dir_path):
            if file.endswith(".model"):
                tokenizer_path = os.path.join(dataset_dir_path, file)
                break

        if tokenizer_path is None:
            raise FileNotFoundError("No file with extension '.model' found in the dataset directory.")

    tokenizer = tokenizer_factory(tokenizer_name, tokenizer_path)
    return tokenizer


def train(
        # Dataset
        dataset_dir_path: str,
        tokenizer_name: str = 'bpe',
        tokenizer_path: Optional[str] = None,
        # Model parameters
        batch_size: int = 64,
        d_model: int = 336,
        seq_len: int = 160,
        n_heads: int = 6,
        n_layers: int = 6,
        dff_factor: int = 4,
        dropout: float = 0.1,
        # Training parameters
        max_epochs: int = 10,
        max_steps: int = -1,
        overfit_batches: Union[int, float] = 0.0,
        accelerator: str = 'auto',
        val_check_interval: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        log_every_n_steps: int = 5,
        early_stopping_patience: int = 3,
        min_delta: float = 0.01,
        metric_to_monitor: str = 'train_loss',
        # Optimizer parameters
        lr: float = 1e-3,
        scheduler_t0: int = 150,
        scheduler_t_mult: int = 1,
        gradient_clip_val: float = 0.5,
        # Generation parameters
        n_samples: int = 4,
        autoregressive_steps: int = 32,
        generation_top_k: int = 5,
        temperature: float = 1.25,
):
    # Load datasets
    train_loader, val_loader = init_dataset(dataset_dir_path, batch_size, seq_len)

    # Load tokenizer
    tokenizer = init_tokenizer(tokenizer_name, tokenizer_path, dataset_dir_path)

    # Initialize model
    model_config = YaGPTConfig(
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_model * dff_factor,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        vocab_size=tokenizer.vocab_size
    )

    model = YaGPTWrapper(
        model_config,
        lr=lr,
        scheduler_t0=scheduler_t0,
        scheduler_t_mult=scheduler_t_mult,
    )

    # Train model
    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        logger=lightning.pytorch.loggers.WandbLogger(project="YaGPT", log_model='all'),
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        overfit_batches=overfit_batches,
        gradient_clip_val=gradient_clip_val,
        callbacks=[
            ModelCheckpoint(dirpath='checkpoints', monitor=metric_to_monitor, mode='min'),
            EarlyStopping(monitor=metric_to_monitor, patience=early_stopping_patience, mode='min', min_delta=min_delta),
            LearningRateMonitor(logging_interval='step'),
            TrainingGenerationCallback(
                n_samples=n_samples, autoregressive_steps=autoregressive_steps,
                top_k=generation_top_k, temperature=temperature, tokenizer=tokenizer
            ),
        ],
        accelerator=accelerator,
        detect_anomaly=True
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    fire.Fire(train)
