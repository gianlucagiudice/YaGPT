from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from yagpt import model_factory
from yagpt.dataset.divina_commedia import DivinaCommediaDataset, collate_fn
from yagpt.model import YaGPT


def main(
        dataset_path: str,
        batch_size: int,
        d_model: int,
        seq_len: int,
        n_heads: int,
        n_layers: int,
        train_ratio: float = 0.7,
        dropout: float = 0.1,
        n_epochs: int = 50,
        val_check_interval: int = 500,
        limit_val_batches: int = 30,
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
    model: YaGPT = model_factory(d_model, seq_len, n_heads, n_layers, dropout, vocab_size)

    # Train model
    trainer = L.Trainer(
        max_epochs=n_epochs,
        logger=L.pytorch.loggers.WandbLogger(project="YaGPT", log_model='all'),
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(save_last=True),
            L.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        ],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    dataset_path = str(Path(__file__).parent / 'dataset' / 'inferno.txt')
    main(
        dataset_path=dataset_path,
        batch_size=16,
        d_model=512,
        seq_len=512,
        n_heads=4,
        n_layers=8,
        dropout=0.1,
        train_ratio=0.7,
    )
