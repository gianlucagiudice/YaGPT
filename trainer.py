from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from yagpt import model_factory
from yagpt.dataset.divina_commedia import DivinaCommediaDataset, collate_fn
from yagpt.model import YaGPT


@torch.no_grad()
def autoregressive_prediction(
        model: torch.nn.Module,
        validation_set: torch.utils.data.DataLoader,
        device: torch.device,
        idx2token: dict,
        n_samples: int = 4,
        autoregressive_steps: int = 64,
):
    model.eval()
    model.to(device)

    for i, batch in enumerate(validation_set):
        generated_text = []
        x, _ = batch
        x = x.to(device)

        x_original = x.clone()

        for step in range(autoregressive_steps):
            pred = model(x)

            pred_id = torch.argmax(pred[0, -1, :]).item()
            generated_text.append(idx2token[pred_id])

            last_id = torch.tensor(pred_id).unsqueeze(dim=0).to(device)
            x = torch.cat([x[0, 1:], last_id]).unsqueeze(dim=0)

        print(f'\n\nORIGINAL >>>\n'
              f'{''.join([idx2token[token_id] for token_id in x_original.flatten().tolist()])}'
              f'\n'
              f'GENERATED >>>\n'
              f'{''.join(generated_text)}')

        if i == n_samples:
            break


def main(
        dataset_path: str,
        batch_size: int,
        d_model: int,
        seq_len: int,
        n_heads: int,
        n_layers: int,
        train_ratio: float = 0.7,
        dropout: float = 0.1,
        n_epochs: int = 100,
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
        val_check_interval=500,
        limit_val_batches=30,
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
        seq_len=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        train_ratio=0.7
    )
