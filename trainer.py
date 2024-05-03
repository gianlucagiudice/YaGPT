from pathlib import Path
import torch.optim

from yagpt import model_factory
from torch.utils.data import DataLoader
from yagpt.dataset.divina_commedia import DivinaCommediaDataset, collate_fn
from tqdm import tqdm


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
        device: str,
        dataset_path: str,
        batch_size: int,
        d_model: int,
        seq_len: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        n_epochs: int = 100,
):
    # Load datasets
    train_dataset = DivinaCommediaDataset(dataset_path, seq_len, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    val_dataset = DivinaCommediaDataset(dataset_path, seq_len, 'val')
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    # Device
    device = torch.device(device)

    # Init model
    vocab_size = train_dataset.vocab_size
    model = model_factory(d_model, seq_len, n_heads, n_layers, dropout, vocab_size)
    model.train()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())

    pbar_epochs = tqdm(range(n_epochs), position=0)
    for epoch in pbar_epochs:
        pbar_batch = tqdm(train_loader, total=len(train_loader), position=1)
        for i, batch in enumerate(pbar_batch):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(x)

            # Compute softmax
            logits = outputs.transpose(1, 2)

            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            pbar_batch.set_description(f'Loss {loss.item()}')

        autoregressive_prediction(model, val_loader, device, train_dataset.idx2token)


if __name__ == '__main__':
    dataset_path = str(Path(__file__).parent / 'dataset' / 'inferno.txt')
    main(
        dataset_path=dataset_path,
        device='mps',
        batch_size=32,
        d_model=512,
        seq_len=768,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
    )
