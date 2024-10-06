import os
from typing import Literal, Tuple

import torch
from torch.utils.data import Dataset


class YaDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            split: Literal['train', 'val'],
            seq_len: int,
    ):
        self.tokens = self._read_split(data_dir, split)
        self.seq_len = seq_len

    @staticmethod
    def _read_split(data_dir, split):
        match split:
            case 'train':
                file_path = os.path.join(data_dir, 'train.bin')
            case 'val':
                file_path = os.path.join(data_dir, 'val.bin')
            case _:
                raise ValueError(f'split must be either "train" or "val", got {split}')
        tokens = torch.load(file_path)
        return tokens

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + 1 + self.seq_len]
        x, y = torch.tensor(x).long(), torch.tensor(y).long()

        return x, y


if __name__ == '__main__':
    from pathlib import Path

    dataset_path = Path(__file__).parent.parent.parent / 'dataset' / 'divina_commedia' / 'processed'
    dataset_path = str(dataset_path)

    dataset = YaDataset(dataset_path, 'train', 128)

    id_sample = 128
    tokens_sample = dataset[id_sample]

    n_tokens = 8
    xx = tokens_sample[0][:n_tokens].tolist()
    yy = tokens_sample[1][:n_tokens].tolist()

    print(f"Dataset size: {len(dataset)}")
    print(f"\nTokens sample ({len(tokens_sample[0])}):\n"
          f"\tX:\t{xx} ...\n"
          f"\tY:\t{yy} ...\n")
