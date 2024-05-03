from typing import Tuple, List

from torch.utils.data import Dataset
import torch


def collate_fn(batch: List[Tuple[list[int], list[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.tensor(batch).long()
    x, y = batch[:, 0], batch[:, 1]
    return x, y


class DummyDataset(Dataset):
    def __init__(self, seq_len: int, max_value: int):
        self.seq_len = seq_len
        self.max_value = max_value

    def __len__(self):
        return self.max_value - self.seq_len

    def __getitem__(self, idx) -> Tuple[list[int], list[int]]:
        x = list(range(idx, idx + self.seq_len))
        y = list(range(idx + 1, idx + 1 + self.seq_len))
        return x, y
