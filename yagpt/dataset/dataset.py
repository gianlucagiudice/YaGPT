import os
from typing import Literal, Tuple, List

import tiktoken
import torch
from torch.utils.data import Dataset


class YaDataset(Dataset):
    tokenizer = tiktoken.get_encoding('gpt2')

    def __init__(
            self,
            data_dir: str,
            split: Literal['train', 'val'],
            seq_len: int,
    ):
        self.tokens = self._read_tokens_split(data_dir, split)
        self.vocab_size = self.tokenizer.n_vocab
        self.seq_len = seq_len

    @staticmethod
    def _read_tokens_split(data_dir, split):
        match split:
            case 'train':
                file_path = os.path.join(data_dir, 'train.bin')
            case 'val':
                file_path = os.path.join(data_dir, 'val.bin')
            case _:
                raise ValueError(f'split must be either "train" or "val", got {split}')
        tokens = YaDataset.load_from_bin(file_path)
        return tokens

    @staticmethod
    def load_from_bin(file_path):
        tokens = torch.load(file_path)
        return tokens

    @classmethod
    def tokenize(cls, text):
        tokens = cls.tokenizer.encode(text)
        return tokens

    @classmethod
    def untokenize(cls, tokens):
        text = cls.tokenizer.decode(tokens)
        return text

    @staticmethod
    def collate_fn(batch: List[Tuple[list[int], list[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = torch.tensor(batch).long()
        x, y = batch[:, 0], batch[:, 1]
        return x, y

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx) -> Tuple[list[int], list[int]]:
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + 1 + self.seq_len]

        return x, y


if __name__ == '__main__':
    from pathlib import Path
    dataset_path = Path(__file__).parent.parent.parent / 'dataset' / 'divina_commedia'
    dataset_path = str(dataset_path)

    dataset = YaDataset(dataset_path, 'train', 128)

    tokens_sample = dataset[0]

    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Tokens sample ({len(tokens_sample[0])}):\n"
          f"\tTrain:\t{tokens_sample[0][:16]} ..."
          f"\n\tVal:\t{tokens_sample[1][:16]} ...")
