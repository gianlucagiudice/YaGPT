import re
from pathlib import Path
from typing import Tuple, Literal

from nltk import word_tokenize
from torch.utils.data import Dataset


class DivinaCommediaDataset(Dataset):

    def __init__(
            self,
            dataset_path: str,
            seq_len: int,
            split: Literal['train', 'val'],
            train_ratio: float = 0.7,
    ):
        if split not in ['train', 'val']:
            raise ValueError(f'split must be either "train" or "val", got {split}')

        self.full_text, self.split = self.read_dataset(dataset_path, train_ratio, split)
        self.tokens = word_tokenize(self.split, language='italian')
        self.seq_len = seq_len

        self.token2idx = {token: idx for idx, token in enumerate(set(self.tokens))}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)

    @staticmethod
    def read_dataset(
            dataset_path: str,
            train_ratio: float,
            split: Literal['train', 'val'],
    ) -> Tuple[str, str]:
        with open(dataset_path, 'r') as f:
            text = f.read()

        train, val = DivinaCommediaDataset.split_dataset(text, train_ratio)
        split = train if split == 'train' else val

        return text, split

    @staticmethod
    def split_dataset(text: str, p: float) -> Tuple[str, str]:
        chapters = list(re.finditer('CANTO', text))
        train_chapters_idx = round(len(chapters) * p)
        train_text_idx = chapters[train_chapters_idx].regs[0][0]
        train_text, val_text = text[:train_text_idx], text[train_text_idx:]

        return train_text, val_text

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx) -> list[str]:
        tokens = self.tokens[idx: idx + self.seq_len]
        return tokens


if __name__ == '__main__':
    dataset_path = str(Path(__file__).parent.parent / 'dataset' / 'inferno.txt')

    dataset = DivinaCommediaDataset(dataset_path, 512, 'train')
    print(dataset[0])
    print(dataset.token2idx)
    print(dataset.idx2token)
    print(dataset.vocab_size)
