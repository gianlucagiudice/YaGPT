import random
import re
from pathlib import Path
from typing import Tuple, Literal, List

import tiktoken
import torch
from torch.utils.data import Dataset


def collate_fn(batch: List[Tuple[list[int], list[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.tensor(batch).long()
    x, y = batch[:, 0], batch[:, 1]
    return x, y


class DivinaCommediaDataset(Dataset):

    tokenizer = tiktoken.get_encoding('gpt2')

    def __init__(
            self,
            dataset_path: str,
            seq_len: int,
            split: Literal['train', 'val'],
            lower_case: bool = False,
            train_ratio: float = 0.9,
            shuffle_sections: bool = True
    ):
        if split not in ['train', 'val']:
            raise ValueError(f'split must be either "train" or "val", got {split}')

        self.seq_len = seq_len
        self.lower_case = lower_case
        self.shuffle_sections = shuffle_sections

        self.raw_text = self.read_dataset(dataset_path, lower_case, shuffle_sections)
        self.tokens = self.tokenize(self.raw_text)
        self.split_tokens = self.split_dataset(self.tokens, train_ratio, split)

        tokens_set = set(sorted(list(set(self.tokens))))
        self.token2idx = {token: idx for idx, token in enumerate(tokens_set)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)

    def tokenize(self, text):
        tokens = self.tokenizer.encode(text)
        return tokens

    def untokenize(self, tokens):
        text = self.tokenizer.decode(tokens)
        return text

    @staticmethod
    def read_dataset(dataset_path: str, lower_case: bool, shuffle_sections: bool) -> str:
        with open(dataset_path, 'r') as f:
            text = f.read()
        if shuffle_sections:
            sections = re.split(r'\n{3}', text)
            sections = list(map(str.strip, sections))
            random.shuffle(sections)
            text = '\n\n\n'.join(sections)
        if lower_case:
            text = text.lower()
        return text

    @staticmethod
    def split_dataset(tokens: List[int], p: float, split: Literal['train', 'val']) -> List[int]:
        if p <= 1:
            split_idx = round(len(tokens) * p)
        else:
            split_idx = p

        train_text, val_text = tokens[:split_idx], tokens[split_idx:]
        split = train_text if split == 'train' else val_text
        return split

    def __len__(self):
        return len(self.split_tokens) - self.seq_len - 1

    def __getitem__(self, idx) -> Tuple[list[int], list[int]]:
        x = self.split_tokens[idx: idx + self.seq_len]
        y = self.split_tokens[idx + 1: idx + 1 + self.seq_len]

        x = [self.token2idx[token] for token in x]
        y = [self.token2idx[token] for token in y]

        return x, y


if __name__ == '__main__':
    dataset_path = str(Path(__file__).parent.parent.parent / 'dataset' / 'inferno.txt')

    dataset = DivinaCommediaDataset(dataset_path, 128, 'train')
    print(len(dataset))
    tokens = dataset[len(dataset)]
    print(tokens)
    print(len(tokens[0]))
    print(dataset.token2idx)
    print(dataset.idx2token)
    print(dataset.vocab_size)
