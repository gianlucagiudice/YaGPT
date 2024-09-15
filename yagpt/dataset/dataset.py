import os
from typing import Literal, Tuple, List

import torch
from torch.utils.data import Dataset

from yagpt.tokenizer import AbstractTokenizer, tokenizer_factory


class YaDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            split: Literal['train', 'val'],
            seq_len: int,
            tokenizer_name: Literal['gpt2', 'char'] = 'char'
    ):
        self.tokenizer: AbstractTokenizer = tokenizer_factory(tokenizer_name)
        self.tokens, self.id_to_token, self.token_to_id = self._read_split(data_dir, split)
        self.vocab_size = len(self.id_to_token)
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
        data = torch.load(file_path)
        tokens, id_to_token, token_to_id = data['tokens'], data['id_to_token'], data['token_to_id']
        return tokens, id_to_token, token_to_id

    def tokenize(self, text: str) -> List[int]:
        encoded = self.tokenizer.encode(text)
        return encoded

    def untokenize(self, tokens: List[int]) -> str:
        decoded = self.tokenizer.decode(tokens)
        return decoded

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
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"\nTokens sample ({len(tokens_sample[0])}):\n"
          f"\tX:\t{xx} ...\n"
          f"\tY:\t{yy} ...\n"
          f"Untokenized sample:\n"
          f"\tX:\n```text\n\t{dataset.untokenize(xx)}\n```\n"
          f"\tY:\n```text\n\t{dataset.untokenize(yy)}\n```\n")
