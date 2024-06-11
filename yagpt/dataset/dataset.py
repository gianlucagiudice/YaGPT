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

    def tokenize(self, text: str):
        return self.tokenize_helper(text, self.token_to_id)

    def untokenize(self, tokens: List[int]) -> str:
        return self.untokenize_helper(tokens, self.id_to_token)

    @classmethod
    def tokenize_helper(cls, text: str, token_to_id: dict[int, int]) -> list[int]:
        tokens = cls.tokenizer.encode(text)
        tokens = [token_to_id[t] for t in tokens]
        return tokens

    @classmethod
    def untokenize_helper(cls, tokens: list[int], id_to_token: dict[int, int]) -> str:
        tokens = [id_to_token[t] for t in tokens]
        text = cls.tokenizer.decode(tokens)
        return text

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + 1 + self.seq_len]
        x, y = torch.tensor(x).long(), torch.tensor(y).long()

        return x, y


if __name__ == '__main__':
    from pathlib import Path
    dataset_path = Path(__file__).parent.parent.parent / 'dataset' / 'divina_commedia'
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
          f"\tY:\n```text\n\t{dataset.untokenize(yy)}\n```\n"
          )
