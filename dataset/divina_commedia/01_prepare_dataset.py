import os
import random
import re
from typing import Optional

import fire
import torch

from yagpt.dataset import YaDataset


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


def split_dataset(tokens, train_ratio=0.9):
    split_idx = round(len(tokens) * train_ratio)
    train_tokens, val_tokens = tokens[:split_idx], tokens[split_idx:]
    return train_tokens, val_tokens


def main(
        dataset_path: Optional[str] = None,
        output_dir: str = os.path.dirname(__file__)
):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), 'inferno.txt')

    raw_text = read_dataset(dataset_path, lower_case=False, shuffle_sections=True)
    tokens = YaDataset.tokenize(raw_text)
    train_tokens, val_tokens = split_dataset(tokens)

    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_tokens, os.path.join(output_dir, 'train.bin'))
    torch.save(val_tokens, os.path.join(output_dir, 'val.bin'))


if __name__ == '__main__':
    fire.Fire(main)
