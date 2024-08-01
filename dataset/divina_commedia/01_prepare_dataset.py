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


def split_dataset(tokens, train_ratio):
    split_idx = round(len(tokens) * train_ratio)
    train_tokens, val_tokens = tokens[:split_idx], tokens[split_idx:]
    return train_tokens, val_tokens


def remap_tokens(tokens):
    unique_tokens = list(set(tokens))
    id_to_token = {i: t for i, t in enumerate(unique_tokens)}
    token_to_id = {t: i for i, t in id_to_token.items()}

    return id_to_token, token_to_id


def main(
        dataset_path: str,
        output_dir: str = None,
        train_ratio: float = 0.9
):
    if dataset_path is None:
        raise ValueError('dataset_path must be provided')
    if output_dir is None:
        output_dir = os.path.dirname(dataset_path)

    # Read dataset
    raw_text = read_dataset(dataset_path, lower_case=False, shuffle_sections=True)
    # Tokenize dataset
    tokens = YaDataset.tokenizer.encode(raw_text)
    # Remap tokens
    id_to_token, token_to_id = remap_tokens(tokens)
    tokens = [token_to_id[t] for t in tokens]
    # Split dataset
    train_tokens, val_tokens = split_dataset(tokens, train_ratio=train_ratio)
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)

    train_data = dict(tokens=train_tokens, id_to_token=id_to_token, token_to_id=token_to_id)
    torch.save(train_data, os.path.join(output_dir, 'train.bin'))
    val_data = dict(tokens=val_tokens, id_to_token=id_to_token, token_to_id=token_to_id)
    torch.save(val_data, os.path.join(output_dir, 'val.bin'))


if __name__ == '__main__':
    fire.Fire(main)
