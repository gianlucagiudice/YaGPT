import os

import fire
import torch

from yagpt.tokenizer import CharTokenizer


def read_dataset(dataset_path: str, lower_case: bool) -> str:
    with open(dataset_path, 'r') as f:
        text = f.read()
    if lower_case:
        text = text.lower()
    return text


def split_dataset(tokens, train_ratio):
    split_idx = round(len(tokens) * train_ratio)
    train_tokens, val_tokens = tokens[:split_idx], tokens[split_idx:]
    return train_tokens, val_tokens


def main(
        dataset_path: str,
        output_dir: str = None,
        train_ratio: float = 0.8
):
    if dataset_path is None:
        raise ValueError('dataset_path must be provided')
    if output_dir is None:
        output_dir = os.path.dirname(dataset_path)

    # Read dataset
    raw_text = read_dataset(dataset_path, lower_case=False)
    # Tokenize dataset
    char_tokenizer = CharTokenizer()
    tokens = char_tokenizer.encode(raw_text)
    # Remap tokens
    tokenizer_mapping = char_tokenizer.tokenizer_mapping
    id_to_token, token_to_id = tokenizer_mapping.id_to_token, tokenizer_mapping.token_to_id
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
