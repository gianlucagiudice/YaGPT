import os

import fire

from yagpt.tokenizer import tokenizer_factory


def read_dataset(dataset_path: str, lower_case: bool) -> str:
    with open(dataset_path, 'r') as f:
        text = f.read()
    if lower_case:
        text = text.lower()
    return text


def train_tokenizer(
        dataset_path: str,
        tokenizer_name: str = 'bpe',
        vocab_size: int = 1_000,
        output_dir: str = None,
        verbose: bool = True
):
    if dataset_path is None:
        raise ValueError('dataset_path must be provided')
    if output_dir is None:
        output_dir = os.path.dirname(dataset_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read dataset
    raw_text = read_dataset(dataset_path, lower_case=False)
    # Initialize tokenizer
    tokenizer = tokenizer_factory(tokenizer_name)
    # Train tokenizer
    tokenizer.train(
        text=raw_text,
        vocab_size=vocab_size,
        verbose=verbose,
        save_path=os.path.join(output_dir, 'tokenizer')
    )


if __name__ == '__main__':
    fire.Fire(train_tokenizer)
