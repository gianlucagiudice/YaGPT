from typing import Optional

from . import *


def tokenizer_factory(
        tokenizer_name: str,
        tokenizer_path: Optional[str] = None
) -> AbstractTokenizer:
    match tokenizer_name:
        case 'gpt2':
            tokenizer = GPT2Tokenizer()
        case 'char':
            tokenizer = CharTokenizer()
        case 'bpe':
            tokenizer = BPETokenizer()
        case _:
            raise ValueError(f'Unknown tokenizer {tokenizer_name}')

    if tokenizer_path is not None:
        tokenizer.load(tokenizer_path)

    return tokenizer
