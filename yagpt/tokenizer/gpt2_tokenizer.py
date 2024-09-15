from typing import List

import tiktoken

from yagpt.tokenizer import AbstractTokenizer


class GPT2Tokenizer(AbstractTokenizer):
    tokenizer = tiktoken.get_encoding('gpt2')

    @classmethod
    def encode(cls, text) -> List[int]:
        return cls.tokenizer.encode(text)

    @classmethod
    def decode(cls, tokens) -> str:
        return cls.tokenizer.decode(tokens)
