from typing import List

from minbpe import BasicTokenizer
from .tokenizer import AbstractTokenizer


class BPETokenizer(AbstractTokenizer):

    def __init__(self):
        self.tokenizer = None

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized")

        tokens = self.tokenizer.encode(text)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized")

        text = self.tokenizer.decode(tokens)
        return text

    def train(
            self,
            text: str,
            vocab_size: int,
            save_path: str,
            verbose: bool = False
    ):
        # Train the tokenizer
        self.tokenizer = BasicTokenizer()
        self.tokenizer.train(text, vocab_size=vocab_size, verbose=verbose)

        # Save the tokenizer
        self.tokenizer.save(save_path)

    def load(self, model_file: str):
        self.tokenizer = BasicTokenizer()
        self.tokenizer.load(model_file)

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized")

        vocab_size = len(self.tokenizer.vocab)
        return vocab_size

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> 'BPETokenizer':
        tokenizer = BPETokenizer()
        tokenizer.load(checkpoint_path)
        return tokenizer
