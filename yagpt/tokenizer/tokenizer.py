from abc import ABC, abstractmethod
from typing import List


class AbstractTokenizer(ABC):

    @abstractmethod
    def encode(self, text) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens) -> str:
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, model_file: str):
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass
