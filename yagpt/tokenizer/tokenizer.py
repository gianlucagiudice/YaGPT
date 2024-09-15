from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union


@dataclass
class TokenizerMapping:
    id_to_token: dict[int, str]
    token_to_id: dict[str, int]

    def __len__(self):
        return len(self.id_to_token)

    def __getitem__(self, item: Union[int, str]) -> Union[int, str]:
        if isinstance(item, int):
            return self.id_to_token[item]
        if isinstance(item, str):
            return self.token_to_id[item]
        raise ValueError("item must be either int or str")

    def __str__(self) -> str:
        out_str = "TokenizerMapping:\n"
        for i, t in self.id_to_token.items():
            out_str += f"({i}: {t.encode()}) "
        return out_str


class AbstractTokenizer(ABC):

    @abstractmethod
    def encode(self, text) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens) -> str:
        pass
