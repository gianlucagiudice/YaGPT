from dataclasses import dataclass
from typing import List, Union

from yagpt.tokenizer import AbstractTokenizer


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


def init_vocabulary() -> TokenizerMapping:
    # ASCII characters
    ids = list(range(256))
    ascii_chars = list(map(chr, ids))
    # Create a mapping from IDs to characters
    ids_to_token = dict(zip(ids, ascii_chars))
    token_to_ids = dict(zip(ascii_chars, ids))
    # Return the TokenizerMapping object
    tokenizer_mapping = TokenizerMapping(ids_to_token, token_to_ids)
    return tokenizer_mapping


class CharTokenizer(AbstractTokenizer):
    tokenizer_mapping: TokenizerMapping = init_vocabulary()

    @classmethod
    def encode(cls, text: str, **kwargs) -> List[int]:
        encoded = [cls.tokenizer_mapping[c] for c in text]
        return encoded

    @classmethod
    def decode(cls, tokens: List[int], **kwargs) -> str:
        decoded = [cls.tokenizer_mapping[c] for c in tokens]
        decoded = "".join(decoded)
        return decoded

    def __len__(self):
        return len(self.tokenizer_mapping)


if __name__ == "__main__":
    # Init the tokenizer
    char_tokenizer = CharTokenizer()
    # Print the vocabulary
    print(char_tokenizer.tokenizer_mapping)
    # Encode a string
    sample_text = "hello"
    encoded_text = char_tokenizer.encode(sample_text)
    print(encoded_text)
    # Decode the encoded string
    decoded_text = char_tokenizer.decode(encoded_text)
    print(decoded_text)
    # Print the vocabulary size
    assert sample_text == decoded_text
