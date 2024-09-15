from . import GPT2Tokenizer, CharTokenizer, AbstractTokenizer


def tokenizer_factory(tokenizer_name: str) -> AbstractTokenizer:
    match tokenizer_name:
        case 'gpt2':
            return GPT2Tokenizer()
        case 'char':
            return CharTokenizer()
        case _:
            raise ValueError(f'Unknown tokenizer {tokenizer_name}')
