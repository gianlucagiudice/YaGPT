import torch

from yagpt.model import YaGPT
from yagpt.utils import gpt_factory


def main(
        batch_size: int = 8,
        vocab_size: int = 32_000,
        d_model: int = 768,
        seq_len: int = 512,
):
    model: YaGPT = gpt_factory(
        vocab_size=vocab_size,
        d_model=d_model,
        seq_len=seq_len
    )

    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    res = model(input_ids)
    print(res.shape)


if __name__ == '__main__':
    main()
