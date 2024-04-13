import torch

from yagpt.model import YaGPT, YaGPTConfig


def main(
        batch_size: int = 8,
        d_model: int = 256,
        seq_len: int = 512,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        vocab_size: int = 32_000,
):

    config = YaGPTConfig(
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_model * 4,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        vocab_size=vocab_size
    )

    model: YaGPT = YaGPT(config)

    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    res = model(input_ids)
    print(res.shape)


if __name__ == '__main__':
    main()
