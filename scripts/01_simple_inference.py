import torch

from yagpt.model import YaGPT, YaGPTConfig


def main():
    config = YaGPTConfig(
        seq_len=1024,
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        vocab_size=50257,
    )
    model = YaGPT(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (1, config.seq_len))
    res = model(x)
    print(res.shape)


if __name__ == '__main__':
    main()
