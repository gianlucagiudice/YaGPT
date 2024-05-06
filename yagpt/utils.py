from yagpt.model import YaGPTConfig, YaGPTWrapper


def model_factory(
        d_model: int,
        seq_len: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        vocab_size: int,
) -> YaGPTWrapper:

    config = YaGPTConfig(
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_model * 4,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        vocab_size=vocab_size
    )

    model: YaGPTWrapper = YaGPTWrapper(config)

    return model
