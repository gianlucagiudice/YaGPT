from yagpt.model import YaGPTConfig, YaGPT


def model_factory(
        d_model: int,
        seq_len: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
):
    config = YaGPTConfig(
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_model * 4,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )

    model: YaGPT = YaGPT(config)

    return model
