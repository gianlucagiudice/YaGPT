from dataclasses import dataclass


@dataclass
class YaGPTConfig:
    seq_len: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float

    vocab_size: int
