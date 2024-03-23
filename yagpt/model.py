import torch


class Embeddings(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.latent_dim = d_model
        self.embeddings = torch.nn.parameter.Parameter(
            torch.randn(vocab_size, d_model)
        )
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.xavier_normal_(self.embeddings)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # IDS: (B, N), Embeddings: (Vocab_size, D) -> (B, N, D)
        embeddings = self.embeddings[input_ids]
        embeddings = embeddings * torch.sqrt(torch.tensor(self.latent_dim))
        return embeddings


class PositionalEncoding(torch.nn.Module):
    def __init__(self, sequence_len: int, d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_len = sequence_len
        self.d_model = d_model
        self.pe = self._init_pe(sequence_len, d_model)
        self.register_buffer('positional_encoding', self.pe)

    @staticmethod
    def _init_pe(n: int, d: int) -> torch.Tensor:
        pos_ids = torch.arange(0, n).unsqueeze(1).repeat(1, d)
        dim_ids = torch.arange(0, d).unsqueeze(0).repeat(n, 1)

        shared = pos_ids / (torch.tensor(10_000) ** ((2 * dim_ids) / d))

        even_pos = torch.sin(shared)
        odds_pos = torch.cos(shared)

        pe = torch.zeros((n, d))
        pe[..., 0::2] = even_pos[..., 0::2]
        pe[..., 1::2] = odds_pos[..., 1::2]

        return pe

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        res = embeddings + self.pe
        return res


class YaGPT(torch.nn.Module):
    def __init__(self, embeddings: Embeddings, pos_encoding: PositionalEncoding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = embeddings
        self.pos_encoding = pos_encoding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        embeddings = self.pos_encoding(embeddings)

        return embeddings
