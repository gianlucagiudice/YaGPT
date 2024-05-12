from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class YaGPTConfig:
    seq_len: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float

    vocab_size: int


class Embeddings(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embeddings = torch.nn.parameter.Parameter(
            torch.randn(vocab_size, d_model)
        )
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.xavier_normal_(self.embeddings)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # IDS: (B, N), Embeddings: (Vocab_size, D) -> (B, N, D)
        embeddings = self.embeddings[input_ids]
        embeddings = embeddings * torch.sqrt(torch.tensor(self.d_model))
        return embeddings


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.pe = self._init_pe(seq_len, d_model)
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
        res = embeddings + self.pe.to(embeddings.device)
        return res


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class CausalMultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float):
        super().__init__()

        assert d_model % n_heads == 0, ValueError(
            'Error! The model dimension should be divisible by the number of heads')

        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_transform = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_transform = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_transform = torch.nn.Linear(d_model, d_model, bias=False)

        self.linear = torch.nn.Linear(d_model, d_model)

        # Create a causal mask to zero out future positions
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, requires_grad=False)
        mask = torch.tril(mask)
        self.register_buffer('causal_mask', mask)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute QKV
        q = self.q_transform(x)  # (B, N, D)
        k = self.k_transform(x)  # (B, N, D)
        v = self.v_transform(x)  # (B, N, D)

        # Split matrices into multiple heads (B, N, D) -> (B, N, HEADS, D_HEAD)
        q = q.view(x.shape[0], self.seq_len, self.n_heads, self.d_head)
        k = k.view(x.shape[0], self.seq_len, self.n_heads, self.d_head)
        v = v.view(x.shape[0], self.seq_len, self.n_heads, self.d_head)

        # Transpose tensor (B, N, HEADS, D_HEAD) -> (B, HEADS, N, D_HEAD)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attention = q @ k.transpose(2, 3)
        attention = attention / (self.d_head ** 0.5)

        # Apply the causal mask by setting future values to -inf
        attention_mask = self.causal_mask.unsqueeze(0).unsqueeze(0).expand(x.shape[0], self.n_heads, -1, -1)
        attention = attention.masked_fill(attention_mask == 0, float('-inf'))

        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = attention @ v

        # Concatenate heads
        res = attention.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        res = self.linear(res)
        res = self.dropout(res)

        return res


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, seq_len: int, dropout: float):
        super().__init__()
        self.layer_norm_1 = torch.nn.LayerNorm(d_model, bias=False)
        self.causal_self_attention = CausalMultiHeadAttentionLayer(d_model, n_heads, seq_len, dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model, bias=False)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x = self.layer_norm_1(x)
        x = self.causal_self_attention(x)
        x = x + x_res

        x_res = x
        x = self.layer_norm_2(x)
        x = self.ffn(x)
        x = x + x_res

        return x


class Decoder(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, n_heads: int, seq_len: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, d_ff, n_heads, seq_len, dropout) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class YaGPT(torch.nn.Module):
    def __init__(self, config: YaGPTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.embeddings = Embeddings(config.d_model, config.vocab_size)
        self.pos_encoding = PositionalEncoding(config.d_model, config.seq_len)
        self.embeddings_dropout = torch.nn.Dropout(config.dropout)
        self.decoder = Decoder(
            config.d_model,
            config.d_ff,
            config.n_heads,
            config.seq_len,
            config.n_layers,
            config.dropout
        )
        self.normalization = torch.nn.LayerNorm(config.d_model)
        self.head = torch.nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embedding Layer
        embeddings = self.embeddings(input_ids)
        embeddings = self.pos_encoding(embeddings)
        embeddings = self.embeddings_dropout(embeddings)
        # Decoder
        embeddings = self.decoder(embeddings)
        # Normalization
        embeddings = self.normalization(embeddings)
        # Head
        logits = self.head(embeddings)

        return logits

    def generate_text(self, x: torch.Tensor, n_steps: int) -> Iterable[int]:
        assert x.dim() == 2 and x.shape[0] == 1, ValueError('Input tensor should have shape (1, N)')

        # Generate auto-regressively
        for step in range(n_steps):
            # Forward pass
            x_forward = x
            if x_forward.shape[1] > self.config.seq_len:
                x_forward = x_forward[:, -self.config.seq_len:]
            if x_forward.shape[1] < self.config.seq_len:
                padding = torch.zeros(x_forward.shape[0], self.config.seq_len - x_forward.shape[1], dtype=torch.long)
                x_forward = torch.cat([x_forward, padding], dim=1)

            logits = self(x_forward)

            # Sample the next token
            last_pos = min(self.config.seq_len, x.shape[1])

            next_token_pred = torch.argmax(logits[0, last_pos - 1, :], keepdim=True)
            x = torch.cat([x[0, :last_pos], next_token_pred]).unsqueeze(dim=0)

            next_token = next_token_pred.item()
            yield next_token
