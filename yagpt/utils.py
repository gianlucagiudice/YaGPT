from yagpt.model import YaGPT, Embeddings, PositionalEncoding


def gpt_factory(
        vocab_size: int,
        d_model: int,
        seq_len: int
) -> YaGPT:
    input_embeddings = Embeddings(vocab_size, d_model)
    pos_encoding = PositionalEncoding(seq_len, d_model)
    gpt = YaGPT(input_embeddings, pos_encoding)
    return gpt
