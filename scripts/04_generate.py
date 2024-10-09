from textwrap import dedent

import fire
import torch

from yagpt.model import YaGPTWrapper
from yagpt.tokenizer import tokenizer_factory


def generate(
        model_checkpoint_path: str,
        tokenizer_path: str,
        tokenizer_name: str = 'bpe',
        n_steps: int = 100
):
    if model_checkpoint_path is None:
        raise ValueError('Please provide destination_dir or set MODEL_WEIGHTS_DIR in .env file')

    if tokenizer_path is None:
        raise ValueError('Please provide tokenizer_path or set TOKENIZER_PATH in .env file')

    # Load model using Lightning
    model: YaGPTWrapper = YaGPTWrapper.load_from_checkpoint(model_checkpoint_path)
    model.eval()

    # Load Tokenizer
    tokenizer = tokenizer_factory(tokenizer_name, tokenizer_path)

    text = dedent("""
    Tanto gentile e tanto onesta pare
    la donna mia quand'ella altrui saluta,
    ch'ogne lingua deven tremando muta,
    e li occhi no l'ardiscon di guardare.
    Ella si va, sentendosi laudare,
    benignamente d'umiltà vestuta;
    e par che sia una cosa venuta
    da cielo in terra a miracol mostrare.
    Mostrasi sì piacente a chi la mira,
    che dà per li occhi una dolcezza al core,
    """)

    print(f">>> Context:\n"
          f"{text}\n"
          f">>> Generated text:\n")
    tokens = tokenizer.encode(text)
    batch = torch.tensor(tokens).unsqueeze(0).long()
    for token in model.model.generate_text(batch, n_steps, temperature=2, top_k=50):
        token = tokenizer.decode([token])
        print(token, end='')


if __name__ == '__main__':
    fire.Fire(generate)
