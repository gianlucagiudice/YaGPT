import fire
import torch

from yagpt.model import YaGPTWrapper
from yagpt.tokenizer import tokenizer_factory


def generate(
        model_checkpoint_path: str,
        tokenizer_path: str,
        tokenizer_name: str = 'bpe',
        n_steps: int = 200
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

    text = ("Inferno\n"
            "Canto XXXIV\n\n"
            "Giunti là dove ")

    print(f">>> Context:\n"
          f"{text}\n"
          f">>> Generated text:")
    tokens = tokenizer.encode(text)
    batch = torch.tensor(tokens).unsqueeze(0).long()
    for token in model.model.generate_text(batch, n_steps, temperature=1.5, top_k=5):
        token = tokenizer.decode([token])
        print(token, end='')


if __name__ == '__main__':
    fire.Fire(generate)
