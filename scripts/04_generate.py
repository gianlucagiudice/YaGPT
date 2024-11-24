import fire

from yagpt import YaGPTWrapper, BPETokenizer


def generate(
        model_checkpoint_path: str,
        tokenizer_path: str,
        n_steps: int = 200,
        temperature: float = 1.5,
        top_k: int = 5
):
    if model_checkpoint_path is None:
        raise ValueError('Please provide destination_dir or set MODEL_WEIGHTS_DIR in .env file')

    if tokenizer_path is None:
        raise ValueError('Please provide tokenizer_path or set TOKENIZER_PATH in .env file')

    # Load model using Lightning
    model: YaGPTWrapper = YaGPTWrapper.load_from_checkpoint(model_checkpoint_path)

    # Load Tokenizer
    tokenizer: BPETokenizer = BPETokenizer.load_from_checkpoint(tokenizer_path)

    text = ("Inferno\n"
            "Canto XXXIV\n\n"
            "Giunti là dove ")

    print(f">>> Context:\n"
          f"{text}\n"
          f">>> Generated text:")

    for token in model.generate_text(text, n_steps, temperature=temperature, top_k=top_k, tokenizer=tokenizer):
        print(token, end='')


if __name__ == '__main__':
    fire.Fire(generate)
