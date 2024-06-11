import os
from pathlib import Path
from textwrap import dedent
from typing import Optional

import fire
from dotenv import load_dotenv
import torch

from yagpt.model import YaGPTWrapper
from yagpt.dataset import DivinaCommediaDataset

load_dotenv()

ROOT = Path(__file__).parent.parent


def generate(
        model_checkpoint_path: Optional[str] = None,
        n_steps: int = 100
):
    if model_checkpoint_path is None:
        model_dir = os.getenv('MODEL_WEIGHTS_DIR')
        if model_dir is None:
            raise ValueError('Please provide destination_dir or set MODEL_WEIGHTS_DIR in .env file')
        model_checkpoint_path = str(ROOT / model_dir / 'model.ckpt')

    # Load model using Lightning
    model: YaGPTWrapper = YaGPTWrapper.load_from_checkpoint(model_checkpoint_path)
    model.eval()

    text = dedent("""
    O poca nostra nobiltà di sangue,
    se gloriar di te la gente fai
    qua giù dove l’affetto nostro langue,
    
    mirabil cosa non mi sarà mai:
    ché là dove appetito non si torce,
    dico nel cielo, io me ne gloriai.
    
    Ben se’ tu manto che tosto raccorce:
    sì che, se non s’appon di dì in die,
    lo tempo va dintorno con le force.
    
    Dal ‘voi’ che prima a Roma s’offerie,
    in che la sua famiglia men persevra,
    ricominciaron le parole mie;
    """)

    print(f">>> Context:\n"
          f"{text}\n"
          f">>> Generated text:\n")
    tokens = DivinaCommediaDataset.tokenizer.encode(text)
    batch = torch.tensor(tokens).unsqueeze(0).long()
    for token in model.model.generate_text(batch, n_steps):
        token = DivinaCommediaDataset.tokenizer.decode([token])
        print(token, end='')


if __name__ == '__main__':
    fire.Fire(generate)
