import os
from pathlib import Path
from typing import Optional

import fire
import wandb
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent


def download_model_from_wandb(
        artifact_id: Optional[str] = None,
        destination_dir: str = None
):
    if artifact_id is None:
        artifact_id = os.getenv('WANDB_ARTIFACT_ID')
        if artifact_id is None:
            raise ValueError('Please provide artifact_id or set WANDB_ARTIFACT_ID in .env file')

    if destination_dir is None:
        model_dir = os.getenv('MODEL_WEIGHTS_DIR')
        if model_dir is None:
            raise ValueError('Please provide destination_dir or set MODEL_WEIGHTS_DIR in .env file')
        destination_dir = str(ROOT / model_dir)

    run = wandb.init()
    artifact = run.use_artifact(artifact_id, type='model')
    artifact_dir = artifact.download(destination_dir)
    print(f'Model downloaded to {artifact_dir}')


if __name__ == '__main__':
    fire.Fire(download_model_from_wandb)
