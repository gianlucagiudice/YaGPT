from typing import List, Dict

import lightning
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from yagpt.model import YaGPT


class TrainingGenerationCallback(Callback):
    def __init__(self, n_samples: int = 4, autoregressive_steps: int = 16):
        self.n_samples = n_samples
        self.autoregressive_steps = autoregressive_steps

    @staticmethod
    def autoregressive_generation(
            pl_module: YaGPT,
            data_loader: DataLoader,
            n_samples: int,
            autoregressive_steps: int
    ) -> List[Dict]:
        # Shuffle the validation set
        idx2token = data_loader.dataset.idx2token
        random_val_dataloader = DataLoader(data_loader.dataset, batch_size=1, shuffle=True)

        data: List[dict] = []

        # Generate text
        for sample_id in random_val_dataloader.sampler:
            x_sample, _ = random_val_dataloader.dataset[sample_id]
            x_sample = torch.tensor(x_sample).unsqueeze(dim=0)

            generated_tokens = pl_module.generate_text(x_sample, autoregressive_steps)
            # Decode the generated tokens
            context_text = [idx2token[token] for token in x_sample.flatten().tolist()]
            generated_text = [idx2token[token] for token in generated_tokens.flatten().tolist()]

            data.append({
                'context': context_text,
                'generated': generated_text
            })

            if len(data) >= n_samples:
                break

        return data

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: YaGPT):
        # Evaluate Training Set
        splits = [('train', trainer.train_dataloader), ('val', trainer.val_dataloaders)]

        for split, dataloader in splits:
            if dataloader is None or dataloader.dataset is None:
                continue

            autoregressive_generation = self.autoregressive_generation(
                pl_module, dataloader, self.n_samples, self.autoregressive_steps
            )

            generated = [
                {
                    'context': dataloader.dataset.untokenize(x['context']),
                    'generated': dataloader.dataset.untokenize(x['generated']),
                    'split': split,
                    'step': trainer.global_step,
                    'epoch': trainer.current_epoch
                }
                for x in autoregressive_generation
            ]

            trainer.logger.log_table(
                key=f'generation-split-{split}-step-{trainer.global_step}-epoch-{trainer.current_epoch}',
                columns=list(generated[0].keys()),
                data=[list(d.values()) for d in generated]
            )

            print(f'\n{"=" * 25} Generated text from {split.upper()} set {"=" * 25}')
            for x in generated:
                context, generated = x['context'], x['generated']

                print(f'> [Context]:\n{context}\n'
                      f'> [Generated]:\n{generated}\n'
                      f'{"-" * 50}')
