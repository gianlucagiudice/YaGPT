import enum
from typing import List, Dict

import lightning
import wandb
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from yagpt.dataset import YaDataset
from yagpt.model import YaGPTWrapper


class TableColumns(enum.Enum):
    CONTEXT = 'context'
    GENERATED = 'generated'
    SPLIT = 'split'
    STEP = 'step'
    EPOCH = 'epoch'

    @staticmethod
    def get_values():
        return [
            TableColumns.CONTEXT.value,
            TableColumns.GENERATED.value,
            TableColumns.SPLIT.value,
            TableColumns.STEP.value,
            TableColumns.EPOCH.value
        ]


class TrainingGenerationCallback(Callback):
    def __init__(
            self,
            id_to_token: dict[int, int],
            n_samples: int = 4,
            autoregressive_steps: int = 16,
            top_k: int = 5,
            temperature: float = 1.0
    ):
        self.n_samples = n_samples
        self.top_k = top_k
        self.temperature = temperature
        self.autoregressive_steps = autoregressive_steps
        self.table = wandb.Table(columns=TableColumns.get_values())
        self.id_to_token = id_to_token

    @staticmethod
    def autoregressive_generation(
            pl_module: YaGPTWrapper,
            data_loader: DataLoader,
            n_samples: int,
            autoregressive_steps: int,
            top_k: int,
            temperature: float,
            id_to_token: dict[int, int]
    ) -> List[Dict]:
        # Shuffle the validation set
        random_val_dataloader = DataLoader(data_loader.dataset, batch_size=1, shuffle=True)

        data: List[dict] = []

        # Generate text
        for sample_id in random_val_dataloader.sampler:
            x_sample, _ = random_val_dataloader.dataset[sample_id]
            x_sample = x_sample.unsqueeze(dim=0)

            tokens_iterator = pl_module.model.generate_text(
                x_sample, autoregressive_steps, top_k=top_k, temperature=temperature)

            # Decode the generated tokens
            context_text = YaDataset.untokenize_helper(x_sample.flatten().tolist(), id_to_token)
            generated_text = YaDataset.untokenize_helper(list(tokens_iterator), id_to_token)

            data.append({
                'context': context_text,
                'generated': generated_text
            })

            if len(data) >= n_samples:
                break

        return data

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: YaGPTWrapper):
        # Evaluate Training Set
        splits = [('train', trainer.train_dataloader), ('val', trainer.val_dataloaders)]

        for split, dataloader in splits:
            if dataloader is None or dataloader.dataset is None:
                continue

            autoregressive_generation = self.autoregressive_generation(
                pl_module, dataloader, self.n_samples, self.autoregressive_steps,
                self.top_k, self.temperature, self.id_to_token
            )

            generated = [
                {
                    TableColumns.CONTEXT.value: x['context'],
                    TableColumns.GENERATED.value: x['generated'],
                    TableColumns.SPLIT.value: split,
                    TableColumns.STEP.value: trainer.global_step,
                    TableColumns.EPOCH.value: trainer.current_epoch
                }
                for x in autoregressive_generation
            ]

            for data in generated:
                data_values = [data[col.value] for col in TableColumns]
                self.table.add_data(*data_values)

            table_name = f'generation-split-{split}-epoch-{trainer.current_epoch}-step-{trainer.global_step}'
            wandb.log({table_name: self.table}, commit=True)

            print(f'\n{"=" * 25} Generated text from {split.upper()} set {"=" * 25}')
            for x in generated:
                context, generated = x['context'], x['generated']

                print(f'> [Context]:\n{context}\n'
                      f'> [Generated]:\n{generated}\n'
                      f'{"-" * 50}')
