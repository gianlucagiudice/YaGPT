# YaGPT: GPT Implementation from Scratch Using PyTorch

![YaGPT Text Generation](#)  <!-- Placeholder for an image showing the model generating text -->

This repository presents YaGPT (Yet Another GPT), a GPT (Generative Pre-trained Transformer) model implemented entirely from scratch using only PyTorch. The model is trained exclusively on the Divina Commedia, aiming to generate text in the style of Dante Alighieri’s masterpiece. Note that only the pre-training phase has been completed, and no fine-tuning has been performed.

## 🌐 Introduction to Transformer Architecture

Introduced by Vaswani et al. in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), the transformer architecture revolutionized the field of natural language processing with its self-attention mechanism. Unlike traditional RNNs, transformers can process data sequences in parallel, making them highly efficient for modern hardware.

## 🔍 GPT and the Transformer Decoder

GPT (Generative Pre-trained Transformer) is an autoregressive language model that uses the decoder part of the transformer architecture to generate coherent and contextually relevant text. The transformer architecture consists of two main components:

1. **Encoder**: Processes the input sequence and generates a context-aware representation.
2. **Decoder**: Uses the encoder's output to generate the target sequence one token at a time.

In a GPT model, only the decoder component is used. This allows the model to generate text by predicting the next token in a sequence based on the previously generated tokens.

![Transformer Decoder Architecture](#)  <!-- Placeholder for an image depicting the transformer decoder architecture -->

## 🏗️ Model Architecture

YaGPT follows the standard transformer decoder architecture with the following key components:

1. **Tokenization**: Text is tokenized into individual words or subwords.
2. **Embedding Layer**: Tokens are converted into dense vector representations.
3. **Positional Encodings**: Encodings are added to embeddings to retain positional information.
4. **Transformer Decoder Blocks**: A series of transformer layers with self-attention mechanisms and feed-forward networks.
5. **Output Layer**: Generates probabilities for the next token in the sequence.

## 🔧 Model Hyperparameters

Understanding the hyperparameters used in YaGPT is crucial for reproducing and fine-tuning the model:

- **seq_len**: Maximum sequence length for the input text.
- **d_model**: Dimensionality of the token embeddings.
- **n_heads**: Number of attention heads in each transformer layer.
- **n_layers**: Number of transformer layers in the decoder.
- **d_ff**: Dimensionality of the feed-forward network within the transformer.
- **dropout**: Dropout rate for regularization.
- **vocab_size**: Size of the vocabulary derived from the training dataset.

## ✍ YaGPT: Generating the Divina Commedia

YaGPT has been trained from scratch on the Divina Commedia to capture the unique style, language, and structure of this literary work. Below is a breakdown of the implementation and training process.

### 🛠️ Training and Hyperparameters

To train YaGPT, follow these steps:

1. Navigate to the repository's root directory.
2. Execute the training script:

   ```
   python train.py --dataset_path [DATASET_PATH] --batch_size [BATCH_SIZE] --d_model [D_MODEL] --seq_len [SEQ_LEN] --n_heads [N_HEADS] --n_layers [N_LAYERS] --dropout [DROPOUT] --max_epochs [MAX_EPOCHS] --lr [LEARNING_RATE]
   ```

Parameters:

- **dataset_path**: Path to the dataset (Divina Commedia text).
- **batch_size**: Size of each training batch.
- **d_model**: Dimensionality of the token embeddings.
- **seq_len**: Maximum sequence length for the input text.
- **n_heads**: Number of attention heads.
- **n_layers**: Number of transformer layers.
- **dropout**: Dropout rate.
- **max_epochs**: Maximum number of training epochs.
- **lr**: Learning rate.
- **train_ratio**: Ratio of the dataset to use for training (default: 0.9).
- **max_steps**: Maximum number of training steps (optional).
- **accelerator**: Type of hardware accelerator to use (e.g., 'gpu', 'tpu').
- **val_check_interval**: Interval for validation checks during training (optional).
- **limit_val_batches**: Limit on the number of validation batches (optional).
- **log_every_n_steps**: Logging interval during training.
- **gradient_clip_val**: Gradient clipping value to prevent exploding gradients.
- **early_stopping_patience**: Number of epochs with no improvement after which training will be stopped.
- **scheduler_t0**: Initial number of iterations for the Cosine Annealing Warm Restarts learning rate scheduler.
- **scheduler_t_mult**: Factor by which the number of iterations between restarts is multiplied.

## 📊 Performance and Results

### Overview

YaGPT has been evaluated based on its ability to generate text that mimics the style and content of the Divina Commedia. The generated text is assessed qualitatively for coherence, fluency, and adherence to the original text’s style.

### Example Output

Here is a sample output generated by YaGPT after training on the Divina Commedia:

```
Midway upon the journey of our life,
I found myself within a forest dark,
For the straightforward path had been lost.

Ah, how hard it is to tell
What that forest was, wild, rough, and harsh,
The thought of it renews my fear.
```

### Detailed Training Logs

Training logs and metrics are tracked using [Weights & Biases](https://wandb.ai/), providing an in-depth look at the model's training history, including loss curves and generated text samples.

## 🛠️ Using YaGPT

### Text Generation Script

The `generate.py` script allows you to generate text using the trained YaGPT model. To generate text, run:

```
python generate.py --config_path configs/yagpt/yagpt.yaml --input_text "Enter your prompt here"
```

Parameters:

- `--config_path`: Path to the model configuration file.
- `--input_text`: The initial text prompt to start the generation.

## ✍ Conclusion

YaGPT demonstrates the potential of transformer-based models to generate text that closely resembles the style of classical literature. While this implementation focuses on the Divina Commedia, the model can be adapted to other texts and purposes with appropriate training data.

### Future Work

Future improvements could include fine-tuning on a more diverse dataset, experimenting with different model architectures, and exploring various text generation tasks.

**In the realm of text generation, the right model and data can transform your project from mundane to divine.**