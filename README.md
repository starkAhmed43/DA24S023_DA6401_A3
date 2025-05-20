# DA24S023_DA6401_A3: Transliteration with Seq2Seq and Attention

This project implements a **sequence-to-sequence transliteration model** using PyTorch Lightning. It supports both **Luong** and **Bahdanau** attention mechanisms and is designed for transliterating text from **Latin script to Tamil script** using the **Dakshina dataset**.

---

## Student Details
- Name: Adhil Ahmed
- Roll No: DA24S023
- Email: da24s023@smail.iitm.ac.in

---

## Project Overview

- **Goal**: Train and evaluate a deep learning model to transliterate Latin characters to Tamil (default. Can be trained for other languages by passing appropriate args) using RNNs.
- **Architecture**: Encoder-decoder with configurable RNN cells (GRU/LSTM) and attention mechanisms.
- **Framework**: PyTorch Lightning for clean training loops and experiment management.
- **Dataset**: Dakshina transliteration dataset.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/starkAhmed43/DA24S023_DA6401_A3
cd DA24S023_DA6401_A3
```

### 2. Create and Activate a Virtual Environment (Optional)

**Using venv:**
```bash
python -m venv da6401
source da6401/bin/activate  # On Windows: da6401\Scripts\activate
```

**Or, using Conda (recommended for reproducibility):**
```bash
conda create -n da6401 python=3.12 -y
conda activate da6401
```

### 3. Install PyTorch

First, install the appropriate version of PyTorch for your system by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

`torchvision` and `torchaudio` are not needed for this repository.

For example, with Conda and CPU-only:
```bash
conda install pytorch cpuonly -c pytorch
```
Or with pip and CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Other Dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

(Optional but recommended) This project uses [Weights & Biases (Wandb)](https://wandb.ai/) for experiment tracking, logging, and hyperparameter sweeps.

- **Wandb is required for training using the provided scripts as-is.** If you do not want to use Wandb for training, simply remove or comment out the `WandbLogger` in `train.py`.
- For hyperparameter sweeps, Wandb is essential, as the sweep functionality is built around Wandb's tools.

To get started:
1. Sign up at [wandb.ai](https://wandb.ai/) if you don't have an account.
2. Log in from the terminal:
    ```bash
    wandb login
    ```
    You will be prompted to paste your API key from your Wandb account page.

If Wandb is not configured, training will fail unless you modify the code to remove Wandb integration. Hyperparameter sweeps cannot be run without Wandb.

---

## Instructions

### Dataset
The code will **automatically download and extract** the required Dakshina dataset as well as the trained checkpoints to the `data/` directory when the training script is first run. No manual steps are needed. 

If you would like to play around with them before that, you can start the download by running the `datamodule` script:
```bash
python datamodule.py
```

---

## Training

To train a model with a specific configuration, use:

```bash
python train.py --config model_configs/bahdanau_rnn.yaml
```

Alternate config options include:

- `model_configs/luong_rnn.yaml` – For Luong Attention
- `model_configs/bahdanau_rnn.yaml` – For Bahdanau Attention

These configs control hyperparameters like embedding size, hidden dimensions, RNN layers, learning rate, attention type, and more. The hparams are set to values that resulted in the most performant run during hyperparameter sweeps.

**Arguments for both `train.py` and `test.py`:**

You can specify any of the following arguments on the command line. If a value is provided both as a command-line argument and in the YAML config, the value from the YAML config will take precedence.

- `--config`: Path to YAML config file
- `--lang`: Language code (default: `ta`)
- `--batch_size`: Batch size (default: `512`)
- `--dropout`: Dropout rate (default: `0.2`)
- `--embedding_dim`: Embedding dimension (default: `512`)
- `--epochs`: Number of epochs (default: `20`)
- `--hidden_dim`: Hidden dimension (default: `384`)
- `--learning_rate`: Learning rate (default: `0.0005`)
- `--num_layers`: Number of RNN layers (default: `6`)
- `--num_workers`: Number of data loader workers (default: `32`)
- `--project`: Wandb project name (default: `DA24S023_DA6401_A3`)
- `--rnn_cell`: RNN cell type (`GRU`, `LSTM`, or `RNN`; default: `GRU`)
- `--attention_type`: Attention mechanism (`luong`, `bahdanau`, or `none`; default: `none`)
- `--train_teacher_forcing_ratio`: Teacher forcing ratio (default: `0.5`)

---

## Evaluation

To evaluate the model on the test split:

```bash
python test.py --config model_configs/<config_file_name>.yaml
```

The checkpoint files follow a naming scheme derived from the hparams specified in the yaml config.

---

## Example Predictions

Once a model is trained, you can use it to generate predictions by modifying `test.py` or by exporting a notebook. Here’s a sample output for a trained Bahdanau RNN model:

```text
Input: duuglas
Bahdanau Prediction: டக்ளஸ்
Correct: டக்ளஸ்
```

---

## Hyperparameter Tuning

You can run a hyperparameter search using:

```bash
python hparam_search.py
```

The sweep configuration used is:

```python
sweep_configuration = {
    'method': 'bayes',
    'name': 'DA24S023_DA6401_A3',
    'metric': {'name': 'val/word_accuracy', 'goal': 'maximize'},
    'parameters': {
        'embedding_dim': {'values': [192, 256, 384, 512]},
        'hidden_dim': {'values': [384, 512, 768, 1024]},
        'num_layers': {'values': [4, 6, 8, 10]},
        'rnn_cell': {'values': ['RNN', 'LSTM', 'GRU']},
        'dropout': {'values': [0.2, 0.3, 0.4]},
        'learning_rate': {'values': [5e-3, 1e-3, 5e-4]},
        'train_teacher_forcing_ratio': {'values': [0.3, 0.5, 0.7]},
    }
}
```

You can control sweep runs and other options with the following arguments:

- `-sc, --sweep_count`: Number of Wandb sweeps (default: 50)
- `-l, --lang`: Language to train (default: ta). Choices: hi, bn, gu, kn, ml, or, pa, ta, te
- `-p, --project`: Wandb project name (default: DA24S023_DA6401_A3)
- `-b, --batch_size`: Batch size for training (default: 512)
- `-nw, --num_workers`: Number of workers for data loading (default: 32)
- `-ep, --epochs`: Number of epochs for training (default: 20)
- `-atn, --attention`: Attention mechanism to use (default: none). Choices: bahdanau, luong, none

---

## GPU Selection and Multi-GPU Usage

This project supports training, validation, testing, and inference on CUDA GPUs, including multi-GPU setups.

> **Note:**
> - If both `CUDA_VISIBLE_DEVICES` is set in the environment and `--cuda_devices`/`cuda_devices` is provided, the config/CLI value will override the environment.
> - For multi-GPU training, PyTorch Lightning will automatically use all visible GPUs. You can control the number of devices with the `--devices` argument if needed.
> - The code uses `auto` device selection in Lightning Trainer, so **any backend supported by your installed PyTorch should work** (e.g., CUDA, Apple Silicon Metal, etc.).
> - This codebase was tested on both CUDA (NVIDIA GPUs) and Apple Silicon Metal (M1/M2 Macs).
> - **Theoretically, it should also work on Intel Macs with dedicated AMD GPUs that support Metal,** but PyTorch must be built from source for those devices, which is typically not worth the trouble (see [PyTorch Metal backend docs](https://pytorch.org/docs/stable/notes/mps.html)).

### 1. Using CUDA_VISIBLE_DEVICES (Recommended)

You can select which GPUs to use by setting the `CUDA_VISIBLE_DEVICES` environment variable in your command line before running any script:

```bash
# Use only GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py --config model_configs/bahdanau_rnn.yaml

# Use GPUs 0 and 1 (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 python train.py --config model_configs/bahdanau_rnn.yaml

# For testing/inference:
CUDA_VISIBLE_DEVICES=1 python test.py --config model_configs/bahdanau_rnn.yaml
```

This works for all scripts: `train.py`, `test.py`, and `hparam_search.py`.

### 2. Setting GPUs via Config or CLI (Optional)

You can also specify GPUs to use via a command-line argument or YAML config. For example:

```bash
python train.py --config model_configs/bahdanau_rnn.yaml --cuda_devices 0,1
```

Or in your YAML config:

```yaml
cuda_devices: 0,1
```

If set, the code will set `os.environ["CUDA_VISIBLE_DEVICES"]` before initializing PyTorch Lightning. This allows for reproducible GPU selection from config files or sweeps.

> **Note:**
> - For multi-GPU training, the code uses `auto` device selection, so PyTorch Lightning will automatically use all visible GPUs. As mentioned earlier, you can control the number of devices with the `CUDA_VISIBLE_DEVICES` variable.
> - The code uses `auto` accelerator selection in Lightning Trainer, so **any backend supported by your installed PyTorch should work** (e.g., CUDA, Apple Silicon Metal, etc.).
> - This codebase was tested on both CUDA (NVIDIA GPUs) and Apple Silicon Metal (M1/M2 Macs).
> - **Theoretically, it should also work on Intel Macs with dedicated AMD GPUs that support Metal,** but PyTorch must be built from source for those devices, which is typically not worth the trouble (see [PyTorch Metal backend docs](https://pytorch.org/docs/stable/notes/mps.html)).

---

### Directory Structure

```
├── datamodule.py            # Data loading and processing
├── model.py                 # Seq2Seq model architecture
├── train.py                 # Training loop
├── test.py                  # Evaluation script
├── hparam_search.py         # Hyperparameter tuning
├── model_configs/           # YAML configuration files
├── fonts/                   # Tamil font for rendering predictions (optional)
├── requirements.txt         # Python dependencies
├── data/                    # Downloaded datasets and checkpoints (created automatically, gitignored)
├── lightning_logs/          # PyTorch Lightning logs (created automatically, gitignored)
├── wandb/                   # Wandb run logs (created automatically, gitignored)
├── predictions_*            # Output prediction folders (created automatically, gitignored)
```

> **Note:**  
> Some directories (like `data/`, `lightning_logs/`, `wandb/`, and `predictions_*`) are created and used automatically as you run training, testing, or logging scripts. These are included in `.gitignore` and will not appear in a fresh clone until you run the code.

---

## License

This repository is part of an academic assignment. Please check licensing and usage terms if you intend to reuse or distribute.
