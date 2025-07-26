# GPT from Scratch

This repository provides a minimal yet extensible implementation of the GPT architecture, enabling users to train, evaluate, and analyze transformer-based language models on custom datasets. It is designed for research, education, and small-scale experiments, with a focus on clarity and modularity.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Visualization](#visualization)
- [Examples](#examples)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Custom GPT Models**: Supports multiple model sizes (GPT2-14M, GPT2-29M, GPT2-49M), easily extensible.
- **Flexible Training**: Train on your own datasets with configurable hyperparameters in shell scripts.
- **Robust Inference**: Generate text with various sampling strategies and checkpoint selection.
- **Comprehensive Visualization**: Analyze metrics, activations, and attention maps to understand model behavior.
- **Modular Utilities**: Includes reusable utilities for data processing, logging, and parameter calculation.
- **Datasets Analysis**:Evaluate datasets metrics, including basic statistic features, sentence complexity, vocalbulary&domain diversity.

## Project Structure

- `train.py`: Main script for training GPT models.
- `inference.py`: Script for running inference with trained models.
- `scripts/`: Shell scripts for streamlined training and inference workflows.
- `utils/`: Utility modules for argument parsing, data loading, logging, learning rate scheduling, parameter calculation, and tokenization.
- `visualize/`: Python scripts for visualizing activations, attention, and metrics.
- `data/`: Contains datasets and tokenized data for training and validation.
- `logs/`: Stores training logs, metrics, and model checkpoints.
- `models/`: Implementation of the GPT model.
- `report/`: LaTeX report and documentation.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (recommended for training, optional for inference)
- [PyTorch](https://pytorch.org/) and other dependencies listed in `requirements.txt`

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/GPTfromScratch.git
cd GPTfromScratch
pip install -r requirements.txt
```

## Training

To train a GPT model, use the provided shell scripts or run `train.py` directly. Example for training the 14M model:

```bash
bash scripts/train_14M.sh
```

Or customize training via Python:

```bash
python train.py --model GPT2-14M --epochs 10 --batch_size 256 --lr 3e-4 --data_path data/tinystories/tokenized_train_bs256
```

Checkpoints and logs will be saved in `logs/GPT2-14M/ckpts` and `logs/GPT2-14M/train.log`.

## Inference

Generate text using a trained model checkpoint:

```bash
bash scripts/inference.sh -m GPT2-14M -p "Once upon a time" -l 100
```

Or use Python directly:

```bash
python inference.py --model_path logs/GPT2-14M/ckpts/best_model.pth --prompt "In a distant galaxy" --max_length 50
```

You can list available checkpoints:

```bash
bash scripts/inference.sh --list-ckpts -m GPT2-29M
```

## Visualization

Visualize training metrics, activations, or attention maps to better understand model performance:

```bash
python utils/visualize_metrics.py
python utils/visualize_activations.py
python utils/visualize_attention.py
```

Outputs are saved in the `visualize/` directory and can be used for further analysis or reporting.

