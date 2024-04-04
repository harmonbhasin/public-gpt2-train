# GPT-2 Training from Scratch

A learning implementation of GPT-2 training from scratch (inspired by [NanoGPT](https://github.com/karpathy/nanoGPT)) on WikiText-103. This project demonstrates the ability to implement transformer architectures and set up modern ML training workflows.

## Project Overview

This is a **demonstration project** showcasing:
- Implementation of the GPT-2 architecture from scratch
- Modern ML engineering practices and tooling
- Successful training and inference pipeline
- Clean, modular code organization

## Key Features

- **GPT-2 Architecture**: Full transformer implementation with multi-head attention and positional embeddings
- **Modern ML Stack**: Hydra configuration, Weights & Biases tracking, UV package management
- **Training Pipeline**: Mixed precision training, gradient clipping, learning rate scheduling
- **Clean Code**: Modular design with type hints and clear separation of concerns

## Quick Start

### Installation

```bash
# Clone and setup environment
git clone <repo-url>
cd public-gpt2-train

# Create virtual environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Training

```bash
# Train with default configuration
python train.py

# Override specific parameters
python train.py model.model.n_layer=6 model.model.n_head=8 model.training.batch_size=32

# Use predefined configs
python train.py model=bigger_default experiment=default
```

Training automatically downloads WikiText-103 and logs metrics to Weights & Biases.

### Configuration

The project uses Hydra for hierarchical configuration:

```yaml
# configs/model/default.yaml
model:
  vocab_size: 50257
  n_positions: 1024
  n_embd: 768
  n_layer: 12
  n_head: 12
  
optimizer:
  lr: 3e-4
  weight_decay: 0.01
  
training:
  batch_size: 4
  sequence_length: 64
  checkpoint_frequency: 50
```

Override any parameter via CLI or create custom config files in `configs/model/` and `configs/experiment/`.

### Inference

Deploy trained models for text generation with configurable sampling strategies:

```bash
# Standard inference with checkpoint path
python inference.py path/to/checkpoint.pt

# Generate with custom prompt and length constraints
python inference.py <checkpoint_path> --prompt "The future of AI is" --max_length 100

# Configure sampling parameters for generation quality
python inference.py <checkpoint_path> --temperature 0.8 --top_k 40
```

**Inference Parameters:**
- `--temperature`: Controls randomness (0.0-1.0, higher = more diverse)
- `--top_k`: Limits vocabulary for each prediction (higher = more options)
- `--max_length`: Maximum sequence length for generation

## Understanding Hydra Configuration

This project uses [Hydra](https://hydra.cc) for configuration management, which provides:
- Hierarchical configuration with YAML files
- Command-line overrides for any parameter
- Configuration composition from multiple files
- Automatic output directory management

### Viewing Available Options

```bash
# See all configuration options and current values
python train.py --help

# View available configuration groups
python train.py --help
# Output shows:
# model: bigger_default, default, test
# experiment: default
```

### Configuration Structure

The configuration is composed from multiple files:
- `configs/config.yaml` - Base configuration
- `configs/model/*.yaml` - Model architecture presets
- `configs/experiment/*.yaml` - Training experiment settings

### Common Configuration Overrides

```bash
# Change any parameter using dot notation
python train.py model.model.n_layer=6 model.optimizer.lr=1e-4

# Use different model presets
python train.py model=test              # Small model for testing
python train.py model=bigger_default     # Larger model with scheduler

# Combine model and experiment configs
python train.py model=bigger_default experiment=default

# Change multiple parameters
python train.py \
  model.training.batch_size=8 \
  model.training.sequence_length=512 \
  experiment.trainer.num_iterations=5000
```

### Key Configuration Parameters

- **Model Architecture** (`model.model.*`)
  - `vocab_size`: Vocabulary size (default: 50257)
  - `n_layer`: Number of transformer layers (default: 12)
  - `n_head`: Number of attention heads (default: 12)
  - `n_embd`: Embedding dimension (default: 768)

- **Training** (`model.training.*`)
  - `batch_size`: Training batch size (default: 4)
  - `sequence_length`: Context window size (default: 64)
  - `checkpoint_frequency`: Save model every N iterations (default: 50)

- **Optimizer** (`model.optimizer.*`)
  - `lr`: Learning rate (default: 3e-4)
  - `weight_decay`: L2 regularization (default: 0.01)

- **Experiment** (`experiment.trainer.*`)
  - `num_iterations`: Total training iterations (default: 10000)
  - `seed`: Random seed (default: 1337)

## Project Structure

```
├── configs/           # Hydra configuration files
├── models/           # Model architecture (GPT-2)
├── src/
│   ├── data/        # Data loading and tokenization
│   ├── training/    # Training loop and optimization
│   └── utils/       # Logging and helpers
├── train.py         # Main training script
└── inference.py     # Text generation script
```

## Technical Highlights

- **Scalable Architecture**: Modular design allows easy experimentation with model sizes
- **Efficient Data Loading**: Streaming dataset processing with HuggingFace datasets
- **Robust Training**: Gradient clipping, learning rate scheduling, automatic mixed precision
- **Comprehensive Logging**: Real-time metrics, loss curves, and sample generations in W&B

## Development

```bash
# Format code
ruff format .

# Run linter
ruff check .

# Run tests
python train.py model=test experiment=default
```

## Roadmap

### Code Quality & Testing
- [ ] **Comprehensive Testing**: Unit tests for model components, training loop, and utilities
- [ ] **Documentation**: Docstrings for all public methods and classes
- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing and linting
- [ ] **Error Handling**: Robust error handling for data loading and training edge cases

### Performance Optimizations
- [ ] **Distributed Training**: Multi-GPU support with PyTorch DDP for large-scale training
- [ ] **Flash Attention**: Integration of FlashAttention-2 for 2-4x training speedup
- [ ] **Gradient Accumulation**: Support for larger effective batch sizes on limited hardware

### Advanced Features
- [ ] **Fine-tuning Pipeline**: Domain adaptation with LoRA/QLoRA support
- [ ] **Extended Metrics**: BLEU, ROUGE, and perplexity tracking during training
- [ ] **Custom Datasets**: Easy integration of new text datasets beyond WikiText-103

### Infrastructure
- [ ] **Checkpoint Management**: Automatic model versioning and experiment tracking
- [ ] **Data Pipeline**: Support for custom datasets and streaming data loaders
- [ ] **Deployment Ready**: ONNX export and TensorRT optimization paths

