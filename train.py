import os

import hydra
import torch
import torch.optim.lr_scheduler
from omegaconf import DictConfig, OmegaConf

import wandb
from models.gpt2 import GPT, GPTConfig
from src.data.data_loader import DataLoaderLite
from src.training.trainer import Trainer
from src.utils.utils import get_device


@hydra.main(version_base=None, config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    # Set random seed for reproducibility
    torch.manual_seed(cfg.experiment.trainer.seed)

    # Initialize wandb
    wandb.init(
        project='exp1-wikitext-103',
        config={
            'learning_rate': cfg.model.optimizer.lr,
            'batch_size': cfg.model.training.batch_size,
            'sequence_length': cfg.model.training.sequence_length,
            'vocab_size': cfg.model.model.vocab_size,
            'n_positions': cfg.model.model.n_positions,
            'n_embd': cfg.model.model.n_embd,
            'n_layer': cfg.model.model.n_layer,
            'n_head': cfg.model.model.n_head,
            'num_iterations': cfg.experiment.trainer.num_iterations,
            'seed': cfg.experiment.trainer.seed,
            'weight_decay': cfg.model.optimizer.weight_decay,
            'dropout': cfg.model.model.dropout,
            'validate_every': cfg.experiment.trainer.validate_every,
            'metrics': ['loss', 'perplexity'],
        },
    )

    # Create output directories
    os.makedirs(cfg.outputs.dir, exist_ok=True)
    for subdir in cfg.outputs.subdirs.values():
        os.makedirs(os.path.join(cfg.outputs.dir, subdir), exist_ok=True)

    # Get device
    device = get_device()
    print(f'Using device: {device}')

    # Initialize data loader
    train_loader = DataLoaderLite(
        B=cfg.model.training.batch_size,
        T=cfg.model.training.sequence_length,
    )

    # Initialize model
    model_config = GPTConfig(
        vocab_size=cfg.model.model.vocab_size,
        block_size=cfg.model.model.n_positions,
        n_embd=cfg.model.model.n_embd,
        n_layer=cfg.model.model.n_layer,
        n_head=cfg.model.model.n_head,
        dropout=cfg.model.model.dropout,
    )
    model = GPT(model_config)
    model.to(device)
    # Doesn't work in mps due to dropout
    if device != 'mps':
        model = torch.compile(model)

    # Log model architecture to wandb
    wandb.watch(model)

    # Initialize optimizer using Hydra
    # Ensure the optimizer config has _target_
    optimizer_cfg = cfg.model.optimizer
    if '_target_' not in optimizer_cfg:
        raise ValueError("Optimizer config must have a '_target_' field")
    optimizer = hydra.utils.instantiate(optimizer_cfg, params=model.parameters())
    print(f'Initialized optimizer: {optimizer}')

    # Initialize scheduler
    scheduler = None
    if cfg.model.scheduler.enabled:
        scheduler_cfg = cfg.model.scheduler
        if '_target_' not in scheduler_cfg:
            raise ValueError("Scheduler config must have a '_target_' field")
        # Remove 'enabled' flag before instantiation if it exists
        scheduler_conf = OmegaConf.to_container(scheduler_cfg, resolve=True)
        scheduler_conf.pop('enabled', None)
        # Instantiate the scheduler
        scheduler = hydra.utils.instantiate(scheduler_conf, optimizer=optimizer)
        print(f'Initialized scheduler: {scheduler}')

    # Lower precision, https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision('high')

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=os.path.join(cfg.outputs.dir, cfg.outputs.subdirs.checkpoints),
        save_every=cfg.model.training.checkpoint_frequency,
        validate_every=cfg.experiment.trainer.validate_every,
    )

    # Train the model
    trainer.train(num_iterations=cfg.experiment.trainer.num_iterations)

    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
