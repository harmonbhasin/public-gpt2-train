import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim.lr_scheduler
from torch.cuda.amp import GradScaler

import wandb
from models.gpt2 import GPT
from src.data.data_loader import DataLoaderLite


class Trainer:
    def __init__(
        self,
        model: GPT,
        train_loader: DataLoaderLite,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: str,
        checkpoint_dir: str = 'checkpoints/',
        save_every: int = 10,
        patience: int = 5,
        min_delta: float = 0.001,
        validate_every: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every = save_every
        self.patience = patience
        self.min_delta = min_delta
        self.validate_every = validate_every

        # Create validation loader with same batch size and sequence length
        self.val_loader = DataLoaderLite(B=train_loader.B, T=train_loader.T, split='validation')
        # Initialize GradScaler only for CUDA devices
        self.scaler = GradScaler() if self.device != 'mps' else None

        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> float:
        """Run validation and return validation loss."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            x, y = self.val_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches

    def save_checkpoint(self, iter: int, loss: float, is_best: bool = False):
        """Save a checkpoint of the model and optimizer state."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.checkpoint_dir / f'model_iter{iter}_{timestamp}.pt'
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'

        checkpoint = {
            'iter': iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path: str) -> dict | None:
        """Load a checkpoint from disk."""
        if not os.path.exists(checkpoint_path):
            print(f'No checkpoint found at {checkpoint_path}')
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if (
            self.scheduler
            and 'scheduler_state_dict' in checkpoint
            and checkpoint['scheduler_state_dict'] is not None
        ):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if (
            self.scaler
            and 'scaler_state_dict' in checkpoint
            and checkpoint['scaler_state_dict'] is not None
        ):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint

    def train(self, num_iterations: int):
        """Train the model for the specified number of iterations."""
        for iter in range(num_iterations):
            # Training step
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            # Cast to fp16 to improve speed
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                logits, loss = self.model(x, y)
            # Scale the loss using GradScaler if available
            if self.scaler:
                self.scaler.scale(loss).backward()
                # Gradient norm clipping (unscale gradients first)
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            # Gradient norm clipping
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Step the optimizer (via scaler if available)
            if self.scaler:
                self.scaler.step(self.optimizer)
                # Update the scaler for the next iteration
                self.scaler.update()
            else:
                self.optimizer.step()
            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            # Calculate training perplexity
            train_perplexity = torch.exp(loss).item()

            # Log training metrics
            wandb.log(
                {
                    'train_loss': loss.item(),
                    'train_perplexity': train_perplexity,
                    'epoch': iter // len(self.train_loader),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.0),
                    'norm': norm,
                }
            )

            print(
                f'Iteration {iter}, Train Loss: {loss.item():.4f}, Train Perplexity: {train_perplexity:.4f}'
            )

            # Validation step (if it's time to validate)
            if (iter + 1) % self.validate_every == 0:
                val_loss = self.validate()

                # Calculate validation perplexity
                val_perplexity = torch.exp(torch.tensor(val_loss)).item()

                # Log validation metrics
                wandb.log({'val_loss': val_loss, 'val_perplexity': val_perplexity})
                print(
                    f'Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}'
                )

                # Early stopping check
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                    self.save_checkpoint(iter, val_loss, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f'Early stopping triggered after {iter + 1} iterations')
                        # Restore best model
                        self.model.load_state_dict(self.best_model_state)
                        break

            # Save checkpoint periodically
            if (iter + 1) % self.save_every == 0:
                self.save_checkpoint(iter, loss.item())
