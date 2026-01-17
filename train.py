"""Training pipeline for QD-MSA model on ISIC2017 dataset."""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing import get_train_transforms, get_val_transforms
from data.dataset import create_data_loaders

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for QD-MSA model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
    ):
        """Initialize the trainer.

        Args:
            model: The model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: How often to log training progress.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}', leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Validate the model.

        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation', leave=False):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train.
            early_stopping_patience: Number of epochs without improvement before stopping.

        Returns:
            Dictionary containing training history.
        """
        patience_counter = 0

        logger.info("Starting training for %d epochs", num_epochs)

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log progress
            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.2f%%, "
                "Val Loss: %.4f, Val Acc: %.2f%%, LR: %.6f",
                self.current_epoch, num_epochs, train_loss, train_acc,
                val_loss, val_acc, current_lr
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
                logger.info("New best model saved with Val Acc: %.2f%%", val_acc)
            else:
                patience_counter += 1

            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d", self.current_epoch)
                break

        # Save final model
        self.save_checkpoint('final_model.pth')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
        }

    def save_checkpoint(self, filename: str):
        """Save a checkpoint.

        Args:
            filename: Name of the checkpoint file.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load a checkpoint.

        Args:
            filename: Name of the checkpoint file.
        """
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info("Loaded checkpoint from epoch %d", self.current_epoch)


def create_model(
    input_channels: int = 3,
    num_classes: int = 3,
    model_type: str = 'split',
    hidden_dim: int = 512,
    with_shortcut: bool = True,
    device: str = 'cuda',
) -> nn.Module:
    """Create a QD-MSA model for image classification.

    This creates a simplified model architecture for single-modality
    image classification using the quantum fusion module.

    Args:
        input_channels: Number of input image channels.
        num_classes: Number of output classes.
        model_type: 'split' or 'unsplit' for quantum model type.
        hidden_dim: Hidden dimension for the model.
        with_shortcut: Whether to use shortcut connections.
        device: Device to create model on.

    Returns:
        The created model.
    """
    from models.quantum_split_model import QNNSplited
    from models.quantum_unsplited_model import QNNUnsplitted

    # Simple CNN feature extractor for images
    class ImageEncoder(nn.Module):
        """Simple CNN encoder for images."""

        def __init__(self, output_dim: int = 256):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(256, output_dim)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Create encoder
    encoder = ImageEncoder(output_dim=hidden_dim)

    # Create quantum head
    if model_type == 'split':
        head = QNNSplited(
            input_shape=hidden_dim,
            output_shape=num_classes,
            hidden_dim=hidden_dim,
            with_shortcut=with_shortcut,
        )
    else:
        head = QNNUnsplitted(
            input_shape=hidden_dim,
            output_shape=num_classes,
            hidden_dim=hidden_dim,
            with_shortcut=with_shortcut,
        )

    # Combined model
    class QDMSAImageClassifier(nn.Module):
        """QD-MSA model for image classification."""

        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            features = self.encoder(x)
            output = self.head(features)
            return output

    model = QDMSAImageClassifier(encoder, head)
    return model.to(device)


def train_model(
    data_root: str,
    model_type: str = 'split',
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    checkpoint_dir: str = 'checkpoints',
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Train the QD-MSA model.

    Args:
        data_root: Root directory of the ISIC2017 dataset.
        model_type: 'split' or 'unsplit'.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        device: Device to train on.
        checkpoint_dir: Directory to save checkpoints.
        num_workers: Number of data loading workers.

    Returns:
        Training history dictionary.
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
        test_transform=get_val_transforms(),
    )

    # Create model
    model = create_model(
        num_classes=3,  # ISIC2017 has 3 classes
        model_type=model_type,
        device=device,
    )

    # Create optimizer and criterion
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    history = trainer.train(num_epochs=num_epochs)

    return history


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Train QD-MSA model')
    parser.add_argument('--data-root', type=str, required=True, help='Path to ISIC2017 dataset')
    parser.add_argument('--model-type', type=str, default='split', choices=['split', 'unsplit'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()

    history = train_model(
        data_root=args.data_root,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"Training complete. Best validation accuracy: {history['best_val_acc']:.2f}%")
