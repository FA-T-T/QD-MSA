"""Tests for training and evaluation pipelines."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrainer:
    """Tests for the Trainer class."""
    
    def _create_dummy_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
    
    def _create_dummy_loaders(self, batch_size=4, num_samples=16):
        """Create dummy data loaders for testing."""
        # Create random data
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, 3, (num_samples,))
        
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader, loader  # Use same loader for train and val
    
    def test_trainer_initialization(self):
        """Test Trainer can be initialized."""
        from train import Trainer
        
        model = self._create_dummy_model()
        train_loader, val_loader = self._create_dummy_loaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            device='cpu',
        )
        
        assert trainer.model is not None
        assert trainer.current_epoch == 0
        assert trainer.best_val_acc == 0.0
    
    def test_trainer_train_epoch(self):
        """Test single epoch training."""
        from train import Trainer
        
        model = self._create_dummy_model()
        train_loader, val_loader = self._create_dummy_loaders()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters()),
                device='cpu',
                checkpoint_dir=tmpdir,
            )
            
            trainer.current_epoch = 1
            loss, acc = trainer.train_epoch()
            
            assert isinstance(loss, float)
            assert isinstance(acc, float)
            assert 0 <= acc <= 100
    
    def test_trainer_validate(self):
        """Test validation."""
        from train import Trainer
        
        model = self._create_dummy_model()
        train_loader, val_loader = self._create_dummy_loaders()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters()),
                device='cpu',
                checkpoint_dir=tmpdir,
            )
            
            loss, acc = trainer.validate()
            
            assert isinstance(loss, float)
            assert isinstance(acc, float)
            assert 0 <= acc <= 100
    
    def test_trainer_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        from train import Trainer
        
        model = self._create_dummy_model()
        train_loader, val_loader = self._create_dummy_loaders()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters()),
                device='cpu',
                checkpoint_dir=tmpdir,
            )
            
            # Set some state
            trainer.current_epoch = 5
            trainer.best_val_acc = 75.0
            trainer.train_losses = [1.0, 0.8, 0.6]
            
            # Save checkpoint
            trainer.save_checkpoint('test_checkpoint.pth')
            
            # Create new trainer and load
            model2 = self._create_dummy_model()
            trainer2 = Trainer(
                model=model2,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model2.parameters()),
                device='cpu',
                checkpoint_dir=tmpdir,
            )
            
            trainer2.load_checkpoint('test_checkpoint.pth')
            
            assert trainer2.current_epoch == 5
            assert trainer2.best_val_acc == 75.0
            assert trainer2.train_losses == [1.0, 0.8, 0.6]
    
    def test_trainer_train_multiple_epochs(self):
        """Test training for multiple epochs."""
        from train import Trainer
        
        model = self._create_dummy_model()
        train_loader, val_loader = self._create_dummy_loaders()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters()),
                device='cpu',
                checkpoint_dir=tmpdir,
            )
            
            history = trainer.train(num_epochs=3, early_stopping_patience=100)
            
            assert 'train_losses' in history
            assert 'val_losses' in history
            assert 'train_accs' in history
            assert 'val_accs' in history
            assert len(history['train_losses']) == 3


class TestEvaluator:
    """Tests for the Evaluator class."""
    
    def _create_dummy_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
    
    def _create_dummy_loader(self, batch_size=4, num_samples=16):
        """Create dummy data loader for testing."""
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, 3, (num_samples,))
        
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return loader
    
    def test_evaluator_initialization(self):
        """Test Evaluator can be initialized."""
        from test import Evaluator
        
        model = self._create_dummy_model()
        loader = self._create_dummy_loader()
        
        evaluator = Evaluator(model=model, test_loader=loader, device='cpu')
        
        assert evaluator.model is not None
    
    def test_evaluator_evaluate(self):
        """Test evaluation produces correct metrics."""
        from test import Evaluator
        
        model = self._create_dummy_model()
        loader = self._create_dummy_loader()
        
        evaluator = Evaluator(model=model, test_loader=loader, device='cpu')
        
        metrics = evaluator.evaluate()
        
        # Check all expected metrics are present
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
    
    def test_evaluator_print_results(self, capsys):
        """Test result printing."""
        from test import Evaluator
        
        model = self._create_dummy_model()
        loader = self._create_dummy_loader()
        
        evaluator = Evaluator(model=model, test_loader=loader, device='cpu')
        metrics = evaluator.evaluate()
        
        evaluator.print_results(metrics)
        
        captured = capsys.readouterr()
        assert 'EVALUATION RESULTS' in captured.out
        assert 'Accuracy' in captured.out


class TestCreateModel:
    """Tests for model creation function."""
    
    def test_create_split_model(self):
        """Test creating split quantum model."""
        from train import create_model
        
        model = create_model(
            input_channels=3,
            num_classes=3,
            model_type='split',
            hidden_dim=64,
            device='cpu',
        )
        
        assert model is not None
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 3)
    
    def test_create_unsplit_model(self):
        """Test creating unsplit quantum model."""
        from train import create_model
        
        model = create_model(
            input_channels=3,
            num_classes=3,
            model_type='unsplit',
            hidden_dim=64,
            device='cpu',
        )
        
        assert model is not None
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
