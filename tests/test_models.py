"""Tests for model tensor matching and forward pass."""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCommonModels:
    """Tests for common model components."""
    
    def test_gru_forward_pass(self):
        """Test GRU forward pass with correct tensor shapes."""
        from models.common_models import GRU
        
        batch_size = 4
        seq_len = 10
        input_dim = 32
        hidden_dim = 64
        
        model = GRU(indim=input_dim, hiddim=hidden_dim, last_only=True)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        # When last_only=True, output should be (batch_size, hidden_dim)
        assert output.shape == (batch_size, hidden_dim), f"Expected shape {(batch_size, hidden_dim)}, got {output.shape}"
    
    def test_gru_with_dropout(self):
        """Test GRU with dropout enabled."""
        from models.common_models import GRU
        
        batch_size = 4
        seq_len = 10
        input_dim = 32
        hidden_dim = 64
        
        model = GRU(indim=input_dim, hiddim=hidden_dim, dropout=True, dropoutp=0.5)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        model.train()
        output = model(x)
        
        # Output should be (batch_size, seq_len, hidden_dim) when not last_only
        assert output.shape == (batch_size, seq_len, hidden_dim)
    
    def test_gru_with_linear_forward_pass(self):
        """Test GRUWithLinear forward pass."""
        from models.common_models import GRUWithLinear
        
        batch_size = 4
        seq_len = 10
        input_dim = 32
        hidden_dim = 64
        output_dim = 16
        
        model = GRUWithLinear(
            indim=input_dim, 
            hiddim=hidden_dim, 
            outdim=output_dim,
            batch_first=True
        )
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        # Output after linear should include output_dim in the shape
        assert output.shape[-1] == output_dim
    
    def test_concat_forward_pass(self):
        """Test Concat fusion module."""
        from models.common_models import Concat
        
        batch_size = 4
        dim1, dim2, dim3 = 32, 64, 16
        
        # Test with all modalities
        concat = Concat(masks=[0, 1, 2])
        modalities = [
            torch.randn(batch_size, dim1),
            torch.randn(batch_size, dim2),
            torch.randn(batch_size, dim3),
        ]
        
        output = concat(modalities)
        
        expected_dim = dim1 + dim2 + dim3
        assert output.shape == (batch_size, expected_dim)
    
    def test_concat_with_subset(self):
        """Test Concat with subset of modalities."""
        from models.common_models import Concat
        
        batch_size = 4
        dim1, dim2, dim3 = 32, 64, 16
        
        # Only use first two modalities
        concat = Concat(masks=[0, 1])
        modalities = [
            torch.randn(batch_size, dim1),
            torch.randn(batch_size, dim2),
            torch.randn(batch_size, dim3),
        ]
        
        output = concat(modalities)
        
        expected_dim = dim1 + dim2
        assert output.shape == (batch_size, expected_dim)


class TestQuantumModels:
    """Tests for quantum model components."""
    
    @pytest.mark.slow
    def test_qnn_splited_forward_pass(self):
        """Test QNNSplited forward pass with correct tensor shapes."""
        from models.quantum_split_model import QNNSplited
        
        batch_size = 2
        input_shape = 64
        output_shape = 3
        
        model = QNNSplited(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_dim=32,
            with_shortcut=False,
        )
        
        x = torch.randn(batch_size, input_shape)
        output = model(x)
        
        assert output.shape == (batch_size, output_shape), f"Expected {(batch_size, output_shape)}, got {output.shape}"
    
    @pytest.mark.slow
    def test_qnn_splited_with_shortcut(self):
        """Test QNNSplited with shortcut connection."""
        from models.quantum_split_model import QNNSplited
        
        batch_size = 2
        input_shape = 64
        output_shape = 3
        
        model = QNNSplited(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_dim=32,
            with_shortcut=True,
        )
        
        x = torch.randn(batch_size, input_shape)
        output = model(x)
        
        assert output.shape == (batch_size, output_shape)
    
    @pytest.mark.slow
    def test_qnn_unsplited_forward_pass(self):
        """Test QNNUnsplitted forward pass with correct tensor shapes."""
        from models.quantum_unsplited_model import QNNUnsplitted
        
        batch_size = 2
        input_shape = 64
        output_shape = 3
        
        model = QNNUnsplitted(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_dim=32,
            with_shortcut=False,
        )
        
        x = torch.randn(batch_size, input_shape)
        output = model(x)
        
        assert output.shape == (batch_size, output_shape)
    
    @pytest.mark.slow
    def test_qnn_unsplited_with_shortcut(self):
        """Test QNNUnsplitted with shortcut connection."""
        from models.quantum_unsplited_model import QNNUnsplitted
        
        batch_size = 2
        input_shape = 64
        output_shape = 3
        
        model = QNNUnsplitted(
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_dim=32,
            with_shortcut=True,
        )
        
        x = torch.randn(batch_size, input_shape)
        output = model(x)
        
        assert output.shape == (batch_size, output_shape)


class TestModelGradients:
    """Tests for model gradient flow."""
    
    def test_gru_gradient_flow(self):
        """Test that gradients flow through GRU."""
        from models.common_models import GRU
        
        model = GRU(indim=32, hiddim=64, last_only=True)
        x = torch.randn(4, 10, 32, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    @pytest.mark.slow
    def test_qnn_splited_gradient_flow(self):
        """Test that gradients flow through QNNSplited.
        
        Note: Some parameters like shared_weights may not receive gradients
        directly due to how PennyLane's TorchLayer handles weight sharing.
        We check that key trainable parameters (mlp layers) receive gradients.
        """
        from models.quantum_split_model import QNNSplited
        
        model = QNNSplited(input_shape=32, output_shape=3, hidden_dim=16)
        x = torch.randn(2, 32, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that at least the MLP parameters have gradients
        mlp_has_gradients = False
        for name, param in model.named_parameters():
            if 'mlp' in name and param.requires_grad and param.grad is not None:
                mlp_has_gradients = True
                break
        
        assert mlp_has_gradients, "MLP layers should have gradients"


class TestTensorMatching:
    """Tests for input/output tensor matching."""
    
    def test_mmdl_tensor_matching(self):
        """Test MMDL with multiple encoders has matching tensors."""
        from models.common_models import GRU, MMDL, Concat
        
        batch_size = 4
        seq_len = 10
        
        # Create encoders with different hidden dimensions
        encoder1 = GRU(indim=32, hiddim=64, last_only=True)
        encoder2 = GRU(indim=16, hiddim=32, last_only=True)
        
        # Fusion concatenates the outputs
        fusion = Concat(masks=[0, 1])
        
        # Head expects concatenated dimension
        head = torch.nn.Linear(64 + 32, 3)
        
        model = MMDL(
            encoders=[encoder1, encoder2],
            fusion=fusion,
            head=head,
            has_padding=False,
        )
        
        # Create inputs for each encoder
        inputs = [
            torch.randn(batch_size, seq_len, 32),
            torch.randn(batch_size, seq_len, 16),
        ]
        
        output = model(inputs)
        
        assert output.shape == (batch_size, 3)
    
    def test_encoder_output_dimensions(self):
        """Test that encoder output dimensions match fusion input."""
        from models.common_models import GRU
        
        batch_size = 4
        seq_len = 10
        hidden_dims = [64, 128, 256]
        
        for hidden_dim in hidden_dims:
            encoder = GRU(indim=32, hiddim=hidden_dim, last_only=True)
            x = torch.randn(batch_size, seq_len, 32)
            output = encoder(x)
            
            assert output.shape == (batch_size, hidden_dim), \
                f"Hidden dim {hidden_dim}: Expected {(batch_size, hidden_dim)}, got {output.shape}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
