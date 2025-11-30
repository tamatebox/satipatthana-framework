import pytest
import torch
from src.components.adapters import MlpAdapter


def test_mlp_adapter_initialization():
    config = {"dim": 32, "input_dim": 100, "adapter_hidden_dim": 64}
    adapter = MlpAdapter(config)
    assert isinstance(adapter, MlpAdapter)
    assert adapter.dim == 32
    assert adapter.input_dim == 100
    # Check if the network layers are correctly built
    assert len(adapter.net) == 8  # 3 Linear, 2 LayerNorm, 2 ReLU, 1 Tanh


def test_mlp_adapter_forward_pass():
    config = {"dim": 32, "input_dim": 100}
    adapter = MlpAdapter(config)
    batch_size = 4
    input_tensor = torch.randn(batch_size, config["input_dim"])
    output = adapter(input_tensor)
    assert output.shape == (batch_size, config["dim"])
    # Check Tanh activation output range
    assert torch.all((output >= -1.0) & (output <= 1.0))


def test_mlp_adapter_error_on_missing_input_dim():
    config = {"dim": 32}
    with pytest.raises(ValueError, match="MlpAdapter requires 'input_dim' in config."):
        MlpAdapter(config)
