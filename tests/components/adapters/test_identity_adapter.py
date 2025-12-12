import pytest
import torch
from satipatthana.components.adapters import IdentityAdapter
from satipatthana.configs.adapters import IdentityAdapterConfig


def test_identity_adapter_initialization():
    config = IdentityAdapterConfig(input_dim=2)
    adapter = IdentityAdapter(config)
    assert isinstance(adapter, IdentityAdapter)
    assert adapter.dim == 2
    assert config.dim == config.input_dim


def test_identity_adapter_forward_pass():
    config = IdentityAdapterConfig(input_dim=8)
    adapter = IdentityAdapter(config)
    batch_size = 4
    input_tensor = torch.randn(batch_size, config.input_dim)
    output = adapter(input_tensor)
    assert output.shape == (batch_size, config.dim)
    assert torch.allclose(input_tensor, output)


def test_identity_adapter_batch_size_one():
    config = IdentityAdapterConfig(input_dim=3)
    adapter = IdentityAdapter(config)
    input_tensor = torch.randn(1, 3)
    output = adapter(input_tensor)
    assert output.shape == (1, 3)
    assert torch.allclose(input_tensor, output)


def test_identity_adapter_missing_input_dim_raises_error():
    with pytest.raises(TypeError):
        IdentityAdapterConfig()
