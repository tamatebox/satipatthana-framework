import pytest
import torch
from satipatthana.components.adapters.vision import CnnAdapter


@pytest.fixture
def cnn_adapter_config():
    return {
        "dim": 32,  # Latent dimension
        "channels": 3,
        "img_size": 32,
    }


def test_cnn_adapter_init(cnn_adapter_config):
    adapter = CnnAdapter(cnn_adapter_config)
    assert adapter is not None
    assert adapter.dim == cnn_adapter_config["dim"]


def test_cnn_adapter_forward(cnn_adapter_config):
    adapter = CnnAdapter(cnn_adapter_config)

    batch_size = 2
    # Input image (Batch, Channels, Height, Width)
    x = torch.randn(
        batch_size, cnn_adapter_config["channels"], cnn_adapter_config["img_size"], cnn_adapter_config["img_size"]
    )

    z = adapter(x)

    # Output should be (Batch, Latent Dim)
    assert z.shape == (batch_size, cnn_adapter_config["dim"])
    assert torch.all(z >= -1.0) and torch.all(z <= 1.0)  # Due to Tanh activation
