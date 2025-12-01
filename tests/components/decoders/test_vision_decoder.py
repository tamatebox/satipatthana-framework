import pytest
import torch
from samadhi.components.decoders.vision import CnnDecoder


@pytest.fixture
def cnn_decoder_config():
    return {
        "dim": 32,  # Latent dimension
        "channels": 3,
        "img_size": 32,
        "type": "cnn",  # Explicitly set type
    }


def test_cnn_decoder_init(cnn_decoder_config):
    decoder = CnnDecoder(cnn_decoder_config)
    assert decoder is not None
    assert decoder.dim == cnn_decoder_config["dim"]


def test_cnn_decoder_forward(cnn_decoder_config):
    decoder = CnnDecoder(cnn_decoder_config)

    batch_size = 2
    # Latent vector (Batch, Latent Dim)
    s = torch.randn(batch_size, cnn_decoder_config["dim"])

    output = decoder(s)

    # Output should be (Batch, Channels, Height, Width)
    expected_shape = (
        batch_size,
        cnn_decoder_config["channels"],
        cnn_decoder_config["img_size"],
        cnn_decoder_config["img_size"],
    )
    assert output.shape == expected_shape
    assert torch.all(output >= -1.0) and torch.all(
        output <= 1.0
    )  # Due to Tanh activation if present, or just range check
