import pytest
import torch
from src.components.decoders import ReconstructionDecoder


def test_reconstruction_decoder_initialization():
    config = {"dim": 32, "input_dim": 100, "decoder_hidden_dim": 64}
    decoder = ReconstructionDecoder(config)
    assert isinstance(decoder, ReconstructionDecoder)
    assert decoder.dim == 32
    # Check if the network layers are correctly built
    assert len(decoder.net) == 7  # 3 Linear, 2 LayerNorm, 2 ReLU


def test_reconstruction_decoder_forward_pass():
    config = {"dim": 32, "input_dim": 100}
    decoder = ReconstructionDecoder(config)
    batch_size = 4
    input_latent_state = torch.randn(batch_size, config["dim"])
    output = decoder(input_latent_state)
    assert output.shape == (batch_size, config["input_dim"])


def test_reconstruction_decoder_error_on_missing_input_dim():
    config = {"dim": 32}
    with pytest.raises(
        ValueError, match=r"ReconstructionDecoder requires 'input_dim' in config \(target dimension\)."
    ):
        ReconstructionDecoder(config)
