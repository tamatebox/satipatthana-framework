import pytest
import torch
from samadhi.components.decoders.sequence import LstmDecoder, SimpleSequenceDecoder


@pytest.fixture
def sequence_decoder_config():
    return {
        "dim": 32,  # Latent dimension
        "input_dim": 10,  # Output features per timestep (for reconstruction, typically matches input_dim)
        "seq_len": 20,  # Sequence length
        "adapter_hidden_dim": 64,
        "lstm_layers": 1,
    }


def test_lstm_decoder_init(sequence_decoder_config):
    decoder = LstmDecoder(sequence_decoder_config)
    assert decoder is not None
    assert decoder.dim == sequence_decoder_config["dim"]


def test_lstm_decoder_forward(sequence_decoder_config):
    decoder = LstmDecoder(sequence_decoder_config)

    batch_size = 2
    # Latent vector (Batch, Latent Dim)
    s = torch.randn(batch_size, sequence_decoder_config["dim"])

    output = decoder(s)

    # Output should be (Batch, Seq_Len, Output_Dim)
    expected_shape = (
        batch_size,
        sequence_decoder_config["seq_len"],
        sequence_decoder_config["input_dim"],
    )
    assert output.shape == expected_shape


def test_simple_sequence_decoder_init(sequence_decoder_config):
    decoder = SimpleSequenceDecoder(sequence_decoder_config)
    assert decoder is not None
    assert decoder.dim == sequence_decoder_config["dim"]


def test_simple_sequence_decoder_forward(sequence_decoder_config):
    decoder = SimpleSequenceDecoder(sequence_decoder_config)

    batch_size = 2
    # Latent vector (Batch, Latent Dim)
    s = torch.randn(batch_size, sequence_decoder_config["dim"])

    output = decoder(s)

    # Output should be (Batch, Seq_Len, Output_Dim)
    expected_shape = (
        batch_size,
        sequence_decoder_config["seq_len"],
        sequence_decoder_config["input_dim"],
    )
    assert output.shape == expected_shape
