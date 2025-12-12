import pytest
import torch
from satipatthana.components.adapters.sequence import LstmAdapter, TransformerAdapter


@pytest.fixture
def sequence_adapter_config():
    return {
        "dim": 32,  # Latent dimension
        "input_dim": 10,  # Features per timestep
        "seq_len": 20,  # Sequence length
        "adapter_hidden_dim": 64,
        "lstm_layers": 1,
        "transformer_layers": 1,
        "transformer_heads": 2,
    }


def test_lstm_adapter_init(sequence_adapter_config):
    adapter = LstmAdapter(sequence_adapter_config)
    assert adapter is not None
    assert adapter.dim == sequence_adapter_config["dim"]


def test_lstm_adapter_forward(sequence_adapter_config):
    adapter = LstmAdapter(sequence_adapter_config)

    batch_size = 2
    # Input sequence (Batch, Seq_Len, Input_Dim)
    x = torch.randn(batch_size, sequence_adapter_config["seq_len"], sequence_adapter_config["input_dim"])

    z = adapter(x)

    # Output should be (Batch, Latent Dim)
    assert z.shape == (batch_size, sequence_adapter_config["dim"])
    assert torch.all(z >= -1.0) and torch.all(z <= 1.0)  # Due to Tanh activation


def test_transformer_adapter_init(sequence_adapter_config):
    adapter = TransformerAdapter(sequence_adapter_config)
    assert adapter is not None
    assert adapter.dim == sequence_adapter_config["dim"]


def test_transformer_adapter_forward(sequence_adapter_config):
    adapter = TransformerAdapter(sequence_adapter_config)

    batch_size = 2
    # Input sequence (Batch, Seq_Len, Input_Dim)
    x = torch.randn(batch_size, sequence_adapter_config["seq_len"], sequence_adapter_config["input_dim"])

    z = adapter(x)

    # Output should be (Batch, Latent Dim)
    assert z.shape == (batch_size, sequence_adapter_config["dim"])
    assert torch.all(z >= -1.0) and torch.all(z <= 1.0)  # Due to Tanh activation
