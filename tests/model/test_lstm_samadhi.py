import torch
import pytest

from src.model.lstm_samadhi import LstmSamadhiModel
from src.components.vitakka import Vitakka
from src.components.vicara import VicaraBase


class MockConfig:
    def __init__(self):
        self.input_dim = 10
        self.dim = 10
        self.hidden_dim = 20
        self.latent_dim = 5
        self.num_layers = 2  # LSTM can have multiple layers
        self.dropout = 0.0
        self.activation = "relu"
        self.num_probes = 3
        self.n_probes = 3  # Added for components that expect 'n_probes'
        self.num_timesteps = 5
        self.gate_threshold = -1.0  # Always open for basic forward pass tests
        self.attention_mode = "hard"
        self.probe_trainable = True
        self.vicara_type = "shared"
        self.sequence_length = 7  # New for LSTM based models
        self.seq_len = 7  # Added for models that expect 'seq_len'
        self.refine_steps = 3

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def lstm_samadhi_instance():
    config = MockConfig()
    return LstmSamadhiModel(config)


def test_lstm_samadhi_init(lstm_samadhi_instance):
    assert isinstance(lstm_samadhi_instance.vitakka, Vitakka)
    assert isinstance(lstm_samadhi_instance.vicara, VicaraBase)
    assert lstm_samadhi_instance.input_dim == 10
    assert lstm_samadhi_instance.config["num_probes"] == 3


def test_lstm_samadhi_forward_step(lstm_samadhi_instance):
    # Input for LstmSamadhiModel is typically sequential
    # Assuming the input to forward_step is the entire sequence or a single timeframe depending on implementation
    # Let's assume it processes the sequence to get a latent state

    input_tensor = torch.randn(1, lstm_samadhi_instance.config["sequence_length"], lstm_samadhi_instance.input_dim)

    result = lstm_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is not None
    s_final, full_log = result

    # s_final is latent state (Dim)
    assert s_final.shape == (1, lstm_samadhi_instance.dim)

    # Test Decoder - output should be sequential reconstruction
    decoded_output = lstm_samadhi_instance.decoder(s_final)

    # Check if decoder output matches input shape (sequence reconstruction)
    # LstmSamadhiModel usually reconstructs the sequence
    assert decoded_output.shape == input_tensor.shape

    assert isinstance(full_log, dict)
    assert "probe_log" in full_log


def test_lstm_samadhi_gating(lstm_samadhi_instance):
    lstm_samadhi_instance.config.gate_threshold = 100.0

    input_tensor = torch.randn(1, lstm_samadhi_instance.config["sequence_length"], lstm_samadhi_instance.input_dim)
    result = lstm_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is None

    lstm_samadhi_instance.config.gate_threshold = -1.0
