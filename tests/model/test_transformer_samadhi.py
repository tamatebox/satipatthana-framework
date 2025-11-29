import torch
import pytest

from src.model.transformer_samadhi import TransformerSamadhiModel
from src.components.vitakka import Vitakka
from src.components.vicara import VicaraBase


class MockConfig:
    def __init__(self):
        self.input_dim = 10
        self.dim = 10
        self.hidden_dim = 20
        self.latent_dim = 5
        self.num_heads = 2  # New for Transformer
        self.num_layers = 1  # Encoder layers
        self.dropout = 0.0
        self.activation = "relu"
        self.num_probes = 3
        self.n_probes = 3  # Added for components that expect 'n_probes'
        self.num_timesteps = 5
        self.gate_threshold = -1.0  # Always open for basic forward pass tests
        self.attention_mode = "hard"
        self.probe_trainable = True
        self.vicara_type = "shared"
        self.sequence_length = 7  # New for Transformer based models
        self.seq_len = 7  # Added for models that expect 'seq_len'
        self.refine_steps = 3

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def transformer_samadhi_instance():
    config = MockConfig()
    return TransformerSamadhiModel(config)


def test_transformer_samadhi_init(transformer_samadhi_instance):
    assert isinstance(transformer_samadhi_instance.vitakka, Vitakka)
    assert isinstance(transformer_samadhi_instance.vicara, VicaraBase)
    assert transformer_samadhi_instance.input_dim == 10
    assert transformer_samadhi_instance.config["num_probes"] == 3


def test_transformer_samadhi_forward_step(transformer_samadhi_instance):
    # Input for TransformerSamadhi should be (batch_size, sequence_length, input_dim)
    input_tensor = torch.randn(
        1, transformer_samadhi_instance.config["sequence_length"], transformer_samadhi_instance.input_dim
    )

    result = transformer_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is not None
    s_final, full_log = result

    # s_final is latent state (Dim)
    assert s_final.shape == (1, transformer_samadhi_instance.dim)

    # Test Decoder
    decoded_output = transformer_samadhi_instance.decoder(s_final)
    assert decoded_output.shape == input_tensor.shape

    assert isinstance(full_log, dict)
    assert "probe_log" in full_log


def test_transformer_samadhi_gating(transformer_samadhi_instance):
    transformer_samadhi_instance.config.gate_threshold = 100.0

    input_tensor = torch.randn(
        1, transformer_samadhi_instance.config["sequence_length"], transformer_samadhi_instance.input_dim
    )
    result = transformer_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is None

    transformer_samadhi_instance.config.gate_threshold = -1.0
