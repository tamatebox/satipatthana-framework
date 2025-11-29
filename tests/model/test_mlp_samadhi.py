import torch
import pytest

from src.model.mlp_samadhi import MlpSamadhiModel
from src.components.vitakka import Vitakka
from src.components.vicara import VicaraBase


class MockConfig:
    def __init__(self):
        self.input_dim = 10
        self.dim = 10
        self.hidden_dim = 20
        self.latent_dim = 5
        self.num_layers = 1
        self.dropout = 0.0
        self.activation = "relu"
        self.num_probes = 3
        self.n_probes = 3  # Added for components that expect 'n_probes'
        self.num_timesteps = 5
        self.gate_threshold = -1.0  # Always open for basic forward pass tests
        self.attention_mode = "hard"
        self.probe_trainable = True
        self.vicara_type = "shared"
        self.refine_steps = 3
        self.adapter_hidden_dim = 16

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def mlp_samadhi_instance():
    config = MockConfig()
    return MlpSamadhiModel(config)


def test_mlp_samadhi_init(mlp_samadhi_instance):
    assert isinstance(mlp_samadhi_instance.vitakka, Vitakka)
    assert isinstance(mlp_samadhi_instance.vicara, VicaraBase)
    assert mlp_samadhi_instance.input_dim == 10
    assert mlp_samadhi_instance.config["num_probes"] == 3


def test_mlp_samadhi_forward_step(mlp_samadhi_instance):
    input_tensor = torch.randn(1, mlp_samadhi_instance.input_dim)

    # MlpSamadhiModel uses forward_step
    result = mlp_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is not None
    s_final, full_log = result

    # Output of forward_step is the purified state (Dim), not the decoded output (input_dim)
    # The decoder is separate.
    assert s_final.shape == (1, mlp_samadhi_instance.dim)

    # Test Decoder
    decoded_output = mlp_samadhi_instance.decoder(s_final)
    assert decoded_output.shape == input_tensor.shape

    assert isinstance(full_log, dict)
    assert "probe_log" in full_log
    assert "energies" in full_log
    assert "dynamics" in full_log


def test_mlp_samadhi_gating(mlp_samadhi_instance):
    # Temporarily set gate_threshold to a high value to force gating
    # MlpSamadhiModel reads config dict/object. Since we use a Mock object that acts like dict...
    mlp_samadhi_instance.config.gate_threshold = 100.0

    input_tensor = torch.randn(1, mlp_samadhi_instance.input_dim)
    result = mlp_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is None

    # Reset gate_threshold
    mlp_samadhi_instance.config.gate_threshold = -1.0
