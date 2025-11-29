import torch
import torch.nn as nn
import pytest

from src.model.samadhi import SamadhiModel
from src.components.vitakka import Vitakka
from src.components.vicara import VicaraBase


class MockConfig:
    def __init__(self):
        self.input_dim = 10
        self.dim = 10  # SamadhiModel uses 'dim'
        self.hidden_dim = 20
        self.latent_dim = 5
        self.num_layers = 1
        self.dropout = 0.0
        self.activation = "relu"
        self.num_probes = 3
        self.n_probes = 3  # Added for components that expect 'n_probes'
        self.num_timesteps = 5
        self.gate_threshold = 0.5
        self.attention_mode = "hard"  # For inference
        self.probe_trainable = True
        self.vicara_type = "shared"  # For SamadhiModel
        self.refine_steps = 3  # For Vicara

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def samadhi_core_instance():
    config = MockConfig()
    return SamadhiModel(config)


@pytest.fixture
def samadhi_core_instance_soft_attention():
    config = MockConfig()
    config.attention_mode = "soft"
    return SamadhiModel(config)


@pytest.fixture
def samadhi_core_instance_gated():
    config = MockConfig()
    config.gate_threshold = 100.0  # Make gate always closed
    return SamadhiModel(config)


@pytest.fixture
def samadhi_core_instance_no_gate():
    config = MockConfig()
    config.gate_threshold = -1.0  # Make gate always open
    return SamadhiModel(config)


def test_samadhi_core_init(samadhi_core_instance):
    assert isinstance(samadhi_core_instance.vitakka, Vitakka)
    assert isinstance(samadhi_core_instance.vicara, VicaraBase)
    # SamadhiModel doesn't store num_probes/timesteps directly as attributes in the same way SamadhiCore might have
    # Checking via config or sub-components
    assert samadhi_core_instance.config["num_probes"] == 3
    assert samadhi_core_instance.config["num_timesteps"] == 5
    assert samadhi_core_instance.config["gate_threshold"] == 0.5


def test_samadhi_core_forward_step(samadhi_core_instance_no_gate):
    # Initial state (s_t) and input (x)
    # In SamadhiModel, s0 comes from vitakka(x), and then loop starts.
    # forward_step takes x_input and step_idx.

    input_x = torch.randn(1, samadhi_core_instance_no_gate.dim)

    # Perform a single forward step
    result = samadhi_core_instance_no_gate.forward_step(x_input=input_x, step_idx=0)

    assert result is not None
    next_state, full_log = result

    # Assertions for the outputs
    assert next_state.shape == input_x.shape
    assert isinstance(full_log, dict)
    assert "probe_log" in full_log
    assert "energies" in full_log
    assert "dynamics" in full_log

    # Check probe_log structure
    assert "winner_id" in full_log["probe_log"]
    assert "confidence" in full_log["probe_log"]


def test_samadhi_core_gating_mechanism(samadhi_core_instance_gated):
    input_x = torch.randn(1, samadhi_core_instance_gated.dim)
    result = samadhi_core_instance_gated.forward_step(x_input=input_x, step_idx=0)

    assert result is None
