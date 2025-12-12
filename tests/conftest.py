import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


# --- Mock Configuration for all tests ---
class GlobalMockConfig:
    def __init__(self):
        self.input_dim = 10
        self.dim = 10  # SatipatthanaSystem uses 'dim'
        self.hidden_dim = 20
        self.latent_dim = 5
        self.num_layers = 1
        self.dropout = 0.0
        self.activation = "relu"
        self.num_probes = 3
        self.n_probes = 3  # Added for components that expect 'n_probes'
        self.num_timesteps = 5
        self.gate_threshold = -1.0  # Default to always open for general tests
        self.attention_mode = "hard"
        self.probe_trainable = True
        self.vicara_type = "standard"  # Changed from "shared" to "standard" for consistency with current vicara.py
        self.learning_rate = 0.001
        self.stability_loss_weight = 1.0
        self.entropy_loss_weight = 0.1
        self.reconstruction_loss_weight = 1.0
        self.input_channels = 1
        self.input_height = 28
        self.input_width = 28
        self.sequence_length = 7
        self.seq_len = 7  # Added for models that expect 'seq_len'
        self.num_heads = 2
        self.refine_steps = 5  # Required for Vicara

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture(scope="session")
def global_config():
    return GlobalMockConfig()


# --- Data Loaders ---
@pytest.fixture
def mock_dataloader(global_config):
    dataloader = MagicMock()
    dataloader.__iter__.return_value = [
        (torch.randn(4, global_config.input_dim), torch.randn(4, global_config.input_dim), torch.randint(0, 2, (4,))),
        (torch.randn(4, global_config.input_dim), torch.randn(4, global_config.input_dim), torch.randint(0, 2, (4,))),
    ]
    dataloader.__len__.return_value = 2
    return dataloader


@pytest.fixture
def mock_image_dataloader(global_config):
    dataloader = MagicMock()
    dataloader.__iter__.return_value = [
        (
            torch.randn(4, global_config.input_channels, global_config.input_height, global_config.input_width),
            torch.randn(4, global_config.input_channels, global_config.input_height, global_config.input_width),
            torch.randint(0, 2, (4,)),
        ),
        (
            torch.randn(4, global_config.input_channels, global_config.input_height, global_config.input_width),
            torch.randn(4, global_config.input_channels, global_config.input_height, global_config.input_width),
            torch.randint(0, 2, (4,)),
        ),
    ]
    dataloader.__len__.return_value = 2
    return dataloader


@pytest.fixture
def mock_sequence_dataloader(global_config):
    dataloader = MagicMock()
    dataloader.__iter__.return_value = [
        (
            torch.randn(4, global_config.sequence_length, global_config.input_dim),
            torch.randn(4, global_config.sequence_length, global_config.input_dim),
            torch.randint(0, 2, (4,)),
        ),
        (
            torch.randn(4, global_config.sequence_length, global_config.input_dim),
            torch.randn(4, global_config.sequence_length, global_config.input_dim),
            torch.randint(0, 2, (4,)),
        ),
    ]
    dataloader.__len__.return_value = 2
    return dataloader
