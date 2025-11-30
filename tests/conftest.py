import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# from src.model.samadhi import SamadhiModel # Temporarily commented out
# from src.components.vitakka import Vitakka # Temporarily commented out
# from src.components.vicara import StandardVicara, VicaraBase # Temporarily commented out
from src.train.supervised_trainer import SupervisedSamadhiTrainer
from src.train.unsupervised_trainer import UnsupervisedSamadhiTrainer
from src.train.anomaly_trainer import AnomalySamadhiTrainer


# --- Mock Configuration for all tests ---
class GlobalMockConfig:
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
        self.gate_threshold = -1.0  # Default to always open for general tests
        self.attention_mode = "hard"
        self.probe_trainable = True
        self.vicara_type = "shared"
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


# --- Mock Models for Trainers ---
# class MockSamadhiModel(SamadhiModel): # Temporarily commented out
#     def __init__(self, config):
#         # We can pass the global_config directly if it has all the necessary attributes
#         # Or, we can adapt it slightly if the SamadhiCore expects specific config objects
#         super().__init__(config)  # Assuming SamadhiCore can take this config
#         self.output_mock = nn.Linear(config.input_dim, config.input_dim)

#     def forward(self, x_in):
#         output = self.output_mock(x_in)
#         final_state = output
#         meta_cognition_logs = {
#             "probe_log": torch.randn(self.num_timesteps, x_in.shape[0], self.num_probes),
#             "energies": torch.randn(self.num_timesteps, x_in.shape[0], self.num_probes),
#             "dynamics_log": torch.randn(self.num_timesteps, x_in.shape[0], self.vicara.dim),
#         }
#         return output, final_state, meta_cognition_logs


# @pytest.fixture # Temporarily commented out
# def mock_samadhi_model(global_config): # Temporarily commented out
#     return MockSamadhiModel(global_config) # Temporarily commented out


# --- Trainer Instances ---
@pytest.fixture
def supervised_trainer_instance(global_config):
    # Need a mock for SamadhiModel if not using real one
    mock_model = MagicMock()
    mock_model.parameters.return_value = []  # Or mock some parameters
    mock_model.config = global_config  # Attach config for trainer to read

    optimizer = torch.optim.Adam(mock_model.parameters(), lr=global_config.learning_rate)
    return SupervisedSamadhiTrainer(mock_model, optimizer, global_config)


@pytest.fixture
def unsupervised_trainer_instance(global_config):
    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    mock_model.config = global_config
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=global_config.learning_rate)
    return UnsupervisedSamadhiTrainer(mock_model, optimizer, global_config)


@pytest.fixture
def anomaly_trainer_instance(global_config):
    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    mock_model.config = global_config
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=global_config.learning_rate)
    return AnomalySamadhiTrainer(mock_model, optimizer, global_config)


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
