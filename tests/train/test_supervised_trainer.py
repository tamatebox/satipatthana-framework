import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from src.train.supervised_trainer import SupervisedSamadhiTrainer
from src.model.samadhi import SamadhiModel
from src.components.vitakka import Vitakka
from src.components.vicara import StandardVicara


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
        self.num_timesteps = 5
        self.gate_threshold = -1.0  # Always open for trainer tests
        self.attention_mode = "soft"  # Use soft for training to allow gradient flow
        self.probe_trainable = True
        self.learning_rate = 0.001
        self.vicara_type = "shared"
        self.refine_steps = 3
        self.n_probes = 3

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


class MockSamadhiModel(SamadhiModel):
    def __init__(self, config):
        # Initialize parent to get vitakka and vicara
        super().__init__(config)


@pytest.fixture
def supervised_trainer_instance():
    config = MockConfig()
    model = MockSamadhiModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return SupervisedSamadhiTrainer(model, optimizer)


def test_supervised_trainer_init(supervised_trainer_instance):
    assert isinstance(supervised_trainer_instance.model, MockSamadhiModel)
    assert isinstance(supervised_trainer_instance.optimizer, torch.optim.Adam)


def test_supervised_trainer_train_step(supervised_trainer_instance):
    x_in = torch.randn(4, 10)  # Batch size 4, input_dim 10
    y_target = torch.randn(4, 10)  # Supervised target

    # Train step returns loss float
    loss = supervised_trainer_instance.train_step(x_in, y_target)

    assert isinstance(loss, float)
    assert loss > 0  # Expect some loss


def test_supervised_trainer_predict(supervised_trainer_instance):
    # Predict takes a dataloader
    mock_dataloader = MagicMock()
    # Batch of size 2
    mock_dataloader.__iter__.return_value = [torch.randn(2, 10)]
    mock_dataloader.__len__.return_value = 1

    predictions, logs = supervised_trainer_instance.predict(mock_dataloader)

    assert isinstance(predictions, list)
    assert isinstance(logs, list)
    assert len(predictions) == 2  # 2 samples in batch
    assert len(logs) == 2


def test_supervised_trainer_fit_method(supervised_trainer_instance):
    # Mock a DataLoader
    mock_dataloader = MagicMock()
    # Simulate two batches of data
    # fit expects (x, y)
    mock_dataloader.__iter__.return_value = [
        (torch.randn(4, 10), torch.randn(4, 10)),
        (torch.randn(4, 10), torch.randn(4, 10)),
    ]
    # enumerate uses iter

    # Mock optimizer step to verify calls
    supervised_trainer_instance.optimizer.step = MagicMock()
    supervised_trainer_instance.optimizer.zero_grad = MagicMock()

    # Run fit for a single epoch
    supervised_trainer_instance.fit(mock_dataloader, epochs=1)

    # Verify that optimizer step was called for each batch
    assert supervised_trainer_instance.optimizer.step.call_count == 2
    assert supervised_trainer_instance.optimizer.zero_grad.call_count == 2
