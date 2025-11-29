import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from src.train.anomaly_trainer import AnomalySamadhiTrainer
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
        self.gate_threshold = 0.5  # A realistic gate threshold for anomaly detection
        self.attention_mode = "hard"  # Anomaly detection might use hard attention for clear probe selection
        self.probe_trainable = True
        self.learning_rate = 0.001
        self.stability_loss_weight = 1.0
        self.entropy_loss_weight = 0.1
        self.reconstruction_loss_weight = 1.0
        self.vicara_type = "shared"
        self.refine_steps = 3
        self.n_probes = 3
        self.anomaly_margin = 5.0
        self.anomaly_weight = 1.0

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


class MockSamadhiModel(SamadhiModel):
    def __init__(self, config):
        super().__init__(config)


@pytest.fixture
def anomaly_trainer_instance():
    config = MockConfig()
    model = MockSamadhiModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return AnomalySamadhiTrainer(model, optimizer, **config.__dict__)  # Pass config as kwargs


def test_anomaly_trainer_init(anomaly_trainer_instance):
    assert isinstance(anomaly_trainer_instance.model, MockSamadhiModel)
    assert isinstance(anomaly_trainer_instance.optimizer, torch.optim.Adam)
    assert anomaly_trainer_instance.margin == 5.0
    assert anomaly_trainer_instance.anomaly_weight == 1.0


def test_anomaly_trainer_train_step(anomaly_trainer_instance):
    x_in = torch.randn(4, 10)
    # y is label (0 for normal, 1 for anomaly)
    y_target = torch.randint(0, 2, (4,))

    # Train step returns loss float
    loss = anomaly_trainer_instance.train_step(x_in, y_target)

    assert isinstance(loss, float)
    assert loss > 0


def test_anomaly_trainer_predict(anomaly_trainer_instance):
    # Predict takes a dataloader
    mock_dataloader = MagicMock()
    mock_dataloader.__iter__.return_value = [torch.randn(2, 10)]
    mock_dataloader.__len__.return_value = 1

    predictions, logs = anomaly_trainer_instance.predict(mock_dataloader)

    assert isinstance(predictions, list)
    assert isinstance(logs, list)


def test_anomaly_trainer_fit_method(anomaly_trainer_instance):
    mock_dataloader = MagicMock()
    # (x, y)
    mock_dataloader.__iter__.return_value = [
        (torch.randn(4, 10), torch.randint(0, 2, (4,))),
        (torch.randn(4, 10), torch.randint(0, 2, (4,))),
    ]
    mock_dataloader.__len__.return_value = 2

    anomaly_trainer_instance.optimizer.step = MagicMock()
    anomaly_trainer_instance.optimizer.zero_grad = MagicMock()

    anomaly_trainer_instance.fit(mock_dataloader, epochs=1)

    assert anomaly_trainer_instance.optimizer.step.call_count == 2
    assert anomaly_trainer_instance.optimizer.zero_grad.call_count == 2
