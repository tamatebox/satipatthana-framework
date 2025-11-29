import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from src.train.unsupervised_trainer import UnsupervisedSamadhiTrainer
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
        self.stability_loss_weight = 1.0
        self.entropy_loss_weight = 0.1
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
        super().__init__(config)


@pytest.fixture
def unsupervised_trainer_instance():
    config = MockConfig()
    model = MockSamadhiModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return UnsupervisedSamadhiTrainer(model, optimizer)


def test_unsupervised_trainer_init(unsupervised_trainer_instance):
    assert isinstance(unsupervised_trainer_instance.model, MockSamadhiModel)
    assert isinstance(unsupervised_trainer_instance.optimizer, torch.optim.Adam)


def test_unsupervised_trainer_train_step(unsupervised_trainer_instance):
    x_in = torch.randn(4, 10)  # Batch size 4, input_dim 10
    # Unsupervised trainer might receive y_target or labels but typically ignores them for loss calculation
    y_target = torch.randn(4, 10)
    labels = torch.randint(0, 2, (4,))

    # Train step returns float loss
    loss = unsupervised_trainer_instance.train_step(
        x_in, y_target
    )  # labels ignored in train_step args in unsupervised?
    # UnsupervisedSamadhiTrainer.train_step(x, y=None)

    assert isinstance(loss, float)
    assert loss > 0


def test_unsupervised_trainer_fit_method(unsupervised_trainer_instance):
    # Mock a DataLoader
    mock_dataloader = MagicMock()
    # Simulate two batches of data (x_in only, as y and labels are often ignored)
    mock_dataloader.__iter__.return_value = [
        (torch.randn(4, 10), torch.randn(4, 10), torch.randint(0, 2, (4,))),
        (torch.randn(4, 10), torch.randn(4, 10), torch.randint(0, 2, (4,))),
    ]
    mock_dataloader.__len__.return_value = 2

    # Mock optimizer
    unsupervised_trainer_instance.optimizer.step = MagicMock()
    unsupervised_trainer_instance.optimizer.zero_grad = MagicMock()

    # Run fit for a single epoch
    unsupervised_trainer_instance.fit(mock_dataloader, epochs=1)

    # Verify that train_step was called for each batch
    assert unsupervised_trainer_instance.optimizer.step.call_count == 2
    assert unsupervised_trainer_instance.optimizer.zero_grad.call_count == 2
