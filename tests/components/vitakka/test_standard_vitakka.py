import pytest
import torch
import torch.nn as nn
from samadhi.components.vitakka.standard import StandardVitakka
from samadhi.components.adapters.base import BaseAdapter


class MockAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config["input_dim"], config["dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def config():
    return {
        "dim": 32,
        "input_dim": 100,
        "n_probes": 5,
        "gate_threshold": 0.5,
        "probe_trainable": True,
        "mix_alpha": 0.5,
        "softmax_temp": 1.0,
        "training_attention_mode": "soft",
        "prediction_attention_mode": "hard",
    }


def test_standard_vitakka_initialization(config):
    adapter = MockAdapter(config)
    vitakka = StandardVitakka(config)  # No adapter argument needed
    assert isinstance(vitakka, StandardVitakka)
    assert vitakka.probes.shape == (config["n_probes"], config["dim"])


def test_standard_vitakka_forward_soft(config):
    adapter = MockAdapter(config)
    vitakka = StandardVitakka(config)  # No adapter argument needed
    vitakka.train()  # Soft mode

    batch_size = 4
    # x_input is now z_adapted (output of adapter)
    z_adapted = torch.randn(batch_size, config["dim"])

    s0, meta = vitakka(z_adapted)

    assert s0.shape == (batch_size, config["dim"])
    assert "winner_id" in meta
    assert "confidence" in meta
    assert "probs" in meta

    # Check probs shape
    assert meta["probs"].shape == (batch_size, config["n_probes"])


def test_standard_vitakka_forward_hard(config):
    adapter = MockAdapter(config)
    vitakka = StandardVitakka(config)  # No adapter argument needed
    vitakka.eval()  # Hard mode

    batch_size = 4
    # x_input is now z_adapted (output of adapter)
    z_adapted = torch.randn(batch_size, config["dim"])

    s0, meta = vitakka(z_adapted)

    assert s0.shape == (batch_size, config["dim"])

    # In hard mode, winner_id should be indices
    assert meta["winner_id"].shape == (batch_size,)

    # Check gate logic
    # With random init, raw scores might be low, gate might be closed (False)
    # Just check key existence
    assert "gate_open" in meta
