import torch
import pytest
from src.components.vitakka import Vitakka


@pytest.fixture
def base_config():
    return {
        "dim": 16,
        "n_probes": 5,
        "probe_trainable": True,
        "gate_threshold": 0.5,
        "mix_alpha": 0.5,
        "softmax_temp": 1.0,
        "training_attention_mode": "soft",
        "prediction_attention_mode": "hard",
    }


def test_vitakka_initialization(base_config):
    vitakka = Vitakka(base_config)
    assert vitakka.probes.shape == (5, 16)
    # Check normalization
    norms = torch.norm(vitakka.probes, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_vitakka_forward_hard(base_config):
    vitakka = Vitakka(base_config)
    vitakka.eval()  # Ensure prediction mode

    batch_size = 2
    x = torch.randn(batch_size, 16)

    s0, meta = vitakka(x)

    assert s0.shape == (batch_size, 16)
    assert "winner_id" in meta
    assert meta["gate_open"].shape[0] == batch_size

    # Check if hard selection logic holds
    assert meta["winner_id"].shape == (batch_size,)


def test_vitakka_forward_soft(base_config):
    vitakka = Vitakka(base_config)
    vitakka.train()  # Ensure training mode

    batch_size = 2
    x = torch.randn(batch_size, 16)

    s0, meta = vitakka(x)

    assert s0.shape == (batch_size, 16)
    # In soft mode, probs should be used
    assert "probs" in meta
    assert meta["probs"].shape == (batch_size, 5)


def test_load_probes(base_config):
    vitakka = Vitakka(base_config)
    new_probes = torch.randn(5, 16)
    vitakka.load_probes(new_probes)

    # Check if loaded and normalized
    assert torch.allclose(torch.norm(vitakka.probes, dim=1), torch.ones(5))
