import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.core.engine import SamadhiEngine
from src.components.adapters.base import BaseAdapter
from src.components.decoders.base import BaseDecoder
from src.components.vitakka import Vitakka  # Using concrete Vitakka for now
from src.components.vicara import VicaraBase, StandardVicara  # Using concrete Vicara for now


# Mock implementations for abstract classes
class MockAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config["input_dim"], config["dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockDecoder(BaseDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config["dim"], config["output_dim"])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


@pytest.fixture
def basic_config():
    return {
        "dim": 32,
        "input_dim": 100,
        "output_dim": 10,
        "n_probes": 5,
        "gate_threshold": -1.0,
        "refine_steps": 3,
        "vicara_type": "standard",
    }


@pytest.fixture
def mock_components(basic_config):
    adapter = MockAdapter(basic_config)
    vitakka = Vitakka(basic_config)
    vicara = StandardVicara(basic_config)
    decoder = MockDecoder(basic_config)
    return adapter, vitakka, vicara, decoder


def test_samadhi_engine_initialization(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)
    assert engine.adapter == adapter
    assert engine.vitakka == vitakka
    assert engine.vicara == vicara
    assert engine.decoder == decoder
    assert engine.config == basic_config
    assert engine.history_log == []


def test_samadhi_engine_forward_pass(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    batch_size = 2
    input_tensor = torch.randn(batch_size, basic_config["input_dim"])

    output, s_final, meta = engine.forward(input_tensor)

    assert output.shape == (batch_size, basic_config["output_dim"])
    assert s_final.shape == (batch_size, basic_config["dim"])
    assert "winner_id" in meta
    assert "gate_open" in meta


def test_samadhi_engine_forward_step_open_gate(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    # Ensure gate is always open for this test
    basic_config["gate_threshold"] = -100.0  # Make sure gate is open
    vitakka = Vitakka(basic_config)  # Re-init Vitakka with new gate_threshold

    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    input_tensor = torch.randn(1, basic_config["input_dim"])
    s_final, full_log = engine.forward_step(input_tensor, 0)

    assert s_final.shape == (1, basic_config["dim"])
    assert full_log["step"] == 0
    assert full_log["probe_log"]["gate_open"] is True
    assert len(engine.history_log) == 1


def test_samadhi_engine_forward_step_closed_gate(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    # Ensure gate is always closed for this test
    basic_config["gate_threshold"] = 100.0  # Make sure gate is closed
    vitakka = Vitakka(basic_config)  # Re-init Vitakka with new gate_threshold

    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    input_tensor = torch.randn(1, basic_config["input_dim"])
    result = engine.forward_step(input_tensor, 0)

    assert result is None
    assert len(engine.history_log) == 0


def test_samadhi_engine_compute_dynamics(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    # Simulate history log
    engine.history_log = [
        {"step": 0, "probe_log": {"winner_id": 0, "winner_label": "A", "confidence": 0.8}},
    ]
    current_log = {"winner_id": 0, "winner_label": "A", "confidence": 0.85}
    dynamics = engine._compute_dynamics(current_log)
    assert dynamics["type"] == "Sustain"
    assert dynamics["confidence_delta"] == pytest.approx(0.05)

    current_log_shift = {"winner_id": 1, "winner_label": "B", "confidence": 0.7}
    dynamics_shift = engine._compute_dynamics(current_log_shift)
    assert dynamics_shift["type"] == "Shift"
    assert dynamics_shift["confidence_delta"] == pytest.approx(-0.1)
