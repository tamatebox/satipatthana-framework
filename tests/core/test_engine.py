import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from samadhi.core.engine import SamadhiEngine
from samadhi.components.adapters.base import BaseAdapter
from samadhi.components.decoders.base import BaseDecoder
from samadhi.components.vitakka.base import BaseVitakka  # Using abstract Vitakka for now
from samadhi.components.vicara.base import BaseVicara  # Using abstract Vicara for now
from samadhi.components.vitakka.standard import StandardVitakka
from samadhi.components.vicara.standard import StandardVicara
from samadhi.components.refiners.base import BaseRefiner  # Import BaseRefiner for MockRefiner
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.factory import (
    create_adapter_config,
    create_decoder_config,
    create_vitakka_config,
    create_vicara_config,
)


# Mock implementations for abstract classes
class MockAdapter(BaseAdapter):

    def __init__(self, config):
        # Allow legacy dict config for flexibility in tests
        if isinstance(config, dict):
            config = create_adapter_config(config)
        elif isinstance(config, SamadhiConfig):
            config = config.adapter  # Extract relevant config

        super().__init__(config)
        self.linear = nn.Linear(config.input_dim, config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockDecoder(BaseDecoder):
    def __init__(self, config):
        if isinstance(config, dict):
            config = create_decoder_config(config)
        elif isinstance(config, SamadhiConfig):
            config = config.decoder

        super().__init__(config)
        self.linear = nn.Linear(config.dim, config.input_dim)  # output_dim is input_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


# Define a simple MockRefiner for testing purposes
class MockRefiner(BaseRefiner):
    def __init__(self, config):
        if isinstance(config, SamadhiConfig):
            # Refiners take BaseConfig or dict
            config = config.vicara

        super().__init__(config)
        self.linear = nn.Linear(self.dim, self.dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Simple identity-like transformation for mock
        return self.linear(s)


@pytest.fixture
def basic_config():
    config_dict = {
        "dim": 32,
        "input_dim": 100,
        "n_probes": 5,
        "gate_threshold": -1.0,
        "vicara": {
            "refine_steps": 3,
            "type": "standard",
            "dim": 32,  # Explicitly set dim for component config creation if needed, though propagation should handle it
        },
    }
    return SamadhiConfig.from_dict(config_dict)


@pytest.fixture
def mock_components(basic_config):
    adapter = MockAdapter(basic_config)
    vitakka = StandardVitakka(basic_config.vitakka)  # No adapter argument needed
    # Use MockRefiner instead of MagicMock
    refiner = MockRefiner(basic_config)
    vicara = StandardVicara(basic_config.vicara, refiner)
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
    input_tensor = torch.randn(batch_size, basic_config.adapter.input_dim)

    output, s_final, meta = engine.forward(input_tensor)

    assert output.shape == (batch_size, basic_config.decoder.input_dim)  # output_dim = input_dim for reconstruction
    assert s_final.shape == (batch_size, basic_config.dim)
    assert "winner_id" in meta
    assert "gate_open" in meta


def test_samadhi_engine_forward_step_open_gate(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components

    # Ensure gate is always open for this test
    # We need to modify the config object directly since it's a dataclass
    basic_config.vitakka.gate_threshold = -100.0

    vitakka = StandardVitakka(basic_config.vitakka)  # Re-init Vitakka

    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    input_tensor = torch.randn(1, basic_config.adapter.input_dim)
    s_final, full_log = engine.forward_step(input_tensor, 0)

    assert s_final.shape == (1, basic_config.dim)
    assert full_log["step"] == 0
    assert full_log["probe_log"]["gate_open"] is True
    assert len(engine.history_log) == 1


def test_samadhi_engine_forward_step_closed_gate(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components

    # Ensure gate is always closed for this test
    basic_config.vitakka.gate_threshold = 100.0
    vitakka = StandardVitakka(basic_config.vitakka)

    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    input_tensor = torch.randn(1, basic_config.adapter.input_dim)
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


def test_samadhi_engine_forward_skip_vitakka(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    batch_size = 2
    input_tensor = torch.randn(batch_size, basic_config.adapter.input_dim)

    # Skip Vitakka (run_vitakka=False)
    # Vicara runs on adapter output directly
    output, s_final, meta = engine.forward(input_tensor, run_vitakka=False, run_vicara=True)

    assert output.shape == (batch_size, basic_config.decoder.input_dim)
    assert s_final.shape == (batch_size, basic_config.dim)
    assert meta == {}  # Expect empty meta since Vitakka was skipped


def test_samadhi_engine_forward_skip_vicara(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    batch_size = 2
    input_tensor = torch.randn(batch_size, basic_config.adapter.input_dim)

    # Skip Vicara (run_vicara=False)
    output, s_final, meta = engine.forward(input_tensor, run_vitakka=True, run_vicara=False)

    assert output.shape == (batch_size, basic_config.decoder.input_dim)
    assert s_final.shape == (batch_size, basic_config.dim)
    assert "winner_id" in meta


def test_samadhi_engine_forward_skip_both(basic_config, mock_components):
    adapter, vitakka, vicara, decoder = mock_components
    engine = SamadhiEngine(adapter, vitakka, vicara, decoder, basic_config)

    batch_size = 2
    input_tensor = torch.randn(batch_size, basic_config.adapter.input_dim)

    # Skip Both (Autoencoder mode)
    output, s_final, meta = engine.forward(input_tensor, run_vitakka=False, run_vicara=False)

    assert output.shape == (batch_size, basic_config.decoder.input_dim)
    assert s_final.shape == (batch_size, basic_config.dim)
    assert meta == {}
