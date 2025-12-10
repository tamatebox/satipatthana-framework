"""
Tests for Vicara single-step interface (v4.0).
"""

import pytest
import torch
import torch.nn as nn

from samadhi.components.vicara.standard import StandardVicara
from samadhi.components.vicara.weighted import WeightedVicara
from samadhi.components.vicara.probe_specific import ProbeVicara
from samadhi.components.refiners.base import BaseRefiner


class MockRefiner(BaseRefiner):
    """Mock refiner for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config["dim"], config["dim"])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


@pytest.fixture
def config():
    return {
        "dim": 32,
        "refine_steps": 3,
        "inertia": 0.5,
    }


class TestVicaraStepInterface:
    """Tests for the new step() interface."""

    def test_step_returns_tensor(self, config):
        """Test that step() returns a tensor."""
        refiner = MockRefiner(config)
        vicara = StandardVicara(config, refiner)

        s_t = torch.randn(4, config["dim"])
        s_next = vicara.step(s_t)

        assert isinstance(s_next, torch.Tensor)

    def test_step_output_shape(self, config):
        """Test that step() output has correct shape."""
        refiner = MockRefiner(config)
        vicara = StandardVicara(config, refiner)

        batch_size = 8
        s_t = torch.randn(batch_size, config["dim"])
        s_next = vicara.step(s_t)

        assert s_next.shape == (batch_size, config["dim"])

    def test_step_changes_state(self, config):
        """Test that step() changes the state."""
        refiner = MockRefiner(config)
        vicara = StandardVicara(config, refiner)

        s_t = torch.randn(4, config["dim"])
        s_next = vicara.step(s_t)

        # State should change (unless inertia is exactly 1.0)
        assert not torch.allclose(s_t, s_next)

    def test_step_with_context(self, config):
        """Test step() with context parameter."""
        refiner = MockRefiner(config)
        vicara = StandardVicara(config, refiner)

        s_t = torch.randn(4, config["dim"])
        context = {"probs": torch.softmax(torch.randn(4, 3), dim=1)}

        # Should not raise even though StandardVicara ignores context
        s_next = vicara.step(s_t, context)
        assert s_next.shape == s_t.shape

    def test_step_gradient_flow(self, config):
        """Test that gradients flow through step()."""
        refiner = MockRefiner(config)
        vicara = StandardVicara(config, refiner)

        s_t = torch.randn(4, config["dim"], requires_grad=True)
        s_next = vicara.step(s_t)
        loss = s_next.sum()
        loss.backward()

        assert s_t.grad is not None

    def test_step_multiple_calls(self, config):
        """Test multiple step() calls (simulating loop)."""
        refiner = MockRefiner(config)
        vicara = StandardVicara(config, refiner)

        s_t = torch.randn(4, config["dim"])
        states = [s_t.clone()]

        for _ in range(5):
            s_t = vicara.step(s_t)
            states.append(s_t.clone())

        # Should have 6 states (initial + 5 steps)
        assert len(states) == 6

        # States should be different
        for i in range(len(states) - 1):
            assert not torch.allclose(states[i], states[i + 1])

    def test_step_inertia_effect(self, config):
        """Test that inertia affects state update."""
        refiner = MockRefiner(config)

        # High inertia - state changes slowly
        config_high = config.copy()
        config_high["inertia"] = 0.9
        vicara_high = StandardVicara(config_high, refiner)

        # Low inertia - state changes quickly
        config_low = config.copy()
        config_low["inertia"] = 0.1
        vicara_low = StandardVicara(config_low, refiner)

        s_t = torch.randn(4, config["dim"])
        torch.manual_seed(42)
        s_high = vicara_high.step(s_t.clone())
        torch.manual_seed(42)
        s_low = vicara_low.step(s_t.clone())

        # High inertia should keep state closer to original
        dist_high = torch.norm(s_high - s_t).item()
        dist_low = torch.norm(s_low - s_t).item()
        assert dist_high < dist_low


class TestWeightedVicaraStep:
    """Tests for WeightedVicara step() interface."""

    def test_step_output_shape(self, config):
        """Test step() output shape for WeightedVicara."""
        refiner = MockRefiner(config)
        vicara = WeightedVicara(config, refiner)

        s_t = torch.randn(4, config["dim"])
        s_next = vicara.step(s_t)

        assert s_next.shape == s_t.shape


class TestProbeVicaraStep:
    """Tests for ProbeVicara step() interface."""

    @pytest.fixture
    def probe_config(self, config):
        cfg = config.copy()
        cfg["type"] = "probe_specific"  # Important: set correct type
        cfg["n_probes"] = 3
        cfg["training_attention_mode"] = "soft"
        cfg["prediction_attention_mode"] = "hard"
        return cfg

    def test_step_soft_mode(self, probe_config):
        """Test step() in soft attention mode."""
        refiners = nn.ModuleList(
            [MockRefiner(probe_config) for _ in range(probe_config["n_probes"])]
        )
        vicara = ProbeVicara(probe_config, refiners)
        vicara.train()

        s_t = torch.randn(4, probe_config["dim"])
        context = {
            "probs": torch.softmax(torch.randn(4, probe_config["n_probes"]), dim=1)
        }

        s_next = vicara.step(s_t, context)
        assert s_next.shape == s_t.shape

    def test_step_hard_mode(self, probe_config):
        """Test step() in hard attention mode."""
        refiners = nn.ModuleList(
            [MockRefiner(probe_config) for _ in range(probe_config["n_probes"])]
        )
        vicara = ProbeVicara(probe_config, refiners)
        vicara.eval()

        s_t = torch.randn(4, probe_config["dim"])
        context = {"winner_id": torch.randint(0, probe_config["n_probes"], (4,))}

        s_next = vicara.step(s_t, context)
        assert s_next.shape == s_t.shape


class TestStepForwardEquivalence:
    """Test that step() and forward() produce equivalent results."""

    def test_manual_loop_matches_forward(self, config):
        """Test that calling step() in a loop matches forward()."""
        refiner = MockRefiner(config)
        vicara = StandardVicara(config, refiner)
        vicara.train()  # No early stopping in training mode

        torch.manual_seed(42)
        s0 = torch.randn(4, config["dim"])

        # Run forward()
        torch.manual_seed(42)
        s_final_forward, _, _ = vicara(s0.clone())

        # Manual loop using step()
        torch.manual_seed(42)
        s_t = s0.clone()
        for _ in range(config["refine_steps"]):
            s_t = vicara.step(s_t)

        # Results should match
        assert torch.allclose(s_final_forward, s_t)
