"""
Tests for BaseSati interface.
"""

import pytest
import torch
from typing import Tuple, Dict, Any

from satipatthana.components.sati.base import BaseSati
from satipatthana.core.santana import SantanaLog
from satipatthana.configs.sati import FixedStepSatiConfig, ThresholdSatiConfig


class MockFixedStepSati(BaseSati):
    """Mock implementation that never stops early."""

    def forward(self, current_state: torch.Tensor, santana: SantanaLog) -> Tuple[bool, Dict[str, Any]]:
        """Never stop - let the loop run to max_steps."""
        return False, {"reason": "fixed_step"}


class MockThresholdSati(BaseSati):
    """Mock implementation that stops when energy is low."""

    def forward(self, current_state: torch.Tensor, santana: SantanaLog) -> Tuple[bool, Dict[str, Any]]:
        """Stop when final energy is below threshold."""
        if not santana.energies:
            return False, {"reason": "no_energy_data"}

        final_energy = santana.energies[-1]
        should_stop = final_energy < self.config.energy_threshold

        return should_stop, {
            "reason": "threshold_met" if should_stop else "above_threshold",
            "final_energy": final_energy,
        }


class TestBaseSati:
    """Tests for BaseSati interface."""

    def test_interface_inheritance(self):
        """Test that BaseSati inherits from nn.Module."""
        config = FixedStepSatiConfig()
        sati = MockFixedStepSati(config)

        assert isinstance(sati, torch.nn.Module)

    def test_forward_signature(self):
        """Test forward method returns correct types."""
        config = FixedStepSatiConfig()
        sati = MockFixedStepSati(config)

        state = torch.randn(4, 32)
        santana = SantanaLog()
        santana.add(state)

        should_stop, info = sati(state, santana)

        assert isinstance(should_stop, bool)
        assert isinstance(info, dict)

    def test_fixed_step_never_stops(self):
        """Test that FixedStepSati never triggers early stop."""
        config = FixedStepSatiConfig()
        sati = MockFixedStepSati(config)

        state = torch.randn(4, 32)
        santana = SantanaLog()

        # Add multiple states
        for _ in range(10):
            santana.add(state, energy=0.001)

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["reason"] == "fixed_step"

    def test_threshold_stops_when_converged(self):
        """Test that ThresholdSati stops when energy is low."""
        config = ThresholdSatiConfig(energy_threshold=0.01)
        sati = MockThresholdSati(config)

        state = torch.randn(4, 32)
        santana = SantanaLog()

        # Add states with decreasing energy
        santana.add(state, energy=0.5)
        santana.add(state, energy=0.1)
        santana.add(state, energy=0.001)  # Below threshold

        should_stop, info = sati(state, santana)

        assert should_stop is True
        assert info["reason"] == "threshold_met"
        assert info["final_energy"] == 0.001

    def test_threshold_continues_when_not_converged(self):
        """Test that ThresholdSati continues when energy is high."""
        config = ThresholdSatiConfig(energy_threshold=0.01)
        sati = MockThresholdSati(config)

        state = torch.randn(4, 32)
        santana = SantanaLog()

        # Add states with energy above threshold
        santana.add(state, energy=0.5)
        santana.add(state, energy=0.1)

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["reason"] == "above_threshold"

    def test_config_stored(self):
        """Test that config is properly stored."""
        config = ThresholdSatiConfig(energy_threshold=0.05)
        sati = MockThresholdSati(config)

        assert sati.config is config
        assert sati.config.energy_threshold == 0.05

    def test_empty_santana_log(self):
        """Test handling of empty SantanaLog."""
        config = ThresholdSatiConfig()
        sati = MockThresholdSati(config)

        state = torch.randn(4, 32)
        santana = SantanaLog()  # Empty log

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["reason"] == "no_energy_data"
