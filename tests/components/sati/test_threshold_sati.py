"""
Tests for ThresholdSati implementation.
"""

import pytest
import torch

from samadhi.components.sati.threshold import ThresholdSati
from samadhi.configs.sati import ThresholdSatiConfig
from samadhi.core.santana import SantanaLog


class TestThresholdSati:
    """Tests for ThresholdSati implementation."""

    def test_default_config(self):
        """Test initialization with default config."""
        sati = ThresholdSati()

        assert sati.config is not None
        assert isinstance(sati.config, ThresholdSatiConfig)
        assert sati.config.energy_threshold == 1e-4
        assert sati.config.min_steps == 1

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = ThresholdSatiConfig(energy_threshold=0.01, min_steps=3)
        sati = ThresholdSati(config)

        assert sati.config.energy_threshold == 0.01
        assert sati.config.min_steps == 3

    def test_stops_when_below_threshold(self):
        """Test that sati stops when energy is below threshold."""
        config = ThresholdSatiConfig(energy_threshold=0.1, min_steps=1)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.05)  # Below threshold

        should_stop, info = sati(state, santana)

        assert should_stop is True
        assert info["reason"] == "threshold_reached"
        assert info["energy"] == 0.05
        assert info["threshold"] == 0.1

    def test_continues_when_above_threshold(self):
        """Test that sati continues when energy is above threshold."""
        config = ThresholdSatiConfig(energy_threshold=0.01, min_steps=1)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.1)  # Above threshold

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["reason"] == "above_threshold"

    def test_respects_min_steps(self):
        """Test that sati respects min_steps even when energy is low."""
        config = ThresholdSatiConfig(energy_threshold=0.1, min_steps=3)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)

        # Step 1: energy low but min_steps not reached
        santana.add(state, energy=0.01)
        should_stop, info = sati(state, santana)
        assert should_stop is False
        assert info["reason"] == "min_steps_not_reached"

        # Step 2: still not reached
        santana.add(state, energy=0.01)
        should_stop, info = sati(state, santana)
        assert should_stop is False
        assert info["reason"] == "min_steps_not_reached"

        # Step 3: now can stop
        santana.add(state, energy=0.01)
        should_stop, info = sati(state, santana)
        assert should_stop is True
        assert info["reason"] == "threshold_reached"

    def test_handles_no_energy(self):
        """Test behavior when no energy is recorded."""
        config = ThresholdSatiConfig(energy_threshold=0.1, min_steps=1)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state)  # No energy

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["reason"] == "no_energy_recorded"

    def test_handles_empty_santana(self):
        """Test behavior with empty SantanaLog."""
        config = ThresholdSatiConfig(min_steps=1)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["reason"] == "min_steps_not_reached"
        assert info["step_count"] == 0

    def test_info_contains_all_fields(self):
        """Test that info dict contains all expected fields."""
        config = ThresholdSatiConfig(energy_threshold=0.1, min_steps=2)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.05)
        santana.add(state, energy=0.05)

        should_stop, info = sati(state, santana)

        assert "reason" in info
        assert "step_count" in info
        assert "min_steps" in info
        assert "energy" in info
        assert "threshold" in info

        assert info["step_count"] == 2
        assert info["min_steps"] == 2
        assert info["energy"] == 0.05
        assert info["threshold"] == 0.1

    def test_exact_threshold_boundary(self):
        """Test behavior at exact threshold boundary."""
        config = ThresholdSatiConfig(energy_threshold=0.1, min_steps=1)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)

        # Exactly at threshold - should NOT stop (not below)
        santana.add(state, energy=0.1)
        should_stop, info = sati(state, santana)
        assert should_stop is False
        assert info["reason"] == "above_threshold"

        # Just below threshold - should stop
        santana.clear()
        santana.add(state, energy=0.09999)
        should_stop, info = sati(state, santana)
        assert should_stop is True

    def test_gradual_convergence(self):
        """Test typical convergence scenario with decreasing energy."""
        config = ThresholdSatiConfig(energy_threshold=0.05, min_steps=1)
        sati = ThresholdSati(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)

        # Simulate gradual convergence
        energies = [0.5, 0.3, 0.15, 0.08, 0.04]
        stopped_at = None

        for i, energy in enumerate(energies):
            santana.add(state, energy=energy)
            should_stop, info = sati(state, santana)
            if should_stop:
                stopped_at = i
                break

        # Should stop at energy=0.04 (index 4)
        assert stopped_at == 4
        assert info["energy"] == 0.04
