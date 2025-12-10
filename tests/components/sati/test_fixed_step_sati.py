"""
Tests for FixedStepSati implementation.
"""

import pytest
import torch

from samadhi.components.sati.fixed_step import FixedStepSati
from samadhi.configs.sati import FixedStepSatiConfig
from samadhi.core.santana import SantanaLog


class TestFixedStepSati:
    """Tests for FixedStepSati implementation."""

    def test_default_config(self):
        """Test initialization with default config."""
        sati = FixedStepSati()

        assert sati.config is not None
        assert isinstance(sati.config, FixedStepSatiConfig)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = FixedStepSatiConfig()
        sati = FixedStepSati(config)

        assert sati.config is config

    def test_never_stops_early(self):
        """Test that FixedStepSati never returns should_stop=True."""
        sati = FixedStepSati()

        # Create a santana with several steps
        santana = SantanaLog()
        state = torch.randn(4, 32)

        for i in range(10):
            santana.add(state, energy=0.1 / (i + 1))
            should_stop, info = sati(state, santana)
            assert should_stop is False

    def test_returns_correct_info(self):
        """Test that info dict contains expected fields."""
        sati = FixedStepSati()

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.05)

        should_stop, info = sati(state, santana)

        assert "reason" in info
        assert info["reason"] == "fixed_step_no_early_stop"
        assert "step_count" in info
        assert info["step_count"] == 1
        assert "energy" in info
        assert info["energy"] == 0.05

    def test_step_count_tracking(self):
        """Test that step count is correctly tracked."""
        sati = FixedStepSati()

        santana = SantanaLog()
        state = torch.randn(4, 32)

        for expected_count in range(1, 6):
            santana.add(state, energy=0.1)
            _, info = sati(state, santana)
            assert info["step_count"] == expected_count

    def test_handles_empty_santana(self):
        """Test behavior with empty SantanaLog."""
        sati = FixedStepSati()

        santana = SantanaLog()
        state = torch.randn(4, 32)

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["step_count"] == 0
        assert info["energy"] is None

    def test_handles_no_energy(self):
        """Test behavior when no energy is recorded."""
        sati = FixedStepSati()

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state)  # No energy

        should_stop, info = sati(state, santana)

        assert should_stop is False
        assert info["energy"] is None

    def test_returns_latest_energy(self):
        """Test that the final energy is returned."""
        sati = FixedStepSati()

        santana = SantanaLog()
        state = torch.randn(4, 32)

        santana.add(state, energy=0.5)
        santana.add(state, energy=0.3)
        santana.add(state, energy=0.1)

        _, info = sati(state, santana)
        assert info["energy"] == 0.1
