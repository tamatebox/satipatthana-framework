"""
Tests for StandardVipassana implementation.
"""

import pytest
import torch

from samadhi.components.vipassana.standard import StandardVipassana
from samadhi.configs.vipassana import StandardVipassanaConfig
from samadhi.core.santana import SantanaLog


class TestStandardVipassana:
    """Tests for StandardVipassana implementation."""

    def test_default_config(self):
        """Test initialization with default config."""
        vipassana = StandardVipassana()

        assert vipassana.config is not None
        assert isinstance(vipassana.config, StandardVipassanaConfig)
        assert vipassana.config.context_dim == 32
        assert vipassana.config.hidden_dim == 64

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = StandardVipassanaConfig(context_dim=64, hidden_dim=128)
        vipassana = StandardVipassana(config)

        assert vipassana.config.context_dim == 64
        assert vipassana.config.hidden_dim == 128

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        config = StandardVipassanaConfig(context_dim=32)
        vipassana = StandardVipassana(config)

        batch_size = 4
        state_dim = 64

        santana = SantanaLog()
        state = torch.randn(batch_size, state_dim)
        santana.add(state, energy=0.1)
        santana.add(state, energy=0.05)

        s_star = torch.randn(batch_size, state_dim)
        v_ctx, trust_score = vipassana(s_star, santana)

        assert v_ctx.shape == (batch_size, 32)
        assert isinstance(trust_score, float)

    def test_trust_score_in_valid_range(self):
        """Test that trust score is in [0.0, 1.0] range."""
        vipassana = StandardVipassana()

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.1)

        s_star = torch.randn(4, 32)
        _, trust_score = vipassana(s_star, santana)

        assert 0.0 <= trust_score <= 1.0

    def test_handles_empty_santana(self):
        """Test behavior with empty SantanaLog."""
        vipassana = StandardVipassana()

        santana = SantanaLog()  # Empty
        s_star = torch.randn(4, 32)

        v_ctx, trust_score = vipassana(s_star, santana)

        assert v_ctx.shape == (4, 32)
        assert 0.0 <= trust_score <= 1.0

    def test_handles_single_step(self):
        """Test with single step in trajectory."""
        vipassana = StandardVipassana()

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.1)

        s_star = torch.randn(4, 32)
        v_ctx, trust_score = vipassana(s_star, santana)

        assert v_ctx.shape == (4, 32)
        assert 0.0 <= trust_score <= 1.0

    def test_handles_long_trajectory(self):
        """Test with long trajectory."""
        vipassana = StandardVipassana()

        santana = SantanaLog()
        state = torch.randn(4, 32)
        for i in range(20):
            santana.add(state, energy=0.1 / (i + 1))

        s_star = torch.randn(4, 32)
        v_ctx, trust_score = vipassana(s_star, santana)

        assert v_ctx.shape == (4, 32)
        assert 0.0 <= trust_score <= 1.0

    def test_context_dim_customizable(self):
        """Test that context_dim config is respected."""
        for context_dim in [16, 32, 64, 128]:
            config = StandardVipassanaConfig(context_dim=context_dim)
            vipassana = StandardVipassana(config)

            santana = SantanaLog()
            state = torch.randn(4, 32)
            santana.add(state, energy=0.1)

            s_star = torch.randn(4, 32)
            v_ctx, _ = vipassana(s_star, santana)

            assert v_ctx.shape == (4, context_dim)

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        vipassana = StandardVipassana()

        for batch_size in [1, 4, 16, 32]:
            santana = SantanaLog()
            state = torch.randn(batch_size, 32)
            santana.add(state, energy=0.1)

            s_star = torch.randn(batch_size, 32)
            v_ctx, trust_score = vipassana(s_star, santana)

            assert v_ctx.shape == (batch_size, 32)
            assert 0.0 <= trust_score <= 1.0

    def test_different_state_dims(self):
        """Test with various state dimensions."""
        vipassana = StandardVipassana()

        for state_dim in [16, 32, 64, 128]:
            santana = SantanaLog()
            state = torch.randn(4, state_dim)
            santana.add(state, energy=0.1)

            s_star = torch.randn(4, state_dim)
            v_ctx, trust_score = vipassana(s_star, santana)

            assert v_ctx.shape == (4, 32)
            assert 0.0 <= trust_score <= 1.0

    def test_velocity_computed_from_trajectory(self):
        """Test that velocity is computed from trajectory movement."""
        vipassana = StandardVipassana()

        # Case 1: No movement (initial = final)
        santana1 = SantanaLog()
        state = torch.zeros(4, 32)
        santana1.add(state, energy=0.1)

        s_star_same = torch.zeros(4, 32)
        v_ctx1, _ = vipassana(s_star_same, santana1)

        # Case 2: Large movement
        santana2 = SantanaLog()
        santana2.add(torch.zeros(4, 32), energy=0.1)

        s_star_moved = torch.ones(4, 32) * 10
        v_ctx2, _ = vipassana(s_star_moved, santana2)

        # Context vectors should be different
        assert not torch.allclose(v_ctx1, v_ctx2)

    def test_energy_affects_context(self):
        """Test that trajectory energy affects the context vector."""
        vipassana = StandardVipassana()

        state = torch.randn(4, 32)
        s_star = torch.randn(4, 32)

        # High energy trajectory
        santana_high = SantanaLog()
        santana_high.add(state, energy=10.0)
        v_ctx_high, _ = vipassana(s_star, santana_high)

        # Low energy trajectory
        santana_low = SantanaLog()
        santana_low.add(state, energy=0.001)
        v_ctx_low, _ = vipassana(s_star, santana_low)

        # Context vectors should be different due to different energy
        assert not torch.allclose(v_ctx_high, v_ctx_low)

    def test_is_nn_module(self):
        """Test that StandardVipassana is an nn.Module."""
        vipassana = StandardVipassana()

        assert isinstance(vipassana, torch.nn.Module)

    def test_trainable_parameters(self):
        """Test that vipassana has trainable parameters."""
        vipassana = StandardVipassana()

        # First forward pass to initialize networks
        santana = SantanaLog()
        santana.add(torch.randn(4, 32), energy=0.1)
        vipassana(torch.randn(4, 32), santana)

        # Check that parameters exist
        params = list(vipassana.parameters())
        assert len(params) > 0

        # Check that some parameters require grad
        trainable = sum(p.numel() for p in params if p.requires_grad)
        assert trainable > 0

    def test_device_consistency(self):
        """Test that output device matches input device."""
        vipassana = StandardVipassana()

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.1)

        s_star = torch.randn(4, 32)
        v_ctx, _ = vipassana(s_star, santana)

        assert v_ctx.device == s_star.device
