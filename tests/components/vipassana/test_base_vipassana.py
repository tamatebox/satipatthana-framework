"""
Tests for BaseVipassana interface.
"""

import pytest
import torch
from typing import Tuple

from satipatthana.components.vipassana.base import BaseVipassana
from satipatthana.core.santana import SantanaLog
from satipatthana.configs.vipassana import StandardVipassanaConfig, LSTMVipassanaConfig


class MockVipassana(BaseVipassana):
    """Mock implementation of BaseVipassana for testing."""

    def __init__(self, config):
        super().__init__(config)
        # Simple encoder: average pooling over trajectory + MLP
        self.encoder = torch.nn.Linear(config.context_dim, config.context_dim)

    def forward(self, s_star: torch.Tensor, santana: SantanaLog) -> Tuple[torch.Tensor, float]:
        """
        Mock implementation that produces context and trust score.
        """
        batch_size = s_star.size(0)

        # Create context vector based on trajectory length
        num_steps = len(santana)

        # Simple mock: use s_star to create context
        # In real implementation, this would encode the full trajectory
        if s_star.size(-1) >= self.config.context_dim:
            v_ctx = s_star[:, : self.config.context_dim]
        else:
            # Pad if necessary
            v_ctx = torch.zeros(batch_size, self.config.context_dim)
            v_ctx[:, : s_star.size(-1)] = s_star

        v_ctx = self.encoder(v_ctx)

        # Trust score based on convergence (mock: based on num_steps)
        # More steps = lower trust (didn't converge quickly)
        trust_score = max(0.0, 1.0 - num_steps * 0.1)

        return v_ctx, trust_score


class TestBaseVipassana:
    """Tests for BaseVipassana interface."""

    def test_interface_inheritance(self):
        """Test that BaseVipassana inherits from nn.Module."""
        config = StandardVipassanaConfig()
        vipassana = MockVipassana(config)

        assert isinstance(vipassana, torch.nn.Module)

    def test_forward_signature(self):
        """Test forward method returns correct types."""
        config = StandardVipassanaConfig(context_dim=32)
        vipassana = MockVipassana(config)

        s_star = torch.randn(4, 64)
        santana = SantanaLog()
        santana.add(torch.randn(4, 64))

        v_ctx, trust_score = vipassana(s_star, santana)

        assert isinstance(v_ctx, torch.Tensor)
        assert isinstance(trust_score, float)

    def test_context_vector_shape(self):
        """Test that context vector has correct shape."""
        # StandardVipassanaConfig computes context_dim = gru_hidden_dim + metric_proj_dim
        config = StandardVipassanaConfig(gru_hidden_dim=8, metric_proj_dim=8)
        vipassana = MockVipassana(config)

        batch_size = 8
        s_star = torch.randn(batch_size, 64)
        santana = SantanaLog()
        santana.add(torch.randn(batch_size, 64))

        v_ctx, _ = vipassana(s_star, santana)

        assert v_ctx.shape == (batch_size, config.context_dim)
        assert config.context_dim == 16  # 8 + 8

    def test_trust_score_range(self):
        """Test that trust score is in valid range [0, 1]."""
        config = StandardVipassanaConfig()
        vipassana = MockVipassana(config)

        s_star = torch.randn(4, 64)
        santana = SantanaLog()
        santana.add(torch.randn(4, 64))

        _, trust_score = vipassana(s_star, santana)

        assert 0.0 <= trust_score <= 1.0

    def test_trust_score_varies_with_trajectory(self):
        """Test that trust score changes based on trajectory length."""
        config = StandardVipassanaConfig()
        vipassana = MockVipassana(config)

        s_star = torch.randn(4, 64)

        # Short trajectory
        short_santana = SantanaLog()
        short_santana.add(torch.randn(4, 64))
        _, short_trust = vipassana(s_star, short_santana)

        # Long trajectory
        long_santana = SantanaLog()
        for _ in range(10):
            long_santana.add(torch.randn(4, 64))
        _, long_trust = vipassana(s_star, long_santana)

        # In mock implementation, longer trajectory = lower trust
        assert short_trust > long_trust

    def test_config_stored(self):
        """Test that config is properly stored."""
        config = StandardVipassanaConfig(gru_hidden_dim=32, metric_proj_dim=32)
        vipassana = MockVipassana(config)

        assert vipassana.config is config
        assert vipassana.config.context_dim == 64  # 32 + 32

    def test_lstm_config(self):
        """Test LSTMVipassanaConfig can be used."""
        config = LSTMVipassanaConfig(context_dim=32, hidden_dim=64, num_layers=2, bidirectional=True)
        vipassana = MockVipassana(config)

        assert vipassana.config.num_layers == 2
        assert vipassana.config.bidirectional is True

    def test_trainable_parameters(self):
        """Test that mock vipassana has trainable parameters."""
        config = StandardVipassanaConfig()
        vipassana = MockVipassana(config)

        params = list(vipassana.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
