"""
Tests for StandardVipassana implementation (v4.1 GRU-based with Triple Score).
"""

import pytest
import torch

from satipatthana.components.vipassana.standard import StandardVipassana
from satipatthana.components.vipassana.base import VipassanaOutput
from satipatthana.configs.vipassana import StandardVipassanaConfig
from satipatthana.core.santana import SantanaLog


class TestStandardVipassana:
    """Tests for StandardVipassana implementation."""

    def test_default_config(self):
        """Test initialization with default config."""
        vipassana = StandardVipassana()

        assert vipassana.config is not None
        assert isinstance(vipassana.config, StandardVipassanaConfig)
        # context_dim = gru_hidden_dim (32) + metric_proj_dim (32) = 64
        assert vipassana.config.context_dim == 64
        assert vipassana.config.gru_hidden_dim == 32
        assert vipassana.config.metric_proj_dim == 32
        assert vipassana.config.latent_dim == 64

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = StandardVipassanaConfig(
            latent_dim=128,
            gru_hidden_dim=64,
            metric_proj_dim=32,
            max_steps=20,
        )
        vipassana = StandardVipassana(config)

        assert vipassana.config.latent_dim == 128
        assert vipassana.config.gru_hidden_dim == 64
        assert vipassana.config.metric_proj_dim == 32
        assert vipassana.config.context_dim == 96  # 64 + 32

    def test_output_shapes(self):
        """Test that output shapes are correct (Triple Score)."""
        config = StandardVipassanaConfig(
            latent_dim=64,
            gru_hidden_dim=16,
            metric_proj_dim=16,
        )
        vipassana = StandardVipassana(config)

        batch_size = 4
        state_dim = 64

        santana = SantanaLog()
        state = torch.randn(batch_size, state_dim)
        santana.add(state, energy=0.1)
        santana.add(state, energy=0.05)

        s_star = torch.randn(batch_size, state_dim)
        output = vipassana(s_star, santana)

        assert isinstance(output, VipassanaOutput)
        assert output.v_ctx.shape == (batch_size, 32)  # 16 + 16
        assert output.trust_score.shape == (batch_size, 1)
        assert output.conformity_score.shape == (batch_size, 1)
        assert output.confidence_score.shape == (batch_size, 1)

    def test_trust_score_in_valid_range(self):
        """Test that all scores are in [0.0, 1.0] range."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.1)

        s_star = torch.randn(4, 32)
        output = vipassana(s_star, santana)

        assert (output.trust_score >= 0.0).all() and (output.trust_score <= 1.0).all()
        assert (output.conformity_score >= 0.0).all() and (output.conformity_score <= 1.0).all()
        assert (output.confidence_score >= 0.0).all() and (output.confidence_score <= 1.0).all()

    def test_handles_empty_santana(self):
        """Test behavior with empty SantanaLog."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        santana = SantanaLog()  # Empty
        s_star = torch.randn(4, 32)

        output = vipassana(s_star, santana)

        assert output.v_ctx.shape == (4, 64)  # context_dim = 32 + 32
        assert (output.trust_score >= 0.0).all() and (output.trust_score <= 1.0).all()

    def test_handles_single_step(self):
        """Test with single step in trajectory."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.1)

        s_star = torch.randn(4, 32)
        output = vipassana(s_star, santana)

        assert output.v_ctx.shape == (4, 64)
        assert (output.trust_score >= 0.0).all() and (output.trust_score <= 1.0).all()

    def test_handles_long_trajectory(self):
        """Test with long trajectory."""
        config = StandardVipassanaConfig(latent_dim=32, max_steps=30)
        vipassana = StandardVipassana(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        for i in range(20):
            santana.add(state, energy=0.1 / (i + 1))

        s_star = torch.randn(4, 32)
        output = vipassana(s_star, santana)

        assert output.v_ctx.shape == (4, 64)
        assert (output.trust_score >= 0.0).all() and (output.trust_score <= 1.0).all()

    def test_context_dim_customizable(self):
        """Test that context_dim is computed from gru_hidden_dim + metric_proj_dim."""
        test_cases = [
            (8, 8, 16),
            (16, 16, 32),
            (32, 32, 64),
            (64, 64, 128),
        ]
        for gru_dim, metric_dim, expected_ctx_dim in test_cases:
            config = StandardVipassanaConfig(
                latent_dim=32,
                gru_hidden_dim=gru_dim,
                metric_proj_dim=metric_dim,
            )
            vipassana = StandardVipassana(config)

            santana = SantanaLog()
            state = torch.randn(4, 32)
            santana.add(state, energy=0.1)

            s_star = torch.randn(4, 32)
            output = vipassana(s_star, santana)

            assert output.v_ctx.shape == (4, expected_ctx_dim)

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        for batch_size in [1, 4, 16, 32]:
            santana = SantanaLog()
            state = torch.randn(batch_size, 32)
            santana.add(state, energy=0.1)

            s_star = torch.randn(batch_size, 32)
            output = vipassana(s_star, santana)

            assert output.v_ctx.shape == (batch_size, 64)
            assert output.trust_score.shape == (batch_size, 1)
            assert (output.trust_score >= 0.0).all() and (output.trust_score <= 1.0).all()

    def test_gru_encodes_trajectory_sequence(self):
        """Test that GRU encodes sequence information (not just aggregation)."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        batch_size = 4

        # Trajectory 1: increasing states
        santana1 = SantanaLog()
        for i in range(5):
            state = torch.ones(batch_size, 32) * i
            santana1.add(state)

        # Trajectory 2: decreasing states (same values, different order)
        santana2 = SantanaLog()
        for i in range(4, -1, -1):
            state = torch.ones(batch_size, 32) * i
            santana2.add(state)

        s_star = torch.randn(batch_size, 32)
        output1 = vipassana(s_star, santana1)
        output2 = vipassana(s_star, santana2)

        # GRU should produce different contexts for different sequences
        assert not torch.allclose(output1.v_ctx, output2.v_ctx, atol=1e-3)

    def test_convergence_steps_affects_metrics(self):
        """Test that convergence_steps field is used in metrics computation."""
        config = StandardVipassanaConfig(latent_dim=32, max_steps=10)
        vipassana = StandardVipassana(config)

        batch_size = 4

        # Trajectory with early convergence
        santana_early = SantanaLog()
        for i in range(10):
            santana_early.add(torch.randn(batch_size, 32))
        santana_early.convergence_steps = torch.tensor([3, 3, 3, 3])

        # Trajectory with late convergence
        santana_late = SantanaLog()
        for i in range(10):
            santana_late.add(torch.randn(batch_size, 32))
        santana_late.convergence_steps = torch.tensor([9, 9, 9, 9])

        s_star = torch.randn(batch_size, 32)
        output_early = vipassana(s_star, santana_early)
        output_late = vipassana(s_star, santana_late)

        # Different convergence steps should produce different contexts
        assert not torch.allclose(output_early.v_ctx, output_late.v_ctx)

    def test_with_probes(self):
        """Test that probes affect semantic features."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        batch_size = 4
        n_probes = 5

        santana = SantanaLog()
        santana.add(torch.randn(batch_size, 32))

        # s_star close to probe 0
        probes = torch.randn(n_probes, 32)
        s_star_close = probes[0].unsqueeze(0).expand(batch_size, -1) + torch.randn(batch_size, 32) * 0.1

        # s_star far from all probes
        s_star_far = probes.mean(dim=0).unsqueeze(0).expand(batch_size, -1) + torch.randn(batch_size, 32) * 10

        output_close = vipassana(s_star_close, santana, probes=probes)
        output_far = vipassana(s_star_far, santana, probes=probes)

        # Different proximity to probes should produce different contexts
        assert not torch.allclose(output_close.v_ctx, output_far.v_ctx)

    def test_with_recon_error(self):
        """Test that recon_error affects context."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        batch_size = 4
        santana = SantanaLog()
        santana.add(torch.randn(batch_size, 32))

        s_star = torch.randn(batch_size, 32)

        # Low recon error (in-distribution)
        recon_error_low = torch.ones(batch_size, 1) * 0.01
        output_low = vipassana(s_star, santana, recon_error=recon_error_low)

        # High recon error (out-of-distribution)
        recon_error_high = torch.ones(batch_size, 1) * 10.0
        output_high = vipassana(s_star, santana, recon_error=recon_error_high)

        # Different recon errors should produce different contexts
        assert not torch.allclose(output_low.v_ctx, output_high.v_ctx)

    def test_is_nn_module(self):
        """Test that StandardVipassana is an nn.Module."""
        vipassana = StandardVipassana()

        assert isinstance(vipassana, torch.nn.Module)

    def test_trainable_parameters(self):
        """Test that vipassana has trainable parameters."""
        vipassana = StandardVipassana()

        # Check that parameters exist
        params = list(vipassana.parameters())
        assert len(params) > 0

        # Check that some parameters require grad
        trainable = sum(p.numel() for p in params if p.requires_grad)
        assert trainable > 0

    def test_device_consistency(self):
        """Test that output device matches input device."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state, energy=0.1)

        s_star = torch.randn(4, 32)
        output = vipassana(s_star, santana)

        assert output.v_ctx.device == s_star.device
        assert output.trust_score.device == s_star.device
        assert output.conformity_score.device == s_star.device
        assert output.confidence_score.device == s_star.device

    def test_num_metrics_constant(self):
        """Test that NUM_METRICS is correctly set."""
        assert StandardVipassana.NUM_METRICS == 8

    def test_gradient_flow(self):
        """Test that gradients flow through the model (especially via conformity/confidence)."""
        config = StandardVipassanaConfig(latent_dim=32)
        vipassana = StandardVipassana(config)

        santana = SantanaLog()
        state = torch.randn(4, 32)
        santana.add(state)
        santana.add(state + torch.randn(4, 32) * 0.1)

        s_star = torch.randn(4, 32, requires_grad=True)
        output = vipassana(s_star, santana)

        # Backprop through all three scores (conformity and confidence provide gradients to GRU)
        loss = output.trust_score.mean() + output.conformity_score.mean() + output.confidence_score.mean()
        loss.backward()

        # Check gradients exist
        assert s_star.grad is not None
        assert not torch.all(s_star.grad == 0)
