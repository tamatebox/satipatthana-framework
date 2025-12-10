"""
Tests for ConditionalDecoder implementation.
"""

import pytest
import torch

from samadhi.components.decoders.conditional import ConditionalDecoder
from samadhi.configs.decoders import ConditionalDecoderConfig


class TestConditionalDecoder:
    """Tests for ConditionalDecoder implementation."""

    def test_default_config(self):
        """Test initialization with default-ish config."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        assert decoder.dim == 64
        assert decoder.context_dim == 32
        assert decoder.output_dim == 10

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = ConditionalDecoderConfig(
            dim=128, context_dim=64, output_dim=5, decoder_hidden_dim=256
        )
        decoder = ConditionalDecoder(config)

        assert decoder.dim == 128
        assert decoder.context_dim == 64
        assert decoder.output_dim == 5

    def test_output_shape(self):
        """Test that output shape is correct."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        batch_size = 8
        s_star = torch.randn(batch_size, 64)
        v_ctx = torch.randn(batch_size, 32)
        s_and_ctx = torch.cat([s_star, v_ctx], dim=1)

        output = decoder(s_and_ctx)

        assert output.shape == (batch_size, 10)

    def test_forward_with_concat(self):
        """Test the convenience forward_with_concat method."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        batch_size = 8
        s_star = torch.randn(batch_size, 64)
        v_ctx = torch.randn(batch_size, 32)

        output = decoder.forward_with_concat(s_star, v_ctx)

        assert output.shape == (batch_size, 10)

    def test_forward_equivalence(self):
        """Test that forward and forward_with_concat produce same result."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        batch_size = 4
        s_star = torch.randn(batch_size, 64)
        v_ctx = torch.randn(batch_size, 32)
        s_and_ctx = torch.cat([s_star, v_ctx], dim=1)

        output1 = decoder(s_and_ctx)
        output2 = decoder.forward_with_concat(s_star, v_ctx)

        assert torch.allclose(output1, output2)

    def test_various_batch_sizes(self):
        """Test with various batch sizes."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        for batch_size in [1, 4, 16, 32]:
            s_and_ctx = torch.randn(batch_size, 64 + 32)
            output = decoder(s_and_ctx)
            assert output.shape == (batch_size, 10)

    def test_various_output_dims(self):
        """Test with various output dimensions."""
        for output_dim in [2, 5, 10, 100]:
            config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=output_dim)
            decoder = ConditionalDecoder(config)

            s_and_ctx = torch.randn(4, 64 + 32)
            output = decoder(s_and_ctx)
            assert output.shape == (4, output_dim)

    def test_is_nn_module(self):
        """Test that decoder is an nn.Module."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        assert isinstance(decoder, torch.nn.Module)

    def test_trainable_parameters(self):
        """Test that decoder has trainable parameters."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        params = list(decoder.parameters())
        assert len(params) > 0

        trainable = sum(p.numel() for p in params if p.requires_grad)
        assert trainable > 0

    def test_gradient_flow(self):
        """Test that gradients flow through the decoder."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        s_and_ctx = torch.randn(4, 64 + 32, requires_grad=True)
        output = decoder(s_and_ctx)
        loss = output.sum()
        loss.backward()

        assert s_and_ctx.grad is not None
        assert s_and_ctx.grad.shape == s_and_ctx.shape

    def test_device_consistency(self):
        """Test that output device matches input device."""
        config = ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10)
        decoder = ConditionalDecoder(config)

        s_and_ctx = torch.randn(4, 64 + 32)
        output = decoder(s_and_ctx)

        assert output.device == s_and_ctx.device
