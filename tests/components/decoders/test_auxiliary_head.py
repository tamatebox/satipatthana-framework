"""
Tests for SimpleAuxHead implementation.
"""

import pytest
import torch

from samadhi.components.decoders.auxiliary import SimpleAuxHead
from samadhi.configs.decoders import SimpleAuxHeadConfig


class TestSimpleAuxHead:
    """Tests for SimpleAuxHead implementation."""

    def test_default_config(self):
        """Test initialization with config."""
        config = SimpleAuxHeadConfig(dim=64, output_dim=10)
        head = SimpleAuxHead(config)

        assert head.dim == 64
        assert head.output_dim == 10

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = SimpleAuxHeadConfig(dim=128, output_dim=5, decoder_hidden_dim=128)
        head = SimpleAuxHead(config)

        assert head.dim == 128
        assert head.output_dim == 5

    def test_output_shape(self):
        """Test that output shape is correct."""
        config = SimpleAuxHeadConfig(dim=64, output_dim=10)
        head = SimpleAuxHead(config)

        batch_size = 8
        s_star = torch.randn(batch_size, 64)

        output = head(s_star)

        assert output.shape == (batch_size, 10)

    def test_various_batch_sizes(self):
        """Test with various batch sizes."""
        config = SimpleAuxHeadConfig(dim=64, output_dim=10)
        head = SimpleAuxHead(config)

        for batch_size in [1, 4, 16, 32]:
            s_star = torch.randn(batch_size, 64)
            output = head(s_star)
            assert output.shape == (batch_size, 10)

    def test_various_output_dims(self):
        """Test with various output dimensions."""
        for output_dim in [2, 5, 10, 100]:
            config = SimpleAuxHeadConfig(dim=64, output_dim=output_dim)
            head = SimpleAuxHead(config)

            s_star = torch.randn(4, 64)
            output = head(s_star)
            assert output.shape == (4, output_dim)

    def test_various_input_dims(self):
        """Test with various input dimensions."""
        for dim in [32, 64, 128, 256]:
            config = SimpleAuxHeadConfig(dim=dim, output_dim=10)
            head = SimpleAuxHead(config)

            s_star = torch.randn(4, dim)
            output = head(s_star)
            assert output.shape == (4, 10)

    def test_is_nn_module(self):
        """Test that head is an nn.Module."""
        config = SimpleAuxHeadConfig(dim=64, output_dim=10)
        head = SimpleAuxHead(config)

        assert isinstance(head, torch.nn.Module)

    def test_trainable_parameters(self):
        """Test that head has trainable parameters."""
        config = SimpleAuxHeadConfig(dim=64, output_dim=10)
        head = SimpleAuxHead(config)

        params = list(head.parameters())
        assert len(params) > 0

        trainable = sum(p.numel() for p in params if p.requires_grad)
        assert trainable > 0

    def test_gradient_flow(self):
        """Test that gradients flow through the head."""
        config = SimpleAuxHeadConfig(dim=64, output_dim=10)
        head = SimpleAuxHead(config)

        s_star = torch.randn(4, 64, requires_grad=True)
        output = head(s_star)
        loss = output.sum()
        loss.backward()

        assert s_star.grad is not None
        assert s_star.grad.shape == s_star.shape

    def test_device_consistency(self):
        """Test that output device matches input device."""
        config = SimpleAuxHeadConfig(dim=64, output_dim=10)
        head = SimpleAuxHead(config)

        s_star = torch.randn(4, 64)
        output = head(s_star)

        assert output.device == s_star.device

    def test_simpler_than_conditional(self):
        """Test that SimpleAuxHead has fewer parameters than ConditionalDecoder."""
        from samadhi.components.decoders.conditional import ConditionalDecoder
        from samadhi.configs.decoders import ConditionalDecoderConfig

        aux_config = SimpleAuxHeadConfig(dim=64, output_dim=10, decoder_hidden_dim=64)
        aux_head = SimpleAuxHead(aux_config)

        cond_config = ConditionalDecoderConfig(
            dim=64, context_dim=32, output_dim=10, decoder_hidden_dim=128
        )
        cond_decoder = ConditionalDecoder(cond_config)

        aux_params = sum(p.numel() for p in aux_head.parameters())
        cond_params = sum(p.numel() for p in cond_decoder.parameters())

        # SimpleAuxHead should have fewer parameters
        assert aux_params < cond_params
