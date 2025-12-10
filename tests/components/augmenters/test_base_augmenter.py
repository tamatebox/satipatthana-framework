"""
Tests for BaseAugmenter interface.
"""

import pytest
import torch
from typing import Tuple

from samadhi.components.augmenters.base import BaseAugmenter
from samadhi.configs.augmenter import IdentityAugmenterConfig


class MockAugmenter(BaseAugmenter):
    """Mock implementation of BaseAugmenter for testing."""

    def forward(
        self, x: torch.Tensor, noise_level: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identity augmentation for testing."""
        batch_size = x.size(0)
        severity = torch.full((batch_size,), noise_level)
        return x.clone(), severity


class TestBaseAugmenter:
    """Tests for BaseAugmenter interface."""

    def test_interface_inheritance(self):
        """Test that BaseAugmenter inherits from nn.Module."""
        config = IdentityAugmenterConfig()
        augmenter = MockAugmenter(config)

        assert isinstance(augmenter, torch.nn.Module)

    def test_forward_signature(self):
        """Test forward method signature returns correct types."""
        config = IdentityAugmenterConfig()
        augmenter = MockAugmenter(config)

        x = torch.randn(4, 32)
        x_aug, severity = augmenter(x, noise_level=0.5)

        assert isinstance(x_aug, torch.Tensor)
        assert isinstance(severity, torch.Tensor)

    def test_forward_output_shapes(self):
        """Test that output shapes are correct."""
        config = IdentityAugmenterConfig()
        augmenter = MockAugmenter(config)

        batch_size = 8
        x = torch.randn(batch_size, 32)
        x_aug, severity = augmenter(x, noise_level=0.5)

        # x_augmented should have same shape as input
        assert x_aug.shape == x.shape

        # severity should be (Batch,)
        assert severity.shape == (batch_size,)

    def test_forward_noise_level_zero(self):
        """Test that noise_level=0 produces no augmentation."""
        config = IdentityAugmenterConfig()
        augmenter = MockAugmenter(config)

        x = torch.randn(4, 32)
        x_aug, severity = augmenter(x, noise_level=0.0)

        # For mock identity augmenter, input should be unchanged
        assert torch.allclose(x_aug, x)
        assert (severity == 0.0).all()

    def test_config_stored(self):
        """Test that config is properly stored."""
        config = IdentityAugmenterConfig()
        augmenter = MockAugmenter(config)

        assert augmenter.config is config
