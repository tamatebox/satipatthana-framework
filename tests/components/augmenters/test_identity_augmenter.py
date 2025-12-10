"""
Tests for IdentityAugmenter implementation.
"""

import pytest
import torch

from samadhi.components.augmenters.identity import IdentityAugmenter
from samadhi.configs.augmenter import IdentityAugmenterConfig


class TestIdentityAugmenter:
    """Tests for IdentityAugmenter implementation."""

    def test_default_config(self):
        """Test initialization with default config."""
        augmenter = IdentityAugmenter()

        assert augmenter.config is not None
        assert isinstance(augmenter.config, IdentityAugmenterConfig)

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = IdentityAugmenterConfig()
        augmenter = IdentityAugmenter(config)

        assert augmenter.config is config

    def test_forward_output_shapes(self):
        """Test that output shapes match input."""
        augmenter = IdentityAugmenter()

        batch_size = 8
        x = torch.randn(batch_size, 32)
        x_aug, severity = augmenter(x, noise_level=0.5)

        assert x_aug.shape == x.shape
        assert severity.shape == (batch_size,)

    def test_input_unchanged_zero_noise(self):
        """Test input is unchanged with noise_level=0."""
        augmenter = IdentityAugmenter()

        x = torch.randn(4, 32)
        x_aug, severity = augmenter(x, noise_level=0.0)

        assert torch.allclose(x_aug, x)
        assert (severity == 0.0).all()

    def test_input_unchanged_any_noise_level(self):
        """Test input is unchanged regardless of noise_level (identity)."""
        augmenter = IdentityAugmenter()

        x = torch.randn(4, 32)

        # Test various noise levels
        for noise_level in [0.0, 0.5, 1.0]:
            x_aug, severity = augmenter(x, noise_level=noise_level)
            assert torch.allclose(x_aug, x)
            assert (severity == 0.0).all()

    def test_severity_always_zero(self):
        """Test that severity is always zero for identity augmenter."""
        augmenter = IdentityAugmenter()

        x = torch.randn(8, 64)
        _, severity = augmenter(x, noise_level=1.0)

        assert (severity == 0.0).all()

    def test_output_is_clone(self):
        """Test that output is a clone, not the same tensor."""
        augmenter = IdentityAugmenter()

        x = torch.randn(4, 32)
        x_aug, _ = augmenter(x)

        # Should be a different tensor object
        assert x_aug is not x
        # But with same values
        assert torch.allclose(x_aug, x)

    def test_various_input_shapes(self):
        """Test augmenter works with various input shapes."""
        augmenter = IdentityAugmenter()

        # 2D input (tabular)
        x_2d = torch.randn(4, 32)
        x_aug_2d, sev_2d = augmenter(x_2d)
        assert x_aug_2d.shape == x_2d.shape
        assert sev_2d.shape == (4,)

        # 3D input (sequence)
        x_3d = torch.randn(4, 10, 32)
        x_aug_3d, sev_3d = augmenter(x_3d)
        assert x_aug_3d.shape == x_3d.shape
        assert sev_3d.shape == (4,)

        # 4D input (image)
        x_4d = torch.randn(4, 3, 28, 28)
        x_aug_4d, sev_4d = augmenter(x_4d)
        assert x_aug_4d.shape == x_4d.shape
        assert sev_4d.shape == (4,)

    def test_device_consistency(self):
        """Test that output device matches input device."""
        augmenter = IdentityAugmenter()

        x = torch.randn(4, 32)
        x_aug, severity = augmenter(x)

        assert x_aug.device == x.device
        assert severity.device == x.device

    def test_dtype_consistency(self):
        """Test that output dtype matches input dtype."""
        augmenter = IdentityAugmenter()

        # Test float32
        x_f32 = torch.randn(4, 32, dtype=torch.float32)
        x_aug_f32, sev_f32 = augmenter(x_f32)
        assert x_aug_f32.dtype == x_f32.dtype
        assert sev_f32.dtype == x_f32.dtype

        # Test float64
        x_f64 = torch.randn(4, 32, dtype=torch.float64)
        x_aug_f64, sev_f64 = augmenter(x_f64)
        assert x_aug_f64.dtype == x_f64.dtype
        assert sev_f64.dtype == x_f64.dtype
