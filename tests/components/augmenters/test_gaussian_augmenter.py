"""
Tests for GaussianNoiseAugmenter implementation.
"""

import pytest
import torch

from samadhi.components.augmenters.gaussian import GaussianNoiseAugmenter
from samadhi.configs.augmenter import GaussianNoiseAugmenterConfig


class TestGaussianNoiseAugmenter:
    """Tests for GaussianNoiseAugmenter implementation."""

    def test_default_config(self):
        """Test initialization with default config."""
        augmenter = GaussianNoiseAugmenter()

        assert augmenter.config is not None
        assert isinstance(augmenter.config, GaussianNoiseAugmenterConfig)
        assert augmenter.config.max_noise_std == 0.1

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = GaussianNoiseAugmenterConfig(max_noise_std=0.5)
        augmenter = GaussianNoiseAugmenter(config)

        assert augmenter.config.max_noise_std == 0.5

    def test_forward_output_shapes(self):
        """Test that output shapes match input."""
        augmenter = GaussianNoiseAugmenter()

        batch_size = 8
        x = torch.randn(batch_size, 32)
        x_aug, severity = augmenter(x, noise_level=0.5)

        assert x_aug.shape == x.shape
        assert severity.shape == (batch_size,)

    def test_zero_noise_level_unchanged(self):
        """Test input is unchanged with noise_level=0."""
        augmenter = GaussianNoiseAugmenter()

        x = torch.randn(4, 32)
        x_aug, severity = augmenter(x, noise_level=0.0)

        assert torch.allclose(x_aug, x)
        assert (severity == 0.0).all()

    def test_nonzero_noise_changes_input(self):
        """Test that nonzero noise level changes input."""
        config = GaussianNoiseAugmenterConfig(max_noise_std=0.5)
        augmenter = GaussianNoiseAugmenter(config)

        x = torch.randn(4, 32)
        x_aug, severity = augmenter(x, noise_level=1.0)

        # With max_noise_std=0.5 and noise_level=1.0, input should change
        assert not torch.allclose(x_aug, x)
        assert (severity == 0.5).all()

    def test_severity_calculation(self):
        """Test severity is correctly calculated as noise_level * max_noise_std."""
        config = GaussianNoiseAugmenterConfig(max_noise_std=0.2)
        augmenter = GaussianNoiseAugmenter(config)

        x = torch.randn(4, 32)

        # Test various noise levels
        for noise_level in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _, severity = augmenter(x, noise_level=noise_level)
            expected_severity = noise_level * 0.2
            assert torch.allclose(
                severity, torch.full((4,), expected_severity)
            ), f"Expected {expected_severity}, got {severity[0].item()}"

    def test_noise_scaling_with_level(self):
        """Test that higher noise_level produces more variance."""
        config = GaussianNoiseAugmenterConfig(max_noise_std=1.0)
        augmenter = GaussianNoiseAugmenter(config)

        # Use larger sample for statistical stability
        x = torch.zeros(1000, 64)

        # Get differences at different noise levels
        torch.manual_seed(42)
        x_aug_low, _ = augmenter(x, noise_level=0.1)
        diff_low = (x_aug_low - x).std().item()

        torch.manual_seed(42)
        x_aug_high, _ = augmenter(x, noise_level=1.0)
        diff_high = (x_aug_high - x).std().item()

        # Higher noise level should produce larger differences
        assert diff_high > diff_low

    def test_output_is_clone_at_zero_noise(self):
        """Test that output is a clone when noise is zero."""
        augmenter = GaussianNoiseAugmenter()

        x = torch.randn(4, 32)
        x_aug, _ = augmenter(x, noise_level=0.0)

        # Should be a different tensor object
        assert x_aug is not x
        # But with same values
        assert torch.allclose(x_aug, x)

    def test_various_input_shapes(self):
        """Test augmenter works with various input shapes."""
        augmenter = GaussianNoiseAugmenter()

        # 2D input (tabular)
        x_2d = torch.randn(4, 32)
        x_aug_2d, sev_2d = augmenter(x_2d, noise_level=0.5)
        assert x_aug_2d.shape == x_2d.shape
        assert sev_2d.shape == (4,)

        # 3D input (sequence)
        x_3d = torch.randn(4, 10, 32)
        x_aug_3d, sev_3d = augmenter(x_3d, noise_level=0.5)
        assert x_aug_3d.shape == x_3d.shape
        assert sev_3d.shape == (4,)

        # 4D input (image)
        x_4d = torch.randn(4, 3, 28, 28)
        x_aug_4d, sev_4d = augmenter(x_4d, noise_level=0.5)
        assert x_aug_4d.shape == x_4d.shape
        assert sev_4d.shape == (4,)

    def test_device_consistency(self):
        """Test that output device matches input device."""
        augmenter = GaussianNoiseAugmenter()

        x = torch.randn(4, 32)
        x_aug, severity = augmenter(x, noise_level=0.5)

        assert x_aug.device == x.device
        assert severity.device == x.device

    def test_dtype_consistency(self):
        """Test that output dtype matches input dtype."""
        augmenter = GaussianNoiseAugmenter()

        # Test float32
        x_f32 = torch.randn(4, 32, dtype=torch.float32)
        x_aug_f32, sev_f32 = augmenter(x_f32, noise_level=0.5)
        assert x_aug_f32.dtype == x_f32.dtype
        assert sev_f32.dtype == x_f32.dtype

        # Test float64
        x_f64 = torch.randn(4, 32, dtype=torch.float64)
        x_aug_f64, sev_f64 = augmenter(x_f64, noise_level=0.5)
        assert x_aug_f64.dtype == x_f64.dtype
        assert sev_f64.dtype == x_f64.dtype

    def test_noise_is_gaussian(self):
        """Test that added noise follows Gaussian distribution."""
        config = GaussianNoiseAugmenterConfig(max_noise_std=1.0)
        augmenter = GaussianNoiseAugmenter(config)

        # Large sample for statistical test
        x = torch.zeros(10000, 64)
        x_aug, _ = augmenter(x, noise_level=1.0)

        # Check mean and std of noise (should be ~0 and ~1.0)
        noise = x_aug - x
        noise_mean = noise.mean().item()
        noise_std = noise.std().item()

        # Allow some tolerance for randomness
        assert abs(noise_mean) < 0.05, f"Mean too far from 0: {noise_mean}"
        assert abs(noise_std - 1.0) < 0.1, f"Std too far from 1.0: {noise_std}"
