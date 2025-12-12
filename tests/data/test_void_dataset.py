"""
Tests for VoidDataset and related classes.
"""

import pytest
import torch
from torch.utils.data import Dataset

from satipatthana.data.void_dataset import VoidDataset, GaussianNoiseVoid, UniformNoiseVoid, FilteredNoiseVoid


class DummyStaticDataset(Dataset):
    """Dummy dataset for testing static mode."""

    def __init__(self, size: int = 100, dim: int = 32):
        self.size = size
        self.data = torch.randn(size, dim)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"x": self.data[idx]}


class TestVoidDatasetStaticMode:
    """Tests for VoidDataset in static mode (wrapping existing datasets)."""

    def test_wraps_dataset(self):
        """Test wrapping a PyTorch Dataset."""
        source = DummyStaticDataset(size=50, dim=16)
        void_data = VoidDataset(source)

        assert len(void_data) == 50
        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (16,)

    def test_wraps_list_of_tensors(self):
        """Test wrapping a list of tensors."""
        tensors = [torch.randn(8) for _ in range(30)]
        void_data = VoidDataset(tensors)

        assert len(void_data) == 30
        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (8,)

    def test_wraps_list_of_dicts(self):
        """Test wrapping a list of dicts."""
        dicts = [{"x": torch.randn(8), "y": i} for i in range(20)]
        void_data = VoidDataset(dicts)

        assert len(void_data) == 20
        sample = void_data[5]
        assert "x" in sample

    def test_wraps_list_of_tuples(self):
        """Test wrapping a list of (data, label) tuples."""
        tuples = [(torch.randn(8), i) for i in range(25)]
        void_data = VoidDataset(tuples)

        assert len(void_data) == 25
        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (8,)

    def test_index_wrap_around(self):
        """Test that index wraps around for out-of-bounds access."""
        tensors = [torch.ones(4) * i for i in range(5)]
        void_data = VoidDataset(tensors)

        # Index 7 should wrap to index 2 (7 % 5 = 2)
        sample_2 = void_data[2]
        sample_7 = void_data[7]
        assert torch.allclose(sample_2["x"], sample_7["x"])

    def test_handles_input_values_key(self):
        """Test handling dicts with 'input_values' key."""
        dicts = [{"input_values": torch.randn(8)} for _ in range(10)]
        void_data = VoidDataset(dicts)

        sample = void_data[0]
        assert "x" in sample

    def test_handles_pixel_values_key(self):
        """Test handling dicts with 'pixel_values' key."""
        dicts = [{"pixel_values": torch.randn(3, 32, 32)} for _ in range(10)]
        void_data = VoidDataset(dicts)

        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (3, 32, 32)


class TestVoidDatasetDynamicMode:
    """Tests for VoidDataset in dynamic mode (callable generator)."""

    def test_dynamic_with_callable(self):
        """Test dynamic mode with a callable."""

        def generator():
            return {"x": torch.randn(16)}

        void_data = VoidDataset(generator, length=100)

        assert len(void_data) == 100
        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (16,)

    def test_dynamic_requires_length(self):
        """Test that dynamic mode requires length parameter."""

        def generator():
            return torch.randn(16)

        with pytest.raises(ValueError, match="requires 'length'"):
            VoidDataset(generator)  # No length provided

    def test_dynamic_generates_different_samples(self):
        """Test that dynamic mode generates different samples each call."""

        def generator():
            return {"x": torch.randn(16)}

        void_data = VoidDataset(generator, length=100)

        sample1 = void_data[0]
        sample2 = void_data[0]  # Same index, but should be different

        # With random generation, samples should be different
        assert not torch.allclose(sample1["x"], sample2["x"])

    def test_dynamic_with_lambda(self):
        """Test dynamic mode with lambda function."""
        void_data = VoidDataset(lambda: {"x": torch.randn(8)}, length=50)

        assert len(void_data) == 50
        sample = void_data[0]
        assert sample["x"].shape == (8,)

    def test_dynamic_with_tensor_return(self):
        """Test dynamic mode when callable returns raw tensor."""
        void_data = VoidDataset(lambda: torch.randn(8), length=50)

        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (8,)


class TestVoidDatasetTransform:
    """Tests for VoidDataset with transforms."""

    def test_transform_applied(self):
        """Test that transform is applied to samples."""

        def double_transform(x):
            return x * 2

        tensors = [torch.ones(4) for _ in range(10)]
        void_data = VoidDataset(tensors, transform=double_transform)

        sample = void_data[0]
        assert torch.allclose(sample["x"], torch.ones(4) * 2)

    def test_transform_with_dynamic(self):
        """Test transform with dynamic mode."""

        def normalize(x):
            return x / x.norm()

        void_data = VoidDataset(lambda: {"x": torch.randn(8)}, length=50, transform=normalize)

        sample = void_data[0]
        # Normalized vector should have norm ~= 1
        assert abs(sample["x"].norm().item() - 1.0) < 1e-5


class TestGaussianNoiseVoid:
    """Tests for GaussianNoiseVoid convenience class."""

    def test_basic_usage(self):
        """Test basic Gaussian noise generation."""
        void_data = GaussianNoiseVoid(shape=(32,), length=100)

        assert len(void_data) == 100
        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (32,)

    def test_scale_parameter(self):
        """Test that scale parameter affects noise magnitude."""
        void_small = GaussianNoiseVoid(shape=(1000,), length=100, scale=0.1)
        void_large = GaussianNoiseVoid(shape=(1000,), length=100, scale=10.0)

        # Get multiple samples and compute std
        samples_small = torch.stack([void_small[i]["x"] for i in range(50)])
        samples_large = torch.stack([void_large[i]["x"] for i in range(50)])

        std_small = samples_small.std().item()
        std_large = samples_large.std().item()

        # Large scale should have much larger std
        assert std_large > std_small * 5

    def test_mean_parameter(self):
        """Test that mean parameter shifts the distribution."""
        void_data = GaussianNoiseVoid(shape=(1000,), length=100, mean=5.0, scale=0.1)

        samples = torch.stack([void_data[i]["x"] for i in range(50)])
        mean = samples.mean().item()

        # Mean should be close to 5.0
        assert abs(mean - 5.0) < 0.5

    def test_multidimensional_shape(self):
        """Test with multi-dimensional shape (like images)."""
        void_data = GaussianNoiseVoid(shape=(3, 32, 32), length=50)

        sample = void_data[0]
        assert sample["x"].shape == (3, 32, 32)


class TestUniformNoiseVoid:
    """Tests for UniformNoiseVoid convenience class."""

    def test_basic_usage(self):
        """Test basic uniform noise generation."""
        void_data = UniformNoiseVoid(shape=(32,), length=100)

        assert len(void_data) == 100
        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (32,)

    def test_range_parameters(self):
        """Test that low/high parameters bound the noise."""
        void_data = UniformNoiseVoid(shape=(1000,), length=100, low=-1.0, high=1.0)

        samples = torch.stack([void_data[i]["x"] for i in range(50)])

        assert samples.min().item() >= -1.0
        assert samples.max().item() <= 1.0

    def test_custom_range(self):
        """Test with custom range."""
        void_data = UniformNoiseVoid(shape=(1000,), length=100, low=5.0, high=10.0)

        samples = torch.stack([void_data[i]["x"] for i in range(50)])

        assert samples.min().item() >= 5.0
        assert samples.max().item() <= 10.0


class TestFilteredNoiseVoid:
    """Tests for FilteredNoiseVoid (rejection sampling)."""

    def test_basic_usage(self):
        """Test basic filtered noise generation."""
        # Reference data: cluster around origin
        reference = torch.randn(50, 2) * 0.1

        void_data = FilteredNoiseVoid(
            reference_data=reference,
            shape=(2,),
            length=100,
            min_distance=0.3,
            noise_range=(-1.5, 1.5),
        )

        assert len(void_data) == 100
        sample = void_data[0]
        assert "x" in sample
        assert sample["x"].shape == (2,)

    def test_samples_are_far_from_reference(self):
        """Test that generated samples are far from reference data."""
        # Reference data: tight cluster
        reference = torch.zeros(10, 2)

        void_data = FilteredNoiseVoid(
            reference_data=reference,
            shape=(2,),
            length=50,
            min_distance=0.5,
            noise_range=(-2.0, 2.0),
        )

        # Check multiple samples
        for i in range(20):
            sample = void_data[i]["x"]
            dists = torch.norm(reference - sample.unsqueeze(0), dim=1)
            min_dist = torch.min(dists).item()
            # Should be at least min_distance away
            assert min_dist >= 0.5, f"Sample {i} is too close: {min_dist}"

    def test_respects_noise_range(self):
        """Test that noise is within specified range."""
        reference = torch.randn(10, 2)

        void_data = FilteredNoiseVoid(
            reference_data=reference,
            shape=(2,),
            length=100,
            min_distance=0.01,  # Small to avoid too many rejections
            noise_range=(-1.0, 1.0),
        )

        samples = torch.stack([void_data[i]["x"] for i in range(50)])
        assert samples.min().item() >= -1.0
        assert samples.max().item() <= 1.0


class TestVoidDatasetIntegration:
    """Integration tests for VoidDataset with trainer."""

    def test_compatible_with_trainer_sampling(self):
        """Test that VoidDataset works with trainer's _sample_void_data pattern."""
        void_data = GaussianNoiseVoid(shape=(16,), length=1000)

        # Simulate trainer's sampling logic
        batch_size = 32
        indices = torch.randint(0, len(void_data), (batch_size,))

        samples = []
        for idx in indices:
            sample = void_data[idx.item()]
            x = sample.get("x")
            samples.append(x)

        batch = torch.stack(samples)
        assert batch.shape == (32, 16)

    def test_mixed_dataset_scenario(self):
        """Test scenario: CIFAR-like images as OOD for MNIST-like task."""
        # Simulate CIFAR images (flattened to match MNIST dim)
        cifar_like = [{"x": torch.randn(784)} for _ in range(100)]
        void_data = VoidDataset(cifar_like)

        # Sample like trainer would
        batch_size = 8
        indices = torch.randint(0, len(void_data), (batch_size,))
        samples = torch.stack([void_data[idx.item()]["x"] for idx in indices])

        assert samples.shape == (8, 784)
