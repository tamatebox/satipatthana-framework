"""
VoidDataset: Unified interface for OOD data in Void Path training.

Supports both static datasets (images, embeddings) and dynamic generation (random noise).
"""

from typing import Any, Callable, Optional, Union
import torch
from torch.utils.data import Dataset


class VoidDataset(Dataset):
    """
    Unified Dataset for OOD (Out-of-Distribution) data in Stage 2 Void Path.

    Supports two modes:
    1. Static mode: Wraps an existing Dataset or list of tensors
    2. Dynamic mode: Generates data on-the-fly using a callable

    Args:
        source: Data source, one of:
            - Dataset: PyTorch Dataset (e.g., CIFAR-100, custom OOD dataset)
            - List/Tuple: Collection of tensors or dicts
            - Callable: Function that returns a single sample (called each __getitem__)
        length: Required for dynamic mode (callable source). Defines virtual dataset size.
        transform: Optional transform to apply to each sample.

    Examples:
        # Static mode with existing dataset
        cifar_ood = torchvision.datasets.CIFAR100(...)
        void_data = VoidDataset(cifar_ood)

        # Static mode with tensor list
        ood_tensors = [torch.randn(32) for _ in range(1000)]
        void_data = VoidDataset(ood_tensors)

        # Dynamic mode with random generator
        def random_noise():
            return {"x": torch.randn(32)}
        void_data = VoidDataset(random_noise, length=10000)

        # Dynamic mode with Gaussian noise at specific scale
        void_data = VoidDataset(
            lambda: {"x": torch.randn(784) * 2.0},
            length=5000
        )
    """

    def __init__(
        self,
        source: Union[Dataset, list, tuple, torch.Tensor, Callable[[], Any]],
        length: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.source = source
        self._length = length
        self.transform = transform

        # Determine if source is dynamic (callable without __getitem__)
        self.is_dynamic = callable(source) and not hasattr(source, "__getitem__")

        if self.is_dynamic and length is None:
            raise ValueError(
                "Dynamic source (callable) requires 'length' parameter. "
                "Example: VoidDataset(lambda: torch.randn(32), length=10000)"
            )

    def __len__(self) -> int:
        """Return dataset length."""
        if self.is_dynamic:
            return self._length
        if isinstance(self.source, torch.Tensor):
            return self.source.size(0)
        return len(self.source)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample by index.

        For dynamic sources, idx is ignored and a new sample is generated.
        For static sources, idx is wrapped around if out of bounds.

        Returns:
            dict with "x" key containing the tensor
        """
        if self.is_dynamic:
            # Dynamic: call the generator function
            sample = self.source()  # type: ignore[operator]
        else:
            # Static: index into the source (with wrap-around)
            actual_idx = idx % len(self)
            sample = self.source[actual_idx]

        # Normalize to dict format with "x" key
        sample = self._normalize_sample(sample)

        # Apply transform if provided
        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])

        return sample

    def _normalize_sample(self, sample: Any) -> dict:
        """
        Normalize sample to dict format with "x" key.

        Handles various input formats:
        - dict with "x", "input_values", or "pixel_values" key
        - tuple/list (assumes first element is data)
        - raw tensor
        """
        if isinstance(sample, dict):
            # Already a dict, ensure "x" key exists
            x = sample.get("x")
            if x is None:
                x = sample.get("input_values")
            if x is None:
                x = sample.get("pixel_values")
            if x is None:
                # Try first value if keys don't match
                x = next(iter(sample.values()), None)
            if x is None:
                raise ValueError("dict sample must contain 'x', 'input_values', or 'pixel_values' key")
            return {"x": x}

        elif isinstance(sample, (tuple, list)):
            # Assume first element is the data
            return {"x": sample[0]}

        elif isinstance(sample, torch.Tensor):
            # Raw tensor
            return {"x": sample}

        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}. Expected dict, tuple, list, or Tensor.")


class GaussianNoiseVoid(VoidDataset):
    """
    Convenience class for Gaussian noise OOD data.

    Generates random Gaussian noise with specified shape and scale.

    Args:
        shape: Shape of each noise tensor (excluding batch dimension)
        length: Virtual dataset size
        scale: Standard deviation of the noise (default: 1.0)
        mean: Mean of the noise (default: 0.0)

    Example:
        # 784-dim noise for MNIST-like input
        void_data = GaussianNoiseVoid(shape=(784,), length=10000, scale=1.0)

        # 3x32x32 noise for CIFAR-like input
        void_data = GaussianNoiseVoid(shape=(3, 32, 32), length=5000, scale=0.5)
    """

    def __init__(
        self,
        shape: tuple,
        length: int,
        scale: float = 1.0,
        mean: float = 0.0,
    ):
        self.shape = shape
        self.scale = scale
        self.mean = mean

        def generate_noise():
            noise = torch.randn(shape) * scale + mean
            return {"x": noise}

        super().__init__(source=generate_noise, length=length)


class FilteredNoiseVoid(VoidDataset):
    """
    Noise that is filtered to be far from training data.

    Generates random noise and rejects samples that are too close to
    any point in the reference data. This creates OOD samples that
    are guaranteed to be in "unknown territory".

    Args:
        reference_data: Training data to avoid (N, dim) tensor
        shape: Shape of each noise tensor (must match reference_data dim)
        length: Virtual dataset size
        min_distance: Minimum distance from any reference point (default: 0.1)
        noise_range: (low, high) tuple for uniform noise generation (default: (-1.5, 1.5))
        max_attempts: Maximum rejection sampling attempts (default: 1000)
        reference_sample_size: Maximum number of reference samples to use (default: 2000).
            If reference_data has more samples, a random subset is used for efficiency.

    Example:
        # Generate noise far from training data
        # No need to manually subset - automatic subsampling handles large datasets
        void_data = FilteredNoiseVoid(
            reference_data=train_data,  # Can be large, will be subsampled
            shape=(2,),
            length=1000,
            min_distance=0.1,
            noise_range=(-1.5, 1.5),
            reference_sample_size=2000,  # Use at most 2000 reference points
        )
    """

    def __init__(
        self,
        reference_data: torch.Tensor,
        shape: tuple,
        length: int,
        min_distance: float = 0.1,
        noise_range: tuple = (-1.5, 1.5),
        max_attempts: int = 1000,
        reference_sample_size: int = 2000,
    ):
        # Subsample reference data if too large for efficiency
        if reference_data.size(0) > reference_sample_size:
            indices = torch.randperm(reference_data.size(0))[:reference_sample_size]
            reference_data = reference_data[indices]

        self.reference_data = reference_data
        self.shape = shape
        self.min_distance = min_distance
        self.noise_low, self.noise_high = noise_range
        self.max_attempts = max_attempts

        def generate_filtered_noise():
            for _ in range(max_attempts):
                # Generate uniform noise in range
                noise = torch.rand(shape) * (self.noise_high - self.noise_low) + self.noise_low

                # Compute distance to all reference points
                dists = torch.norm(reference_data - noise.unsqueeze(0), dim=1)
                if torch.min(dists) > min_distance:
                    return {"x": noise}

            # Fallback: return noise anyway (should rarely happen)
            return {"x": torch.rand(shape) * (self.noise_high - self.noise_low) + self.noise_low}

        super().__init__(source=generate_filtered_noise, length=length)


class UniformNoiseVoid(VoidDataset):
    """
    Convenience class for uniform noise OOD data.

    Generates random uniform noise in specified range.

    Args:
        shape: Shape of each noise tensor (excluding batch dimension)
        length: Virtual dataset size
        low: Lower bound of uniform distribution (default: 0.0)
        high: Upper bound of uniform distribution (default: 1.0)

    Example:
        # Uniform noise in [0, 1] for normalized images
        void_data = UniformNoiseVoid(shape=(3, 32, 32), length=5000, low=0.0, high=1.0)
    """

    def __init__(
        self,
        shape: tuple,
        length: int,
        low: float = 0.0,
        high: float = 1.0,
    ):
        self.shape = shape
        self.low = low
        self.high = high

        def generate_noise():
            noise = torch.rand(shape) * (high - low) + low
            return {"x": noise}

        super().__init__(source=generate_noise, length=length)


__all__ = ["VoidDataset", "GaussianNoiseVoid", "UniformNoiseVoid", "FilteredNoiseVoid"]
