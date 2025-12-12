"""
User-facing SatipatthanaConfig for simplified system construction.

This module provides a high-level configuration interface that abstracts away
the complexity of internal configs (SystemConfig, SamathaConfig, etc.).

Usage:
    # Simple
    config = SatipatthanaConfig(input_dim=128, output_dim=10)
    system = config.build()

    # With customization
    config = SatipatthanaConfig(
        input_dim=128,
        output_dim=10,
        latent_dim=128,
        adapter="mlp",
        vicara="weighted",
    )
    system = config.build()
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union, TYPE_CHECKING

from satipatthana.configs.adapters import (
    BaseAdapterConfig,
    MlpAdapterConfig,
    CnnAdapterConfig,
    LstmAdapterConfig,
    TransformerAdapterConfig,
)
from satipatthana.configs.vicara import (
    BaseVicaraConfig,
    StandardVicaraConfig,
    WeightedVicaraConfig,
    ProbeVicaraConfig,
)
from satipatthana.configs.sati import (
    BaseSatiConfig,
    FixedStepSatiConfig,
    ThresholdSatiConfig,
)
from satipatthana.configs.vipassana import (
    BaseVipassanaConfig,
    StandardVipassanaConfig,
    LSTMVipassanaConfig,
)
from satipatthana.configs.augmenter import (
    BaseAugmenterConfig,
    IdentityAugmenterConfig,
    GaussianNoiseAugmenterConfig,
)

if TYPE_CHECKING:
    from satipatthana.core.system import SatipatthanaSystem

# Type aliases for Union types (Phase 2: will accept Config objects)
AdapterSpec = Union[
    Literal["mlp", "cnn", "lstm", "transformer"],
    BaseAdapterConfig,
]
VicaraSpec = Union[
    Literal["standard", "weighted", "probe"],
    BaseVicaraConfig,
]
SatiSpec = Union[
    Literal["fixed", "threshold"],
    BaseSatiConfig,
]
VipassanaSpec = Union[
    Literal["standard", "lstm"],
    BaseVipassanaConfig,
]
AugmenterSpec = Union[
    Literal["identity", "gaussian"],
    BaseAugmenterConfig,
]


@dataclass
class SatipatthanaConfig:
    """
    User-facing simplified configuration for Satipatthana system.

    This config provides a flat interface that automatically propagates
    shared parameters (like latent_dim) to all internal components.

    Args:
        input_dim: Input dimension (required)
        output_dim: Output dimension (required)
        latent_dim: Latent space dimension (propagated to all components)
        context_dim: Vipassana context vector dimension
        seed: Random seed for reproducibility

        adapter: Adapter type or custom config
        vicara: Vicara type or custom config
        sati: Sati type or custom config
        vipassana: Vipassana type or custom config
        augmenter: Augmenter type or custom config

        adapter_hidden_dim: Hidden dimension for MLP/LSTM adapter
        img_size: Image size (required for CNN adapter)
        channels: Number of channels (required for CNN adapter)
        seq_len: Sequence length (required for LSTM/Transformer adapter)

        n_probes: Number of Vitakka probes
        max_steps: Maximum Vicara loop iterations
        vicara_inertia: Vicara inertia (beta) for state update
        sati_threshold: Threshold for convergence detection

        gru_hidden_dim: GRU hidden dimension in Vipassana

        use_label_guidance: Enable label guidance in Stage 1
        task_type: Task type for loss function selection

    Example:
        >>> config = SatipatthanaConfig(input_dim=128, output_dim=10)
        >>> system = config.build()

        >>> # With CNN adapter
        >>> config = SatipatthanaConfig(
        ...     input_dim=784,  # Not used for CNN, but required
        ...     output_dim=10,
        ...     adapter="cnn",
        ...     img_size=28,
        ...     channels=1,
        ... )
    """

    # === Required ===
    input_dim: int
    output_dim: int

    # === Common (propagated to all components) ===
    latent_dim: int = 64
    seed: int = 42

    # === Components (string or Config object) ===
    adapter: AdapterSpec = "mlp"
    vicara: VicaraSpec = "standard"
    sati: SatiSpec = "threshold"
    vipassana: VipassanaSpec = "standard"
    augmenter: AugmenterSpec = "identity"

    # === Adapter-specific (used when adapter is string) ===
    adapter_hidden_dim: int = 256
    img_size: Optional[int] = None
    channels: Optional[int] = None
    seq_len: Optional[int] = None

    # === Samatha parameters ===
    n_probes: int = 10
    max_steps: int = 10
    vicara_inertia: float = 0.5
    sati_threshold: float = 1e-4

    # === Vipassana parameters ===
    gru_hidden_dim: int = 32

    # === Training Strategy ===
    use_label_guidance: bool = False
    task_type: Literal["classification", "regression", "anomaly"] = "classification"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration consistency."""
        # Adapter-specific validation (only when adapter is string)
        if isinstance(self.adapter, str):
            if self.adapter == "cnn":
                if self.img_size is None:
                    raise ValueError("CNN adapter requires img_size")
                if self.channels is None:
                    raise ValueError("CNN adapter requires channels")
            elif self.adapter in ("lstm", "transformer"):
                if self.seq_len is None:
                    raise ValueError(f"{self.adapter.upper()} adapter requires seq_len")

        # Dimension validation
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

    def build(self) -> "SatipatthanaSystem":
        """
        Build SatipatthanaSystem from this configuration.

        Returns:
            Fully constructed SatipatthanaSystem instance.
        """
        from satipatthana.configs.factory import create_system

        return create_system(self)

    # === Helper methods for building internal configs ===

    def _build_adapter_config(self) -> BaseAdapterConfig:
        """Build adapter config from specification."""
        if isinstance(self.adapter, BaseAdapterConfig):
            return self.adapter

        if self.adapter == "mlp":
            return MlpAdapterConfig(
                input_dim=self.input_dim,
                dim=self.latent_dim,
                adapter_hidden_dim=self.adapter_hidden_dim,
            )
        elif self.adapter == "cnn":
            return CnnAdapterConfig(
                img_size=self.img_size,
                channels=self.channels,
                dim=self.latent_dim,
            )
        elif self.adapter == "lstm":
            return LstmAdapterConfig(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                dim=self.latent_dim,
                adapter_hidden_dim=self.adapter_hidden_dim,
            )
        elif self.adapter == "transformer":
            return TransformerAdapterConfig(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                dim=self.latent_dim,
                adapter_hidden_dim=self.adapter_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown adapter type: {self.adapter}")

    def _build_vicara_config(self) -> BaseVicaraConfig:
        """Build vicara config from specification."""
        if isinstance(self.vicara, BaseVicaraConfig):
            return self.vicara

        if self.vicara == "standard":
            return StandardVicaraConfig(
                dim=self.latent_dim,
                inertia=self.vicara_inertia,
            )
        elif self.vicara == "weighted":
            return WeightedVicaraConfig(
                dim=self.latent_dim,
                inertia=self.vicara_inertia,
            )
        elif self.vicara == "probe":
            return ProbeVicaraConfig(
                dim=self.latent_dim,
                inertia=self.vicara_inertia,
                n_probes=self.n_probes,
            )
        else:
            raise ValueError(f"Unknown vicara type: {self.vicara}")

    def _build_sati_config(self) -> BaseSatiConfig:
        """Build sati config from specification."""
        if isinstance(self.sati, BaseSatiConfig):
            return self.sati

        if self.sati == "fixed":
            return FixedStepSatiConfig()
        elif self.sati == "threshold":
            return ThresholdSatiConfig(
                energy_threshold=self.sati_threshold,
            )
        else:
            raise ValueError(f"Unknown sati type: {self.sati}")

    def _build_vipassana_config(self) -> BaseVipassanaConfig:
        """Build vipassana config from specification."""
        if isinstance(self.vipassana, BaseVipassanaConfig):
            return self.vipassana

        if self.vipassana == "standard":
            return StandardVipassanaConfig(
                latent_dim=self.latent_dim,
                gru_hidden_dim=self.gru_hidden_dim,
                max_steps=self.max_steps,
            )
        elif self.vipassana == "lstm":
            # For LSTM vipassana, use gru_hidden_dim * 2 as context_dim
            return LSTMVipassanaConfig(
                context_dim=self.gru_hidden_dim * 2,
            )
        else:
            raise ValueError(f"Unknown vipassana type: {self.vipassana}")

    def _build_augmenter_config(self) -> BaseAugmenterConfig:
        """Build augmenter config from specification."""
        if isinstance(self.augmenter, BaseAugmenterConfig):
            return self.augmenter

        if self.augmenter == "identity":
            return IdentityAugmenterConfig()
        elif self.augmenter == "gaussian":
            return GaussianNoiseAugmenterConfig()
        else:
            raise ValueError(f"Unknown augmenter type: {self.augmenter}")


__all__ = [
    "SatipatthanaConfig",
    "AdapterSpec",
    "VicaraSpec",
    "SatiSpec",
    "VipassanaSpec",
    "AugmenterSpec",
]
