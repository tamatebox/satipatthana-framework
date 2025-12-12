"""
System Configuration classes for Satipatthana Framework v4.0.

This module defines the top-level SystemConfig that orchestrates
all component configurations for the v4.0 architecture.
"""

from dataclasses import dataclass, field
from typing import Optional
from satipatthana.configs.base import BaseConfig
from satipatthana.configs.adapters import BaseAdapterConfig, MlpAdapterConfig
from satipatthana.configs.vicara import BaseVicaraConfig, StandardVicaraConfig
from satipatthana.configs.vitakka import BaseVitakkaConfig, StandardVitakkaConfig
from satipatthana.configs.decoders import BaseDecoderConfig, ReconstructionDecoderConfig
from satipatthana.configs.augmenter import BaseAugmenterConfig, IdentityAugmenterConfig
from satipatthana.configs.sati import BaseSatiConfig, FixedStepSatiConfig
from satipatthana.configs.vipassana import BaseVipassanaConfig, StandardVipassanaConfig


@dataclass
class SamathaConfig(BaseConfig):
    """
    Configuration for SamathaEngine.

    Groups all components that belong to the "thinking" phase:
    Adapter, Augmenter, Vitakka, Vicara, Sati.
    """

    dim: int = 64
    max_steps: int = 10  # Maximum Vicara loop iterations

    # Drunk mode parameters (for Stage 2 negative sampling)
    drunk_skip_prob: float = 0.3  # Probability to skip a refinement step
    drunk_perturbation_std: float = 0.2  # Std of random perturbation added to state

    # Component configs
    adapter: BaseAdapterConfig = field(default_factory=lambda: MlpAdapterConfig(input_dim=10))
    augmenter: BaseAugmenterConfig = field(default_factory=IdentityAugmenterConfig)
    vitakka: BaseVitakkaConfig = field(default_factory=StandardVitakkaConfig)
    vicara: BaseVicaraConfig = field(default_factory=StandardVicaraConfig)
    sati: BaseSatiConfig = field(default_factory=FixedStepSatiConfig)


@dataclass
class VipassanaEngineConfig(BaseConfig):
    """
    Configuration for VipassanaEngine.

    Groups components for the "introspection" phase.
    """

    vipassana: BaseVipassanaConfig = field(default_factory=StandardVipassanaConfig)


@dataclass
class SystemConfig(BaseConfig):
    """
    Root Configuration for Satipatthana v4.0 System.

    This is the top-level config that encompasses all engines and components.
    """

    # --- Global Params ---
    dim: int = 64
    seed: int = 42
    use_label_guidance: bool = False  # Enable label-guided training in Stage 1 & 2

    # --- Engine Configs ---
    samatha: SamathaConfig = field(default_factory=SamathaConfig)
    vipassana: VipassanaEngineConfig = field(default_factory=VipassanaEngineConfig)

    # --- Task Decoder Config ---
    task_decoder: BaseDecoderConfig = field(default_factory=lambda: ReconstructionDecoderConfig(input_dim=10))

    # --- Reconstruction Heads ---
    adapter_recon_head: Optional[BaseDecoderConfig] = None
    samatha_recon_head: Optional[BaseDecoderConfig] = None
