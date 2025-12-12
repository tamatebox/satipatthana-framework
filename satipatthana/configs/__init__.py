"""
Satipatthana Configuration Module.

This module provides:
- User-facing configs: SatipatthanaConfig, CurriculumConfig
- Factory functions: create_system()
- Internal configs: SystemConfig, SamathaConfig, etc.
"""

from satipatthana.configs.config import SatipatthanaConfig
from satipatthana.configs.curriculum import CurriculumConfig, StageConfig, NoisePathRatios
from satipatthana.configs.factory import create_system

from satipatthana.configs.system import SystemConfig, SamathaConfig, VipassanaEngineConfig
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
from satipatthana.configs.vitakka import BaseVitakkaConfig, StandardVitakkaConfig
from satipatthana.configs.sati import BaseSatiConfig, FixedStepSatiConfig, ThresholdSatiConfig
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
from satipatthana.configs.decoders import (
    BaseDecoderConfig,
    ReconstructionDecoderConfig,
    CnnDecoderConfig,
    LstmDecoderConfig,
    SimpleSequenceDecoderConfig,
    ConditionalDecoderConfig,
    SimpleAuxHeadConfig,
)
from satipatthana.configs.enums import AdapterType, VicaraType, DecoderType

__all__ = [
    # User-facing (recommended)
    "SatipatthanaConfig",
    "CurriculumConfig",
    "StageConfig",
    "NoisePathRatios",
    "create_system",
    # Internal configs
    "SystemConfig",
    "SamathaConfig",
    "VipassanaEngineConfig",
    # Adapters
    "BaseAdapterConfig",
    "MlpAdapterConfig",
    "CnnAdapterConfig",
    "LstmAdapterConfig",
    "TransformerAdapterConfig",
    # Vicara
    "BaseVicaraConfig",
    "StandardVicaraConfig",
    "WeightedVicaraConfig",
    "ProbeVicaraConfig",
    # Vitakka
    "BaseVitakkaConfig",
    "StandardVitakkaConfig",
    # Sati
    "BaseSatiConfig",
    "FixedStepSatiConfig",
    "ThresholdSatiConfig",
    # Vipassana
    "BaseVipassanaConfig",
    "StandardVipassanaConfig",
    "LSTMVipassanaConfig",
    # Augmenter
    "BaseAugmenterConfig",
    "IdentityAugmenterConfig",
    "GaussianNoiseAugmenterConfig",
    # Decoders
    "BaseDecoderConfig",
    "ReconstructionDecoderConfig",
    "CnnDecoderConfig",
    "LstmDecoderConfig",
    "SimpleSequenceDecoderConfig",
    "ConditionalDecoderConfig",
    "SimpleAuxHeadConfig",
    # Enums
    "AdapterType",
    "VicaraType",
    "DecoderType",
]
