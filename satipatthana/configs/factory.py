"""
Factory functions for creating Satipatthana system and components.

This module provides:
- create_system(): High-level factory for building SatipatthanaSystem
- create_*_config(): Low-level factories for individual component configs
"""

from typing import Dict, Any, Union, TYPE_CHECKING

from satipatthana.configs.enums import AdapterType, VicaraType, DecoderType
from satipatthana.configs.adapters import (
    BaseAdapterConfig,
    MlpAdapterConfig,
    CnnAdapterConfig,
    LstmAdapterConfig,
    TransformerAdapterConfig,
)
from satipatthana.configs.vicara import BaseVicaraConfig, StandardVicaraConfig, WeightedVicaraConfig, ProbeVicaraConfig
from satipatthana.configs.vitakka import BaseVitakkaConfig, StandardVitakkaConfig
from satipatthana.configs.decoders import (
    BaseDecoderConfig,
    ReconstructionDecoderConfig,
    CnnDecoderConfig,
    LstmDecoderConfig,
    SimpleSequenceDecoderConfig,
    ConditionalDecoderConfig,
    SimpleAuxHeadConfig,
)
from satipatthana.configs.system import SystemConfig, SamathaConfig, VipassanaEngineConfig

if TYPE_CHECKING:
    from satipatthana.core.system import SatipatthanaSystem
    from satipatthana.configs.config import SatipatthanaConfig


def _is_valid_type(type_val: Any, enum_cls: Any) -> bool:
    """Checks if type_val is a valid value in enum_cls."""
    if hasattr(type_val, "value"):
        type_val = type_val.value
    # Simple check against enum values
    return any(type_val == item.value for item in enum_cls)


def _clean_data_for_config(data: Dict[str, Any], type_val: Any) -> Dict[str, Any]:
    """Prepares data dict for config creation by ensuring 'type' matches the intended type."""
    data_copy = data.copy()
    if hasattr(type_val, "value"):
        data_copy["type"] = type_val.value
    else:
        data_copy["type"] = type_val
    return data_copy


def create_adapter_config(data: Dict[str, Any]) -> BaseAdapterConfig:
    """Creates a specific AdapterConfig based on the 'type' field."""
    type_val = data.get("type")

    if type_val is None or not _is_valid_type(type_val, AdapterType):
        type_val = AdapterType.MLP.value

    if hasattr(type_val, "value"):
        type_val = type_val.value

    # Clean data to ensure the config object gets the correct type
    clean_data = _clean_data_for_config(data, type_val)

    if type_val == AdapterType.MLP.value:
        return MlpAdapterConfig.from_dict(clean_data)
    elif type_val == AdapterType.CNN.value:
        return CnnAdapterConfig.from_dict(clean_data)
    elif type_val == AdapterType.LSTM.value:
        return LstmAdapterConfig.from_dict(clean_data)
    elif type_val == AdapterType.TRANSFORMER.value:
        return TransformerAdapterConfig.from_dict(clean_data)
    else:
        return MlpAdapterConfig.from_dict(clean_data)


def create_vicara_config(data: Dict[str, Any]) -> BaseVicaraConfig:
    """Creates a specific VicaraConfig based on the 'type' field."""
    type_val = data.get("vicara_type")
    if type_val is None:
        type_val = data.get("type")

    if type_val is not None and not _is_valid_type(type_val, VicaraType):
        type_val = VicaraType.STANDARD.value

    if type_val is None:
        type_val = VicaraType.STANDARD.value

    if hasattr(type_val, "value"):
        type_val = type_val.value

    clean_data = _clean_data_for_config(data, type_val)

    if type_val == VicaraType.STANDARD.value:
        return StandardVicaraConfig.from_dict(clean_data)
    elif type_val == VicaraType.WEIGHTED.value:
        return WeightedVicaraConfig.from_dict(clean_data)
    elif type_val == VicaraType.PROBE_SPECIFIC.value:
        return ProbeVicaraConfig.from_dict(clean_data)
    else:
        return StandardVicaraConfig.from_dict(clean_data)


def create_vitakka_config(data: Dict[str, Any]) -> BaseVitakkaConfig:
    """Creates a VitakkaConfig."""
    return StandardVitakkaConfig.from_dict(data)


def create_decoder_config(data: Dict[str, Any]) -> BaseDecoderConfig:
    """Creates a specific DecoderConfig based on the 'type' field."""
    type_val = data.get("type")

    if type_val is None or not _is_valid_type(type_val, DecoderType):
        type_val = DecoderType.RECONSTRUCTION.value

    if hasattr(type_val, "value"):
        type_val = type_val.value

    clean_data = _clean_data_for_config(data, type_val)

    if type_val == DecoderType.RECONSTRUCTION.value:
        # Fallback logic with clean data
        if "decoder_hidden_dim" not in clean_data and "adapter_hidden_dim" in clean_data:
            clean_data["decoder_hidden_dim"] = clean_data["adapter_hidden_dim"]
        return ReconstructionDecoderConfig.from_dict(clean_data)
    elif type_val == DecoderType.CNN.value:
        return CnnDecoderConfig.from_dict(clean_data)
    elif type_val == DecoderType.LSTM.value:
        if "output_dim" not in clean_data and "input_dim" in clean_data:
            clean_data["output_dim"] = clean_data["input_dim"]
        if "decoder_hidden_dim" not in clean_data and "adapter_hidden_dim" in clean_data:
            clean_data["decoder_hidden_dim"] = clean_data["adapter_hidden_dim"]
        return LstmDecoderConfig.from_dict(clean_data)
    elif type_val == DecoderType.SIMPLE_SEQUENCE.value:
        if "output_dim" not in clean_data and "input_dim" in clean_data:
            clean_data["output_dim"] = clean_data["input_dim"]
        if "decoder_hidden_dim" not in clean_data and "adapter_hidden_dim" in clean_data:
            clean_data["decoder_hidden_dim"] = clean_data["adapter_hidden_dim"]
        return SimpleSequenceDecoderConfig.from_dict(clean_data)
    else:
        if "decoder_hidden_dim" not in clean_data and "adapter_hidden_dim" in clean_data:
            clean_data["decoder_hidden_dim"] = clean_data["adapter_hidden_dim"]
        return ReconstructionDecoderConfig.from_dict(clean_data)


# =============================================================================
# High-Level Factory Functions
# =============================================================================


def create_system(
    config_or_type: Union["SatipatthanaConfig", str],
    **kwargs,
) -> "SatipatthanaSystem":
    """
    Create a SatipatthanaSystem from a config or preset type.

    This is the main entry point for building a complete system.

    Args:
        config_or_type: Either a SatipatthanaConfig instance or a string
                       specifying the adapter type ("mlp", "cnn", "lstm", "transformer")
        **kwargs: Additional arguments passed to SatipatthanaConfig when
                 config_or_type is a string

    Returns:
        Fully constructed SatipatthanaSystem instance

    Examples:
        # From preset string (simplest)
        system = create_system("mlp", input_dim=128, output_dim=10)

        # From config (customizable)
        config = SatipatthanaConfig(input_dim=128, output_dim=10, latent_dim=128)
        system = create_system(config)

        # With custom parameters
        system = create_system(
            "cnn",
            input_dim=784,
            output_dim=10,
            img_size=28,
            channels=1,
            latent_dim=128,
        )
    """
    from satipatthana.configs.config import SatipatthanaConfig

    # Convert string to config if needed
    if isinstance(config_or_type, str):
        config = SatipatthanaConfig(adapter=config_or_type, **kwargs)
    else:
        config = config_or_type

    # Build internal configs
    internal_config = _build_internal_config(config)

    # Build and return system
    return _build_system(config, internal_config)


def _build_internal_config(config: "SatipatthanaConfig") -> SystemConfig:
    """
    Convert user-facing SatipatthanaConfig to internal SystemConfig.

    This function handles the dimension propagation from the user config
    to all internal component configs.

    Args:
        config: User-facing SatipatthanaConfig

    Returns:
        Internal SystemConfig with all components properly configured
    """
    dim = config.latent_dim

    # Build component configs using helper methods
    adapter_cfg = config._build_adapter_config()
    vicara_cfg = config._build_vicara_config()
    sati_cfg = config._build_sati_config()
    vipassana_cfg = config._build_vipassana_config()
    augmenter_cfg = config._build_augmenter_config()

    # Get context_dim from vipassana config (auto-computed in StandardVipassanaConfig)
    ctx_dim = vipassana_cfg.context_dim

    # Build Samatha config
    samatha_cfg = SamathaConfig(
        dim=dim,
        max_steps=config.max_steps,
        adapter=adapter_cfg,
        augmenter=augmenter_cfg,
        vitakka=StandardVitakkaConfig(dim=dim, n_probes=config.n_probes),
        vicara=vicara_cfg,
        sati=sati_cfg,
    )

    # Build Vipassana engine config
    vipassana_engine_cfg = VipassanaEngineConfig(
        vipassana=vipassana_cfg,
    )

    # Build task decoder config
    task_decoder_cfg = ConditionalDecoderConfig(
        dim=dim,
        context_dim=ctx_dim,
        output_dim=config.output_dim,
    )

    # Build reconstruction head configs
    # Determine input_dim for reconstruction
    if config.adapter == "cnn":
        recon_input_dim = config.channels * config.img_size * config.img_size
    else:
        recon_input_dim = config.input_dim

    adapter_recon_cfg = ReconstructionDecoderConfig(
        dim=dim,
        input_dim=recon_input_dim,
    )

    samatha_recon_cfg = ReconstructionDecoderConfig(
        dim=dim,
        input_dim=recon_input_dim,
    )

    # Build auxiliary head config (for Stage 1 guidance)
    aux_head_cfg = SimpleAuxHeadConfig(
        dim=dim,
        output_dim=config.output_dim,
    )

    # Create SystemConfig
    return (
        SystemConfig(
            dim=dim,
            seed=config.seed,
            use_label_guidance=config.use_label_guidance,
            samatha=samatha_cfg,
            vipassana=vipassana_engine_cfg,
            task_decoder=task_decoder_cfg,
            adapter_recon_head=adapter_recon_cfg,
            samatha_recon_head=samatha_recon_cfg,
        ),
        aux_head_cfg,
    )


def _build_system(
    config: "SatipatthanaConfig",
    internal_config_tuple: tuple,
) -> "SatipatthanaSystem":
    """
    Build SatipatthanaSystem from internal configs.

    Args:
        config: User-facing SatipatthanaConfig (for reference)
        internal_config_tuple: Tuple of (SystemConfig, SimpleAuxHeadConfig)

    Returns:
        Fully constructed SatipatthanaSystem
    """
    import torch.nn as nn
    from satipatthana.core.system import SatipatthanaSystem
    from satipatthana.core.engines import SamathaEngine, VipassanaEngine

    # Component imports
    from satipatthana.components.adapters.mlp import MlpAdapter
    from satipatthana.components.adapters.vision import CnnAdapter
    from satipatthana.components.adapters.sequence import LstmAdapter, TransformerAdapter
    from satipatthana.components.augmenters.identity import IdentityAugmenter
    from satipatthana.components.augmenters.gaussian import GaussianNoiseAugmenter
    from satipatthana.components.vitakka.standard import StandardVitakka
    from satipatthana.components.vicara.standard import StandardVicara
    from satipatthana.components.vicara.weighted import WeightedVicara
    from satipatthana.components.vicara.probe_specific import ProbeVicara
    from satipatthana.components.refiners.mlp import MlpRefiner
    from satipatthana.components.sati.fixed_step import FixedStepSati
    from satipatthana.components.sati.threshold import ThresholdSati
    from satipatthana.components.vipassana.standard import StandardVipassana
    from satipatthana.components.decoders.conditional import ConditionalDecoder
    from satipatthana.components.decoders.reconstruction import ReconstructionDecoder
    from satipatthana.components.decoders.auxiliary import SimpleAuxHead

    system_config, aux_head_cfg = internal_config_tuple
    samatha_cfg = system_config.samatha
    vipassana_cfg = system_config.vipassana

    # Build Adapter
    adapter_cfg = samatha_cfg.adapter
    if isinstance(adapter_cfg, MlpAdapterConfig):
        adapter = MlpAdapter(adapter_cfg)
    elif isinstance(adapter_cfg, CnnAdapterConfig):
        adapter = CnnAdapter(adapter_cfg)
    elif isinstance(adapter_cfg, LstmAdapterConfig):
        adapter = LstmAdapter(adapter_cfg)
    elif isinstance(adapter_cfg, TransformerAdapterConfig):
        adapter = TransformerAdapter(adapter_cfg)
    else:
        raise ValueError(f"Unknown adapter config type: {type(adapter_cfg)}")

    # Build Augmenter
    augmenter_cfg = samatha_cfg.augmenter
    from satipatthana.configs.augmenter import GaussianNoiseAugmenterConfig

    if isinstance(augmenter_cfg, GaussianNoiseAugmenterConfig):
        augmenter = GaussianNoiseAugmenter(augmenter_cfg)
    else:
        augmenter = IdentityAugmenter(augmenter_cfg)

    # Build Vitakka
    vitakka = StandardVitakka(samatha_cfg.vitakka)

    # Build Refiner (needed by Vicara)
    refiner = MlpRefiner({"dim": system_config.dim})

    # Build Vicara
    vicara_cfg = samatha_cfg.vicara
    if isinstance(vicara_cfg, WeightedVicaraConfig):
        vicara = WeightedVicara(vicara_cfg, refiner)
    elif isinstance(vicara_cfg, ProbeVicaraConfig):
        # ProbeVicara needs ModuleList of refiners, one per probe
        probe_refiners = nn.ModuleList([MlpRefiner({"dim": system_config.dim}) for _ in range(vicara_cfg.n_probes)])
        vicara = ProbeVicara(vicara_cfg, probe_refiners)
    else:
        vicara = StandardVicara(vicara_cfg, refiner)

    # Build Sati
    sati_cfg = samatha_cfg.sati
    from satipatthana.configs.sati import ThresholdSatiConfig

    if isinstance(sati_cfg, ThresholdSatiConfig):
        sati = ThresholdSati(sati_cfg)
    else:
        sati = FixedStepSati(sati_cfg)

    # Build SamathaEngine
    samatha_engine = SamathaEngine(
        config=samatha_cfg,
        adapter=adapter,
        augmenter=augmenter,
        vitakka=vitakka,
        vicara=vicara,
        sati=sati,
    )

    # Build Vipassana
    vipassana_module = StandardVipassana(vipassana_cfg.vipassana)
    vipassana_engine = VipassanaEngine(
        config=vipassana_cfg,
        vipassana=vipassana_module,
    )

    # Build Decoders
    task_decoder = ConditionalDecoder(system_config.task_decoder)
    adapter_recon_head = ReconstructionDecoder(system_config.adapter_recon_head)
    samatha_recon_head = ReconstructionDecoder(system_config.samatha_recon_head)
    auxiliary_head = SimpleAuxHead(aux_head_cfg)

    # Build System
    return SatipatthanaSystem(
        config=system_config,
        samatha=samatha_engine,
        vipassana=vipassana_engine,
        task_decoder=task_decoder,
        adapter_recon_head=adapter_recon_head,
        samatha_recon_head=samatha_recon_head,
        auxiliary_head=auxiliary_head,
    )
