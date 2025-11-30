from typing import Dict, Any
from src.configs.enums import AdapterType, VicaraType, DecoderType
from src.configs.adapters import (
    BaseAdapterConfig,
    MlpAdapterConfig,
    CnnAdapterConfig,
    LstmAdapterConfig,
    TransformerAdapterConfig,
)
from src.configs.vicara import BaseVicaraConfig, StandardVicaraConfig, WeightedVicaraConfig, ProbeVicaraConfig
from src.configs.vitakka import BaseVitakkaConfig, StandardVitakkaConfig
from src.configs.decoders import (
    BaseDecoderConfig,
    ReconstructionDecoderConfig,
    CnnDecoderConfig,
    LstmDecoderConfig,
    SimpleSequenceDecoderConfig,
)
from src.configs.objectives import ObjectiveConfig


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


def create_objective_config(data: Dict[str, Any]) -> ObjectiveConfig:
    """Creates an ObjectiveConfig."""
    return ObjectiveConfig.from_dict(data)
