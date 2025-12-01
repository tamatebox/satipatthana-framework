from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import inspect
from samadhi.configs.base import BaseConfig
from samadhi.configs.adapters import BaseAdapterConfig, MlpAdapterConfig
from samadhi.configs.vicara import BaseVicaraConfig, StandardVicaraConfig
from samadhi.configs.vitakka import BaseVitakkaConfig, StandardVitakkaConfig
from samadhi.configs.decoders import BaseDecoderConfig, ReconstructionDecoderConfig
from samadhi.configs.objectives import ObjectiveConfig
from samadhi.configs.factory import (
    create_adapter_config,
    create_vicara_config,
    create_vitakka_config,
    create_decoder_config,
    create_objective_config,
)


@dataclass
class SamadhiConfig(BaseConfig):
    """
    Root Configuration for Samadhi Model.
    """

    # --- Global Shared Params ---
    dim: int = 64
    seed: int = 42
    labels: list = field(default_factory=list)  # Probe labels for logging

    # --- Nested Component Configs ---
    # Defaulting to standard MLP/Reconstruction setup with dummy input_dim for initialization
    adapter: BaseAdapterConfig = field(default_factory=lambda: MlpAdapterConfig(input_dim=10))
    vitakka: BaseVitakkaConfig = field(default_factory=StandardVitakkaConfig)
    vicara: BaseVicaraConfig = field(default_factory=StandardVicaraConfig)
    decoder: BaseDecoderConfig = field(default_factory=lambda: ReconstructionDecoderConfig(input_dim=10))
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamadhiConfig":
        """
        Creates SamadhiConfig from a dictionary, handling nested component configs.
        If nested keys ('adapter', 'vicara', etc.) are present as dicts,
        they are converted using factories.
        If flattened keys are used (legacy style), they are passed to component factories
        (which filter relevant keys).
        """
        sig = inspect.signature(cls)
        config_args = {}

        # Extract global dim for propagation
        global_dim = data.get("dim")

        # 1. Handle global params
        for k in sig.parameters:
            if k in data:
                # If the field is one of the components and the value is a dict, use factory
                if k == "adapter" and isinstance(data[k], dict):
                    adapter_data = data[k].copy()
                    if global_dim is not None and "dim" not in adapter_data:
                        adapter_data["dim"] = global_dim
                    config_args[k] = create_adapter_config(adapter_data)

                elif k == "vitakka" and isinstance(data[k], dict):
                    vitakka_data = data[k].copy()
                    if global_dim is not None and "dim" not in vitakka_data:
                        vitakka_data["dim"] = global_dim
                    config_args[k] = create_vitakka_config(vitakka_data)

                elif k == "vicara" and isinstance(data[k], dict):
                    vicara_data = data[k].copy()
                    if global_dim is not None and "dim" not in vicara_data:
                        vicara_data["dim"] = global_dim
                    config_args[k] = create_vicara_config(vicara_data)

                elif k == "decoder" and isinstance(data[k], dict):
                    decoder_data = data[k].copy()
                    if global_dim is not None and "dim" not in decoder_data:
                        decoder_data["dim"] = global_dim
                    config_args[k] = create_decoder_config(decoder_data)

                elif k == "objective" and isinstance(data[k], dict):
                    config_args[k] = create_objective_config(data[k])

                else:
                    # Scalar values (dim, seed) or already Config objects
                    config_args[k] = data[k]

            # 2. If component keys are missing but data is flat (Legacy support),
            # try to construct components from the flat data.
            # This relies on the fact that factories filter keys.
            elif k == "adapter":
                config_args[k] = create_adapter_config(data)
            elif k == "vitakka":
                config_args[k] = create_vitakka_config(data)
            elif k == "vicara":
                config_args[k] = create_vicara_config(data)
            elif k == "decoder":
                config_args[k] = create_decoder_config(data)
            elif k == "objective":
                # For objective, we pass the whole flat data, factory will extract relevant keys
                config_args[k] = create_objective_config(data)

        return cls(**config_args)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the SamadhiConfig object to a dictionary.
        Recursively converts nested dataclass objects.
        """
        return asdict(self)
