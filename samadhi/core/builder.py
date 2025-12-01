from typing import Dict, Any, Optional, Union
import torch.nn as nn
from dataclasses import replace

from samadhi.core.engine import SamadhiEngine
from samadhi.components.vitakka.base import BaseVitakka
from samadhi.components.vitakka.standard import StandardVitakka
from samadhi.components.vicara.base import BaseVicara
from samadhi.components.vicara.standard import StandardVicara
from samadhi.components.vicara.weighted import WeightedVicara
from samadhi.components.vicara.probe_specific import ProbeVicara
from samadhi.components.refiners.mlp import MlpRefiner

# Import concrete Adapters
from samadhi.components.adapters.mlp import MlpAdapter
from samadhi.components.adapters.vision import CnnAdapter
from samadhi.components.adapters.sequence import LstmAdapter, TransformerAdapter

# Import concrete Decoders
from samadhi.components.decoders.reconstruction import ReconstructionDecoder
from samadhi.components.decoders.vision import CnnDecoder
from samadhi.components.decoders.sequence import LstmDecoder, SimpleSequenceDecoder

from samadhi.configs.main import SamadhiConfig
from samadhi.configs.enums import AdapterType, VicaraType, DecoderType
from samadhi.configs.vicara import ProbeVicaraConfig, StandardVicaraConfig, WeightedVicaraConfig
from samadhi.configs.factory import create_vicara_config


class SamadhiBuilder:
    """
    Builder for SamadhiEngine.
    Allows constructing the model component by component.
    """

    def __init__(self, config: SamadhiConfig):
        if isinstance(config, dict):
            config = SamadhiConfig.from_dict(config)
        self.config = config

        self.adapter = None
        self.vitakka = None
        self.vicara = None
        self.decoder = None

    def set_adapter(self, adapter: nn.Module = None, type: str = None):
        """
        Sets the Adapter component.
        """
        if adapter is not None:
            self.adapter = adapter
            return self

        # Determine type
        if type is None:
            type_val = self.config.adapter.type
        else:
            type_val = type

        if hasattr(type_val, "value"):
            type_val = type_val.value

        if type_val == AdapterType.MLP.value:
            self.adapter = MlpAdapter(self.config.adapter)
        elif type_val == AdapterType.CNN.value:
            # Check if config is compatible, if not try to upgrade/convert or fail
            # For tests, we might override type via argument but config remains old.
            # In real usage, config should match.
            # Ideally, we should recreate config here if type override is provided.
            # But adapter config is immutable dataclass usually.
            # Let's rely on MlpAdapter/CnnAdapter __init__ handling config dictionary if passed,
            # BUT here we pass a Config object.
            # If type override mismatches config type, we have a problem.
            # We should probably re-create config if type is overridden.

            # Simple fix for tests: If type is overridden, we can't easily patch the Config object
            # if it's strictly typed and immutable.
            # However, for CnnAdapter, it accesses .channels which MlpAdapterConfig lacks.
            # We can re-create default config for that type if mismatch.
            cfg = self.config.adapter
            if type is not None and cfg.type != type_val:
                # Create default config for this type, copying common params if possible
                # This is tricky without a proper factory method that takes 'base params'.
                # For now, let's assume if type is passed, the user knows what they are doing
                # or we construct a minimal config dict and convert it.
                from samadhi.configs.factory import create_adapter_config

                cfg_dict = {"type": type_val, "dim": self.config.dim}
                # Copy potentially available fields
                if hasattr(cfg, "dropout"):
                    cfg_dict["dropout"] = cfg.dropout
                # For specific fields like channels/img_size, we can't guess them from MlpConfig.
                # They must rely on defaults or what's in the original config if it was a dict?
                # No, self.config is already a structured object.
                # If we are switching type dynamically, we better assume defaults for new type.
                cfg = create_adapter_config(cfg_dict)

            self.adapter = CnnAdapter(cfg)
        elif type_val == AdapterType.LSTM.value:
            cfg = self.config.adapter
            if type is not None and cfg.type != type_val:
                from samadhi.configs.factory import create_adapter_config

                cfg_dict = {"type": type_val, "dim": self.config.dim}
                cfg = create_adapter_config(cfg_dict)
            self.adapter = LstmAdapter(cfg)
        elif type_val == AdapterType.TRANSFORMER.value:
            cfg = self.config.adapter
            if type is not None and cfg.type != type_val:
                from samadhi.configs.factory import create_adapter_config

                cfg_dict = {"type": type_val, "dim": self.config.dim}
                cfg = create_adapter_config(cfg_dict)
            self.adapter = TransformerAdapter(cfg)
        else:
            raise ValueError(f"Unsupported adapter type: {type_val}")
        return self

    def set_vitakka(self, vitakka: BaseVitakka = None):
        """
        Sets Vitakka. If None, builds default StandardVitakka from config.
        """
        if vitakka is not None:
            self.vitakka = vitakka
        else:
            self.vitakka = StandardVitakka(self.config.vitakka)
        return self

    def set_vicara(
        self, vicara: BaseVicara = None, vicara_type: str = None, refiner_type: str = "mlp", n_probes: int = None
    ):
        """
        Sets Vicara. If None, builds based on config or vicara_type.
        Requires a refiner to be set or built.
        """

        if vicara is not None:
            self.vicara = vicara
            return self

        # Determine vicara_type: prioritize argument, then config
        if vicara_type is not None:
            v_type = vicara_type
        else:
            v_type = self.config.vicara.type

        if hasattr(v_type, "value"):
            v_type = v_type.value

        # Handle Config Mismatch for Vicara
        vicara_cfg = self.config.vicara
        if vicara_type is not None and vicara_cfg.type != v_type:
            cfg_dict = {"type": v_type, "dim": self.config.dim}
            if n_probes is not None:
                cfg_dict["n_probes"] = n_probes
            elif hasattr(self.config.vicara, "n_probes"):
                cfg_dict["n_probes"] = self.config.vicara.n_probes

            # Attempt to copy other fields if they exist
            if hasattr(vicara_cfg, "refine_steps"):
                cfg_dict["refine_steps"] = vicara_cfg.refine_steps
            if hasattr(vicara_cfg, "inertia"):
                cfg_dict["inertia"] = vicara_cfg.inertia

            vicara_cfg = create_vicara_config(cfg_dict)
        elif n_probes is not None and isinstance(vicara_cfg, ProbeVicaraConfig):
            # Override n_probes if passed explicitly
            # Since dataclass is frozen/managed, create new one or update if mutable
            # Ideally configs should be treated as immutable, but for builder pattern...
            # Let's create a new one to be safe
            vicara_cfg = replace(vicara_cfg, n_probes=n_probes)

        # Config for refiner: use vicara config as it has dim
        refiner_config = vicara_cfg

        if refiner_type == "mlp":
            # Create a single refiner for Standard/Weighted Vicara
            if v_type in [VicaraType.STANDARD.value, VicaraType.WEIGHTED.value]:
                refiner = MlpRefiner(refiner_config)
                refiners_list = nn.ModuleList([refiner])
            # Create multiple refiners for ProbeVicara
            elif v_type == VicaraType.PROBE_SPECIFIC.value:
                # Need n_probes
                n_probes_val = n_probes
                if n_probes_val is None:
                    if hasattr(vicara_cfg, "n_probes"):
                        n_probes_val = vicara_cfg.n_probes
                    else:
                        raise ValueError("ProbeVicara requires 'n_probes' in config or passed explicitly.")

                refiners_list = nn.ModuleList([MlpRefiner(refiner_config) for _ in range(n_probes_val)])
            else:
                raise ValueError(f"Unsupported vicara type for mlp refiner: {v_type}")
        else:
            raise ValueError(f"Unsupported refiner_type: {refiner_type}")

        if v_type == VicaraType.PROBE_SPECIFIC.value:
            self.vicara = ProbeVicara(vicara_cfg, refiners_list)
        elif v_type == VicaraType.WEIGHTED.value:
            self.vicara = WeightedVicara(vicara_cfg, refiners_list[0])
        else:
            self.vicara = StandardVicara(vicara_cfg, refiners_list[0])
        return self

    def set_decoder(self, decoder: nn.Module = None, type: str = None):
        """
        Sets the Decoder component.
        """
        if decoder is not None:
            self.decoder = decoder
            return self

        if type is None:
            type_val = self.config.decoder.type
        else:
            type_val = type

        if hasattr(type_val, "value"):
            type_val = type_val.value

        # Handle Config Mismatch for Decoder
        cfg = self.config.decoder
        if type is not None and cfg.type != type_val:
            from samadhi.configs.factory import create_decoder_config

            cfg_dict = {"type": type_val, "dim": self.config.dim}
            # Basic copy
            if hasattr(cfg, "input_dim"):
                cfg_dict["input_dim"] = cfg.input_dim
            if hasattr(cfg, "decoder_hidden_dim"):
                cfg_dict["decoder_hidden_dim"] = cfg.decoder_hidden_dim

            # Fallbacks for specific types
            if type_val == DecoderType.LSTM.value or type_val == DecoderType.SIMPLE_SEQUENCE.value:
                if hasattr(cfg, "input_dim"):
                    cfg_dict["output_dim"] = cfg.input_dim

            cfg = create_decoder_config(cfg_dict)

        if type_val == DecoderType.RECONSTRUCTION.value:
            self.decoder = ReconstructionDecoder(cfg)
        # Placeholder for ClassificationDecoder if implemented later
        # elif type_val == "classification":
        #     self.decoder = ClassificationDecoder(self.config.decoder)
        elif type_val == DecoderType.CNN.value:
            self.decoder = CnnDecoder(cfg)
        elif type_val == DecoderType.LSTM.value:
            self.decoder = LstmDecoder(cfg)
        elif type_val == DecoderType.SIMPLE_SEQUENCE.value:
            self.decoder = SimpleSequenceDecoder(cfg)
        else:
            raise ValueError(f"Unsupported decoder type: {type_val}")
        return self

    def build(self) -> SamadhiEngine:
        if not all([self.adapter, self.vitakka, self.vicara, self.decoder]):
            raise ValueError("All components (adapter, vitakka, vicara, decoder) must be set before building.")

        return SamadhiEngine(
            adapter=self.adapter, vitakka=self.vitakka, vicara=self.vicara, decoder=self.decoder, config=self.config
        )
