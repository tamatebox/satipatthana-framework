from typing import Dict, Any, Optional
import torch.nn as nn

from src.core.engine import SamadhiEngine
from src.components.vitakka.base import BaseVitakka
from src.components.vitakka.standard import StandardVitakka
from src.components.vicara.base import BaseVicara
from src.components.vicara.standard import StandardVicara
from src.components.vicara.weighted import WeightedVicara
from src.components.vicara.probe_specific import ProbeVicara
from src.components.refiners.mlp import MlpRefiner

# Import concrete Adapters
from src.components.adapters.mlp import MlpAdapter
from src.components.adapters.vision import CnnAdapter
from src.components.adapters.sequence import LstmAdapter, TransformerAdapter

# Import concrete Decoders
from src.components.decoders.reconstruction import ReconstructionDecoder
from src.components.decoders.vision import CnnDecoder
from src.components.decoders.sequence import LstmDecoder, SimpleSequenceDecoder


class SamadhiBuilder:
    """
    Builder for SamadhiEngine.
    Allows constructing the model component by component.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapter = None
        self.vitakka = None
        self.vicara = None
        self.decoder = None

    def set_adapter(self, adapter: nn.Module = None, type: str = None):
        """
        Sets the Adapter component.
        Args:
            adapter: An instantiated Adapter module.
            type: A string specifying the adapter type to build ('mlp', 'cnn', 'lstm', 'transformer').
                  Used if 'adapter' is None.
        """
        if adapter is not None:
            self.adapter = adapter
        elif type is not None:
            if type == "mlp":
                self.adapter = MlpAdapter(self.config)
            elif type == "cnn":
                self.adapter = CnnAdapter(self.config)
            elif type == "lstm":
                self.adapter = LstmAdapter(self.config)
            elif type == "transformer":
                self.adapter = TransformerAdapter(self.config)
            else:
                raise ValueError(f"Unsupported adapter type: {type}")
        return self

    def set_vitakka(self, vitakka: BaseVitakka = None):
        """
        Sets Vitakka. If None, builds default StandardVitakka from config.
        """
        if vitakka is not None:
            self.vitakka = vitakka
        else:
            self.vitakka = StandardVitakka(self.config)
        return self

    def set_vicara(
        self, vicara: BaseVicara = None, vicara_type: str = "standard", refiner_type: str = "mlp", n_probes: int = None
    ):
        """
        Sets Vicara. If None, builds based on config or vicara_type.
        Requires a refiner to be set or built.
        """

        if vicara is not None:
            self.vicara = vicara
        else:
            # Determine vicara_type: prioritize argument, then config, then default "standard"
            if vicara_type != "standard":  # If an explicit type was passed to the method
                v_type = vicara_type
            else:  # Otherwise, rely on config
                v_type = self.config.get("vicara_type", "standard")

            # For now, only MlpRefiner is available
            if refiner_type == "mlp":
                # Create a single refiner for Standard/Weighted Vicara
                if v_type in ["standard", "weighted"]:
                    refiner = MlpRefiner(self.config)
                    refiners_list = nn.ModuleList([refiner])
                # Create multiple refiners for ProbeVicara
                elif v_type == "probe_specific":
                    n_probes_val = n_probes if n_probes is not None else self.config.get("n_probes")
                    if n_probes_val is None:
                        raise ValueError("ProbeVicara requires 'n_probes' in config or passed explicitly.")
                    refiners_list = nn.ModuleList([MlpRefiner(self.config) for _ in range(n_probes_val)])
                else:
                    raise ValueError(f"Unsupported refiner_type: {refiner_type}")
            else:
                raise ValueError(f"Unsupported refiner_type: {refiner_type}")

            if v_type == "probe_specific":
                self.vicara = ProbeVicara(self.config, refiners_list)
            elif v_type == "weighted":
                self.vicara = WeightedVicara(
                    self.config, refiners_list[0]
                )  # WeightedVicara expects a single refiner for now
            else:
                self.vicara = StandardVicara(self.config, refiners_list[0])  # StandardVicara expects a single refiner
        return self

    def set_decoder(self, decoder: nn.Module = None, type: str = None):
        """
        Sets the Decoder component.
        Args:
            decoder: An instantiated Decoder module.
            type: A string specifying the decoder type to build ('reconstruction', 'classification', 'cnn', 'lstm', 'simple_sequence').
                  Used if 'decoder' is None.
        """
        if decoder is not None:
            self.decoder = decoder
        elif type is not None:
            if type == "reconstruction":
                self.decoder = ReconstructionDecoder(self.config)
            # Placeholder for ClassificationDecoder if implemented later
            # elif type == "classification":
            #     self.decoder = ClassificationDecoder(self.config)
            elif type == "cnn":
                self.decoder = CnnDecoder(self.config)
            elif type == "lstm":
                self.decoder = LstmDecoder(self.config)
            elif type == "simple_sequence":
                self.decoder = SimpleSequenceDecoder(self.config)
            else:
                raise ValueError(f"Unsupported decoder type: {type}")
        return self

    def build(self) -> SamadhiEngine:
        if not all([self.adapter, self.vitakka, self.vicara, self.decoder]):
            raise ValueError("All components (adapter, vitakka, vicara, decoder) must be set before building.")

        return SamadhiEngine(
            adapter=self.adapter, vitakka=self.vitakka, vicara=self.vicara, decoder=self.decoder, config=self.config
        )
