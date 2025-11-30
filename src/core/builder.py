from typing import Dict, Any
import torch.nn as nn

from src.core.engine import SamadhiEngine
from src.components.vitakka import Vitakka
from src.components.vicara import StandardVicara, WeightedVicara, ProbeVicara

# Note: These will be concrete implementations, but for now we rely on user passing instances
# or we will implement basic ones here/elsewhere.
# For the Builder to be fully functional, we need the specific Adapter/Decoder classes.
# Since we haven't implemented MlpAdapter etc. yet, the builder will be partial.


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

    def set_adapter(self, adapter: nn.Module):
        self.adapter = adapter
        return self

    def set_vitakka(self, vitakka: nn.Module = None):
        """
        Sets Vitakka. If None, builds default Vitakka from config.
        """
        if vitakka is not None:
            self.vitakka = vitakka
        else:
            self.vitakka = Vitakka(self.config)
        return self

    def set_vicara(self, vicara: nn.Module = None, vicara_type: str = "standard"):
        """
        Sets Vicara. If None, builds based on config or vicara_type.
        """
        if vicara is not None:
            self.vicara = vicara
        else:
            # Determine vicara_type: prioritize argument, then config, then default "standard"
            if vicara_type != "standard":  # If an explicit type was passed to the method
                v_type = vicara_type
            else:  # Otherwise, rely on config
                v_type = self.config.get("vicara_type", "standard")

            if v_type == "probe_specific":
                self.vicara = ProbeVicara(self.config)
            elif v_type == "weighted":
                self.vicara = WeightedVicara(self.config)
            else:
                self.vicara = StandardVicara(self.config)
        return self

    def set_decoder(self, decoder: nn.Module):
        self.decoder = decoder
        return self

    def build(self) -> SamadhiEngine:
        if not all([self.adapter, self.vitakka, self.vicara, self.decoder]):
            raise ValueError("All components (adapter, vitakka, vicara, decoder) must be set before building.")

        return SamadhiEngine(
            adapter=self.adapter, vitakka=self.vitakka, vicara=self.vicara, decoder=self.decoder, config=self.config
        )
