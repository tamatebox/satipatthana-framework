from typing import Dict, Any, Optional
import torch.nn as nn

from src.core.engine import SamadhiEngine
from src.components.vitakka.base import BaseVitakka
from src.components.vitakka.standard import StandardVitakka  # Assuming StandardVitakka as default for now
from src.components.vicara.base import BaseVicara
from src.components.vicara.standard import StandardVicara
from src.components.vicara.weighted import WeightedVicara
from src.components.vicara.probe_specific import ProbeVicara
from src.components.refiners.mlp import MlpRefiner  # For now, only MlpRefiner is available
from src.components.adapters.base import BaseAdapter  # To be used with Vitakka

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

    def set_decoder(self, decoder: nn.Module):
        self.decoder = decoder
        return self

    def build(self) -> SamadhiEngine:
        if not all([self.adapter, self.vitakka, self.vicara, self.decoder]):
            raise ValueError("All components (adapter, vitakka, vicara, decoder) must be set before building.")

        return SamadhiEngine(
            adapter=self.adapter, vitakka=self.vitakka, vicara=self.vicara, decoder=self.decoder, config=self.config
        )
