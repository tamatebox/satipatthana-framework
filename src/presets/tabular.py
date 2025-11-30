from typing import Dict, Any
import torch.nn as nn
from src.core.builder import SamadhiBuilder
from src.components.adapters.mlp import MlpAdapter
from src.components.decoders.reconstruction import ReconstructionDecoder


def create_mlp_samadhi(config: Dict[str, Any]) -> nn.Module:
    """
    Creates a standard MLP-based Samadhi model suitable for tabular data.
    Corresponds to the old MlpSamadhiModel.
    """
    # Ensure config has necessary keys or set defaults
    # (Builder handles defaults mostly, but Adapter needs config)

    # Instantiate specific components
    adapter = MlpAdapter(config)

    # For now, using ReconstructionDecoder as default for tabular/unsupervised
    # If classification, user might need to change decoder or use a different preset
    decoder = ReconstructionDecoder(config)

    # Build engine
    engine = (
        SamadhiBuilder(config)
        .set_adapter(adapter)
        .set_vitakka()  # Default StandardVitakka
        .set_vicara(refiner_type="mlp")  # Default StandardVicara with MlpRefiner
        .set_decoder(decoder)
        .build()
    )

    return engine
