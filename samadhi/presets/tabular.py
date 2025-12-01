from typing import Dict, Any
import torch.nn as nn
from samadhi.core.builder import SamadhiBuilder
from samadhi.components.adapters.mlp import MlpAdapter
from samadhi.components.decoders.reconstruction import ReconstructionDecoder
from samadhi.configs.main import SamadhiConfig


def create_mlp_samadhi(config: SamadhiConfig) -> nn.Module:
    """
    Creates a standard MLP-based Samadhi model suitable for tabular data.
    Corresponds to the old MlpSamadhiModel.
    """
    # Ensure config has necessary keys or set defaults
    # (Builder handles defaults mostly, but Adapter needs config)

    # Instantiate specific components
    adapter = MlpAdapter(config.adapter)

    # For now, using ReconstructionDecoder as default for tabular/unsupervised
    # If classification, user might need to change decoder or use a different preset
    decoder = ReconstructionDecoder(config.decoder)

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
