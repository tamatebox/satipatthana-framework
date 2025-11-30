from typing import Dict, Any
import torch.nn as nn
from src.core.builder import SamadhiBuilder
from src.components.adapters.vision import CnnAdapter
from src.components.decoders.vision import CnnDecoder


def create_conv_samadhi(config: Dict[str, Any]) -> nn.Module:
    """
    Creates a standard Convolutional Samadhi model suitable for image data.
    Corresponds to the old ConvSamadhiModel.
    """
    # Instantiate specific components
    adapter = CnnAdapter(config)

    # Use CnnDecoder for reconstruction
    decoder = CnnDecoder(config)

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
