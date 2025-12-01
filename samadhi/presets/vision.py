from typing import Dict, Any
import torch.nn as nn
from samadhi.core.builder import SamadhiBuilder
from samadhi.components.adapters.vision import CnnAdapter
from samadhi.components.decoders.vision import CnnDecoder
from samadhi.configs.main import SamadhiConfig


def create_conv_samadhi(config: SamadhiConfig) -> nn.Module:
    """
    Creates a standard Convolutional Samadhi model suitable for image data.
    Corresponds to the old ConvSamadhiModel.
    """
    # Instantiate specific components
    adapter = CnnAdapter(config.adapter)

    # Use CnnDecoder for reconstruction
    decoder = CnnDecoder(config.decoder)

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
