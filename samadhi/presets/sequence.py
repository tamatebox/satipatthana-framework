from typing import Dict, Any
import torch.nn as nn
from samadhi.core.builder import SamadhiBuilder
from samadhi.components.adapters.sequence import LstmAdapter, TransformerAdapter
from samadhi.components.decoders.sequence import LstmDecoder, SimpleSequenceDecoder
from samadhi.configs.main import SamadhiConfig


def create_lstm_samadhi(config: SamadhiConfig) -> nn.Module:
    """
    Creates an LSTM-based Samadhi model suitable for time-series/sequential data.
    Corresponds to the old LstmSamadhiModel.
    """
    adapter = LstmAdapter(config.adapter)
    decoder = LstmDecoder(config.decoder)

    engine = (
        SamadhiBuilder(config)
        .set_adapter(adapter)
        .set_vitakka()  # Default StandardVitakka
        .set_vicara(refiner_type="mlp")  # Default StandardVicara with MlpRefiner
        .set_decoder(decoder)
        .build()
    )

    return engine


def create_transformer_samadhi(config: SamadhiConfig) -> nn.Module:
    """
    Creates a Transformer-based Samadhi model suitable for sequential data.
    Corresponds to the old TransformerSamadhiModel.
    """
    adapter = TransformerAdapter(config.adapter)
    # Using SimpleSequenceDecoder (MLP) as default, matching old implementation
    decoder = SimpleSequenceDecoder(config.decoder)

    engine = (
        SamadhiBuilder(config)
        .set_adapter(adapter)
        .set_vitakka()  # Default StandardVitakka
        .set_vicara(refiner_type="mlp")  # Default StandardVicara with MlpRefiner
        .set_decoder(decoder)
        .build()
    )

    return engine
