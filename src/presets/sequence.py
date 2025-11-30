from typing import Dict, Any
import torch.nn as nn
from src.core.builder import SamadhiBuilder
from src.components.adapters.sequence import LstmAdapter, TransformerAdapter
from src.components.decoders.sequence import LstmDecoder, SimpleSequenceDecoder


def create_lstm_samadhi(config: Dict[str, Any]) -> nn.Module:
    """
    Creates an LSTM-based Samadhi model suitable for time-series/sequential data.
    Corresponds to the old LstmSamadhiModel.
    """
    adapter = LstmAdapter(config)
    decoder = LstmDecoder(config)

    engine = (
        SamadhiBuilder(config)
        .set_adapter(adapter)
        .set_vitakka()  # Default StandardVitakka
        .set_vicara(refiner_type="mlp")  # Default StandardVicara with MlpRefiner
        .set_decoder(decoder)
        .build()
    )

    return engine


def create_transformer_samadhi(config: Dict[str, Any]) -> nn.Module:
    """
    Creates a Transformer-based Samadhi model suitable for sequential data.
    Corresponds to the old TransformerSamadhiModel.
    """
    adapter = TransformerAdapter(config)
    # Using SimpleSequenceDecoder (MLP) as default, matching old implementation
    decoder = SimpleSequenceDecoder(config)

    engine = (
        SamadhiBuilder(config)
        .set_adapter(adapter)
        .set_vitakka()  # Default StandardVitakka
        .set_vicara(refiner_type="mlp")  # Default StandardVicara with MlpRefiner
        .set_decoder(decoder)
        .build()
    )

    return engine
