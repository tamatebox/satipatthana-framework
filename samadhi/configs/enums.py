from enum import Enum


class AdapterType(str, Enum):
    MLP = "mlp"
    CNN = "cnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    # Future extension: SIMPLE_SEQUENCE = "simple_sequence"


class VicaraType(str, Enum):
    STANDARD = "standard"
    WEIGHTED = "weighted"
    PROBE_SPECIFIC = "probe_specific"


class DecoderType(str, Enum):
    RECONSTRUCTION = "reconstruction"
    CNN = "cnn"
    LSTM = "lstm"
    SIMPLE_SEQUENCE = "simple_sequence"
    # CLASSIFICATION = "classification"
