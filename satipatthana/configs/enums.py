from enum import Enum


class AdapterType(str, Enum):
    IDENTITY = "identity"
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
    CONDITIONAL = "conditional"
    SIMPLE_AUX_HEAD = "simple_aux_head"


class AugmenterType(str, Enum):
    IDENTITY = "identity"
    GAUSSIAN_NOISE = "gaussian_noise"


class SatiType(str, Enum):
    FIXED_STEP = "fixed_step"
    THRESHOLD = "threshold"


class VipassanaType(str, Enum):
    STANDARD = "standard"
    LSTM = "lstm"
