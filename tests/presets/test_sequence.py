import pytest
import torch
from samadhi.presets.sequence import create_lstm_samadhi, create_transformer_samadhi
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.enums import AdapterType, VicaraType, DecoderType


@pytest.fixture
def sequence_config() -> SamadhiConfig:
    config_data = {
        "dim": 32,
        "seed": 42,
        "adapter": {
            "type": AdapterType.LSTM.value,
            "input_dim": 10,
            "seq_len": 20,
            "adapter_hidden_dim": 64,
            "lstm_layers": 1,
        },
        "vitakka": {"n_probes": 5, "gate_threshold": -1.0},
        "vicara": {"type": VicaraType.STANDARD.value, "refine_steps": 2},
        "decoder": {
            "type": DecoderType.LSTM.value,
            "output_dim": 10,
            "seq_len": 20,
            "decoder_hidden_dim": 64,
            "lstm_layers": 1,
        },
    }
    return SamadhiConfig.from_dict(config_data)


@pytest.fixture
def transformer_sequence_config() -> SamadhiConfig:
    config_data = {
        "dim": 32,
        "seed": 42,
        "adapter": {
            "type": AdapterType.TRANSFORMER.value,
            "input_dim": 10,
            "seq_len": 20,
            "adapter_hidden_dim": 64,
            "transformer_layers": 1,
            "transformer_heads": 2,
        },
        "vitakka": {"n_probes": 5, "gate_threshold": -1.0},
        "vicara": {"type": VicaraType.STANDARD.value, "refine_steps": 2},
        "decoder": {
            "type": DecoderType.SIMPLE_SEQUENCE.value,
            "output_dim": 10,
            "seq_len": 20,
            "decoder_hidden_dim": 64,
        },
    }
    return SamadhiConfig.from_dict(config_data)


def test_create_lstm_samadhi(sequence_config: SamadhiConfig):
    model = create_lstm_samadhi(sequence_config)
    assert model is not None

    x = torch.randn(2, sequence_config.adapter.seq_len, sequence_config.adapter.input_dim)
    output, s_final, meta = model(x)

    assert output.shape == x.shape
    assert s_final.shape == (2, sequence_config.dim)


def test_create_transformer_samadhi(transformer_sequence_config: SamadhiConfig):
    model = create_transformer_samadhi(transformer_sequence_config)
    assert model is not None

    x = torch.randn(2, transformer_sequence_config.adapter.seq_len, transformer_sequence_config.adapter.input_dim)
    output, s_final, meta = model(x)

    assert output.shape == x.shape
    assert s_final.shape == (2, transformer_sequence_config.dim)
