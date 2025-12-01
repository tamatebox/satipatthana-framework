import pytest
import torch
from samadhi.presets.vision import create_conv_samadhi
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.enums import AdapterType, VicaraType, DecoderType


@pytest.fixture
def vision_config() -> SamadhiConfig:
    config_data = {
        "dim": 32,
        "seed": 42,
        "adapter": {"type": AdapterType.CNN.value, "channels": 3, "img_size": 32},
        "vitakka": {"n_probes": 5, "gate_threshold": -1.0},
        "vicara": {"type": VicaraType.STANDARD.value, "refine_steps": 2},
        "decoder": {"type": DecoderType.CNN.value, "channels": 3, "img_size": 32, "input_dim": 32 * 32 * 3},
    }
    return SamadhiConfig.from_dict(config_data)


def test_create_conv_samadhi(vision_config: SamadhiConfig):
    model = create_conv_samadhi(vision_config)
    assert model is not None

    # Check forward pass
    # Input: (Batch, Channels, Height, Width)
    x = torch.randn(
        2,
        vision_config.adapter.channels,
        vision_config.adapter.img_size,
        vision_config.adapter.img_size,
    )
    output, s_final, meta = model(x)

    # Output should match input shape for reconstruction
    assert output.shape == x.shape
    assert s_final.shape == (2, vision_config.dim)
