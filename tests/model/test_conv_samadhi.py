import torch
import pytest

from src.model.conv_samadhi import ConvSamadhiModel
from src.components.vitakka import Vitakka
from src.components.vicara import VicaraBase


class MockConfig:
    def __init__(self):
        self.input_channels = 1
        self.channels = 1  # ConvSamadhiModel expects 'channels'
        self.input_height = 32
        self.input_width = 32
        self.img_size = 32  # ConvSamadhiModel expects 'img_size' multiple of 16
        self.hidden_channels = 16
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.latent_dim = 5
        self.dim = 10  # ConvSamadhiModel needs dim for latent/samadhi space
        self.num_probes = 3
        self.n_probes = 3  # Added for components that expect 'n_probes'
        self.num_timesteps = 5
        self.gate_threshold = -1.0  # Always open for basic forward pass tests
        self.attention_mode = "hard"
        self.probe_trainable = True
        self.vicara_type = "shared"
        self.refine_steps = 3
        # For ConvSamadhi, input_dim in SamadhiCore is not directly used for conv layers,
        # but for flattening/linear layers if any. Let's make it consistent with flattened dim.
        self.input_dim = self.input_channels * self.input_height * self.input_width

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def conv_samadhi_instance():
    config = MockConfig()
    return ConvSamadhiModel(config)


def test_conv_samadhi_init(conv_samadhi_instance):
    assert isinstance(conv_samadhi_instance.vitakka, Vitakka)
    assert isinstance(conv_samadhi_instance.vicara, VicaraBase)
    assert conv_samadhi_instance.channels == 1
    assert conv_samadhi_instance.config["num_probes"] == 3


def test_conv_samadhi_forward_step(conv_samadhi_instance):
    # Set to eval mode to avoid BatchNorm error with batch_size=1
    conv_samadhi_instance.eval()

    # Input for ConvSamadhi should be (batch_size, channels, height, width)
    input_tensor = torch.randn(
        1,
        conv_samadhi_instance.config["channels"],
        conv_samadhi_instance.config["img_size"],
        conv_samadhi_instance.config["img_size"],
    )

    result = conv_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is not None
    s_final, full_log = result

    # s_final is latent state (Dim)
    assert s_final.shape == (1, conv_samadhi_instance.dim)

    # Test Decoder
    decoded_output = conv_samadhi_instance.decoder(s_final)
    assert decoded_output.shape == input_tensor.shape

    assert isinstance(full_log, dict)
    assert "probe_log" in full_log


def test_conv_samadhi_gating(conv_samadhi_instance):
    # Set to eval mode
    conv_samadhi_instance.eval()
    conv_samadhi_instance.config.gate_threshold = 100.0

    input_tensor = torch.randn(
        1,
        conv_samadhi_instance.config["channels"],
        conv_samadhi_instance.config["img_size"],
        conv_samadhi_instance.config["img_size"],
    )
    result = conv_samadhi_instance.forward_step(input_tensor, step_idx=0)

    assert result is None

    conv_samadhi_instance.config.gate_threshold = -1.0
