import pytest
from samadhi.configs.adapters import CnnAdapterConfig, BaseAdapterConfig
from samadhi.configs.vicara import BaseVicaraConfig


def test_adapter_dropout_validation():
    # Valid
    BaseAdapterConfig(dropout=0.5)

    # Invalid
    with pytest.raises(ValueError, match="Dropout"):
        BaseAdapterConfig(dropout=1.5)
    with pytest.raises(ValueError, match="Dropout"):
        BaseAdapterConfig(dropout=-0.1)


def test_cnn_adapter_img_size_validation():
    # Valid
    CnnAdapterConfig(img_size=32, channels=3)

    # Invalid
    with pytest.raises(ValueError, match="img_size"):
        CnnAdapterConfig(img_size=0, channels=3)
    with pytest.raises(ValueError, match="img_size"):
        CnnAdapterConfig(img_size=-10, channels=3)


def test_vicara_inertia_validation():
    # Valid
    BaseVicaraConfig(inertia=0.7)

    # Invalid
    with pytest.raises(ValueError, match="Inertia"):
        BaseVicaraConfig(inertia=1.1)
