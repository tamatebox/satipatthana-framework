import pytest
import torch
from satipatthana.components.refiners import MlpRefiner


def test_mlp_refiner_initialization():
    config = {"dim": 32}
    refiner = MlpRefiner(config)
    assert isinstance(refiner, MlpRefiner)
    assert refiner.dim == 32
    assert len(refiner.net) == 5  # 2 Linear, 1 LayerNorm, 1 ReLU, 1 Tanh


def test_mlp_refiner_forward_pass():
    config = {"dim": 32}
    refiner = MlpRefiner(config)
    batch_size = 4
    input_state = torch.randn(batch_size, config["dim"])
    output = refiner(input_state)
    assert output.shape == (batch_size, config["dim"])
    assert torch.all((output >= -1.0) & (output <= 1.0))
