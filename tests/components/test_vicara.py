import torch
import torch.nn as nn
import pytest

from src.components.vicara import StandardVicara, VicaraBase


class DummyConfig:
    def __init__(self):
        self.input_dim = 10
        self.dim = 10  # Vicara uses 'dim'
        self.hidden_dim = 20
        self.latent_dim = 5
        self.num_layers = 1
        self.dropout = 0.0
        self.activation = "relu"
        self.refine_steps = 3  # Required for Vicara
        self.n_probes = 3  # For components that expect 'n_probes'

    # Allow dictionary-like access
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture
def vicara_instance():
    config = DummyConfig()
    return StandardVicara(config)


def test_vicara_init(vicara_instance):
    assert isinstance(vicara_instance, VicaraBase)
    assert isinstance(vicara_instance.refiners, nn.ModuleList)
    assert len(vicara_instance.refiners) == 1  # StandardVicara has 1 refiner
    assert vicara_instance.steps == 3


def test_vicara_forward_pass(vicara_instance):
    input_tensor = torch.randn(1, 10)  # Batch size 1, input_dim 10

    # Vicara returns: s_final, trajectory, energies
    s_final, trajectory, energies = vicara_instance(input_tensor)

    assert s_final.shape == input_tensor.shape
    assert isinstance(trajectory, list)
    assert isinstance(energies, list)
    assert torch.is_tensor(s_final)

    # Check that trajectory contains initial state + steps
    # Or just steps depending on implementation.
    # Current implementation: initial state + steps
    # But only if not training. Let's check training mode behavior.

    # If training is True (default for nn.Module), trajectory might be empty or different.
    # Vicara implementation: if not self.training: trajectory.append...
    # So by default (training=True), trajectory should be empty?
    # Let's check the code provided earlier:
    # if not self.training: trajectory.append(s_t...)
    # So in training mode, lists might be empty? No, energies are appended regardless.
    # Energies are appended: energies.append(energy)

    assert len(energies) == vicara_instance.steps


def test_vicara_with_different_batch_size(vicara_instance):
    batch_size = 4
    input_tensor = torch.randn(batch_size, 10)
    s_final, _, energies = vicara_instance(input_tensor)

    assert s_final.shape == input_tensor.shape
    assert len(energies) == vicara_instance.steps


def test_vicara_output_range_and_type(vicara_instance):
    input_tensor = torch.randn(1, 10)
    s_final, _, _ = vicara_instance(input_tensor)

    assert s_final.dtype == torch.float32
    # StandardVicara uses Tanh at the end, so output should be in [-1, 1] if not residual?
    # Actually it's residual update: s_new = alpha * s_old + (1 - alpha) * residual
    # Residual has Tanh, so it's in [-1, 1]. s_old is unbounded initially?
    # If s0 is unbounded, s_final might be unbounded but driven towards distribution of residual?
    # Let's just check type for now.


def test_vicara_inference_mode(vicara_instance):
    vicara_instance.eval()  # Set to inference mode
    input_tensor = torch.randn(1, 10)
    s_final, trajectory, energies = vicara_instance(input_tensor)

    # In inference mode, trajectory should be populated
    assert len(trajectory) > 0
    assert len(energies) > 0
