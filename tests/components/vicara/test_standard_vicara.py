import pytest
import torch
import torch.nn as nn
from samadhi.components.vicara.standard import StandardVicara
from samadhi.components.refiners.base import BaseRefiner


class MockRefiner(BaseRefiner):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config["dim"], config["dim"])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


@pytest.fixture
def config():
    return {
        "dim": 32,
        "refine_steps": 3,
        "inertia": 0.5,
        "gate_threshold": -1.0,  # Not used in Vicara but common in config
    }


def test_standard_vicara_initialization(config):
    refiner = MockRefiner(config)
    vicara = StandardVicara(config, refiner)
    assert isinstance(vicara, StandardVicara)
    assert len(vicara.refiners) == 1
    assert vicara.refiners[0] == refiner


def test_standard_vicara_forward(config):
    refiner = MockRefiner(config)
    vicara = StandardVicara(config, refiner)

    batch_size = 4
    s0 = torch.randn(batch_size, config["dim"])

    # Run forward
    s_final, trajectory, energies = vicara(s0)

    assert s_final.shape == (batch_size, config["dim"])
    # Trajectory length: initial state + refine_steps (in eval mode)
    # In training mode (default for pytest unless model.eval() called), trajectory might be different logic?
    # Let's check logic: if not self.training: append.
    # By default module is in training mode.

    vicara.eval()
    s_final_eval, trajectory_eval, energies_eval = vicara(s0)

    # In eval: initial(1) + steps(3) = 4 states in trajectory
    assert len(trajectory_eval) == config["refine_steps"] + 1
    assert len(energies_eval) == config["refine_steps"]

    # Energies should be scalar values (float)
    assert isinstance(energies_eval[0], float)


def test_standard_vicara_training_mode(config):
    refiner = MockRefiner(config)
    vicara = StandardVicara(config, refiner)
    vicara.train()  # Ensure training mode

    s0 = torch.randn(2, config["dim"])
    s_final, trajectory, energies = vicara(s0)

    # In training mode, trajectory logging is skipped to save memory
    assert len(trajectory) == 0
    assert len(energies) == config["refine_steps"]
