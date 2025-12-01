from typing import Dict, Any, Optional, Union, Tuple, List
import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments
from samadhi.train.hf_trainer import SamadhiTrainer
from samadhi.train.objectives.unsupervised import UnsupervisedObjective
from samadhi.configs.main import SamadhiConfig  # Import SamadhiConfig

# --- Mocks ---


class MockVicara(nn.Module):
    def refine_step(self, s, meta):
        return torch.zeros_like(s)

    def update_state(self, s, res):
        return s


class MockModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vicara = MockVicara()
        self.adapter = nn.Linear(10, 10)
        self.decoder_layer = nn.Linear(10, 10)

    def forward(
        self, x: torch.Tensor, run_vitakka: bool = True, run_vicara: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        z = self.adapter(x)
        s: torch.Tensor
        meta: Dict[str, Any] = {}

        if run_vitakka:
            s, meta = self.vitakka_search(x)  # Calls the mock vitakka_search
        else:
            s = z
            meta = {
                "gate_open": True,
                "winner_id": -1,
                "confidence": 1.0,
                "raw_score": 1.0,
                "probs": torch.softmax(torch.randn(x.size(0), self.config.vitakka.n_probes), dim=1),
            }

        s_final: torch.Tensor
        if run_vicara and run_vitakka:  # Simplified, assuming vicara uses vitakka output if run_vitakka
            # In mock, vicara does not actually use s0 or meta in refine_step so we just pass s.
            # For a real test, this would involve a loop with model.vicara.refine_step and update_state
            s_final = s  # In mock, just pass through
        elif run_vicara and not run_vitakka:
            s_final = s  # Vicara runs on adapter output
        else:
            s_final = s  # Vicara skipped

        output = self.decoder(s_final)

        return output, s_final, meta

    def vitakka_search(self, x):
        batch_size = x.size(0)
        s0 = self.adapter(x)
        probs = torch.softmax(torch.randn(batch_size, self.config.vitakka.n_probes), dim=1)
        return s0, {"probs": probs}

    def decoder(self, s):
        return self.decoder_layer(s)


# --- Tests ---


@pytest.fixture
def mock_config():
    # Use SamadhiConfig
    config_dict = {
        "dim": 10,
        "adapter": {"input_dim": 10},
        "decoder": {"input_dim": 10},
        "objective": {
            "stability_coeff": 0.1,
            "entropy_coeff": 0.1,
            "balance_coeff": 0.1,
        },
        "vitakka": {"n_probes": 5},
        "vicara": {"refine_steps": 2},
    }
    return SamadhiConfig.from_dict(config_dict)


def test_hf_trainer_compute_loss(mock_config, tmp_path):
    # Setup
    model = MockModel(mock_config)
    objective = UnsupervisedObjective(mock_config, device="cpu")

    args = TrainingArguments(output_dir=tmp_path, use_cpu=True)

    trainer = SamadhiTrainer(model=model, args=args, objective=objective)

    # Dummy inputs
    # HF Trainer usually gets a dict from collator
    inputs = {"x": torch.randn(4, 10)}

    # Test compute_loss
    loss = trainer.compute_loss(model, inputs)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_hf_trainer_compute_loss_with_return_outputs(mock_config, tmp_path):
    model = MockModel(mock_config)
    objective = UnsupervisedObjective(mock_config, device="cpu")
    args = TrainingArguments(output_dir=tmp_path, use_cpu=True)
    trainer = SamadhiTrainer(model=model, args=args, objective=objective)

    inputs = {"x": torch.randn(4, 10)}

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (4, 10)
