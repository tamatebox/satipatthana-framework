"""
Tests for VipassanaObjective, GuidanceLoss, and StabilityLoss.
"""

import pytest
import torch

from satipatthana.components.objectives.vipassana import (
    VipassanaObjective,
    GuidanceLoss,
    StabilityLoss,
)
from satipatthana.core.santana import SantanaLog


class TestVipassanaObjective:
    """Tests for VipassanaObjective."""

    def test_initialization(self):
        """Test objective initialization."""
        obj = VipassanaObjective()
        assert obj.device is not None

    def test_compute_loss_basic(self):
        """Test basic BCE loss computation."""
        obj = VipassanaObjective()

        trust_scores = torch.tensor([[0.8], [0.2], [0.9], [0.1]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        loss, components = obj.compute_loss(trust_scores, targets)

        assert loss > 0
        assert "bce_loss" in components
        assert "accuracy" in components
        # With these values, accuracy should be 1.0
        assert components["accuracy"] == 1.0

    def test_compute_loss_mismatch(self):
        """Test loss with mismatched predictions."""
        obj = VipassanaObjective()

        # Predictions opposite to targets
        trust_scores = torch.tensor([[0.1], [0.9], [0.2], [0.8]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        loss, components = obj.compute_loss(trust_scores, targets)

        assert loss > 0
        # Accuracy should be 0 when predictions are opposite
        assert components["accuracy"] == 0.0

    def test_contrastive_loss(self):
        """Test contrastive loss between good and bad trajectories."""
        obj = VipassanaObjective()

        good_trust = torch.tensor([[0.9], [0.85], [0.95], [0.88]])
        bad_trust = torch.tensor([[0.1], [0.2], [0.05], [0.15]])

        loss, components = obj.compute_contrastive_loss(good_trust, bad_trust)

        assert loss > 0
        assert "margin_loss" in components
        assert "bce_loss" in components
        assert "trust_diff_mean" in components

        # Good should be higher than bad
        assert components["good_trust_mean"] > components["bad_trust_mean"]

    def test_contrastive_loss_margin_violation(self):
        """Test contrastive loss when margin is violated."""
        obj = VipassanaObjective()

        # Good and bad are too close
        good_trust = torch.tensor([[0.55], [0.52], [0.58], [0.51]])
        bad_trust = torch.tensor([[0.45], [0.48], [0.42], [0.49]])

        loss, components = obj.compute_contrastive_loss(good_trust, bad_trust, margin=0.5)

        # Margin loss should be positive since diff < margin
        assert components["margin_loss"] > 0


class TestGuidanceLoss:
    """Tests for GuidanceLoss."""

    def test_initialization_classification(self):
        """Test classification mode initialization."""
        loss_fn = GuidanceLoss(task_type="classification")
        assert loss_fn.task_type == "classification"

    def test_initialization_regression(self):
        """Test regression mode initialization."""
        loss_fn = GuidanceLoss(task_type="regression")
        assert loss_fn.task_type == "regression"

    def test_classification_loss(self):
        """Test classification loss computation."""
        loss_fn = GuidanceLoss(task_type="classification")

        aux_output = torch.tensor(
            [
                [2.0, 0.5, 0.1],
                [0.1, 2.0, 0.5],
                [0.5, 0.1, 2.0],
                [1.5, 0.5, 0.1],
            ]
        )
        targets = torch.tensor([0, 1, 2, 0])

        loss, components = loss_fn.compute_loss(aux_output, targets)

        assert loss > 0
        assert "guidance_loss" in components
        assert "guidance_accuracy" in components
        # All predictions should be correct
        assert components["guidance_accuracy"] == 1.0

    def test_classification_loss_wrong_predictions(self):
        """Test classification with wrong predictions."""
        loss_fn = GuidanceLoss(task_type="classification")

        aux_output = torch.tensor(
            [
                [0.1, 2.0, 0.5],  # Predicts 1, target 0 -> WRONG
                [0.1, 0.5, 2.0],  # Predicts 2, target 1 -> WRONG
                [2.0, 0.1, 0.5],  # Predicts 0, target 2 -> WRONG
                [0.5, 2.0, 0.1],  # Predicts 1, target 0 -> WRONG
            ]
        )
        targets = torch.tensor([0, 1, 2, 0])

        loss, components = loss_fn.compute_loss(aux_output, targets)

        assert loss > 0
        # All predictions are wrong
        assert components["guidance_accuracy"] == 0.0

    def test_regression_loss(self):
        """Test regression loss computation."""
        loss_fn = GuidanceLoss(task_type="regression")

        aux_output = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targets = torch.tensor([[1.1], [1.9], [3.2], [3.8]])

        loss, components = loss_fn.compute_loss(aux_output, targets)

        assert loss > 0
        assert "guidance_loss" in components
        assert "guidance_mse" in components

    def test_regression_loss_1d_target(self):
        """Test regression with 1D target."""
        loss_fn = GuidanceLoss(task_type="regression")

        aux_output = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss, components = loss_fn.compute_loss(aux_output, targets)

        # Perfect prediction should have ~0 loss
        assert loss < 0.01


class TestStabilityLoss:
    """Tests for StabilityLoss."""

    def test_initialization(self):
        """Test stability loss initialization."""
        loss_fn = StabilityLoss()
        assert loss_fn.device is not None

    def test_identical_states(self):
        """Test with identical states (zero loss)."""
        loss_fn = StabilityLoss()
        s_T = torch.randn(4, 32)
        s_T_1 = s_T.clone()  # Same state
        stability_pair = (s_T, s_T_1)

        loss, components = loss_fn.compute_loss(stability_pair)

        assert loss.item() == 0.0
        assert components["stability_loss"] == 0.0

    def test_with_state_difference(self):
        """Test with different states (non-zero loss)."""
        loss_fn = StabilityLoss()
        s_T = torch.randn(4, 32)
        s_T_1 = torch.randn(4, 32)  # Different state
        stability_pair = (s_T, s_T_1)

        loss, components = loss_fn.compute_loss(stability_pair)

        assert loss.item() > 0
        assert components["stability_loss"] > 0

    def test_high_vs_low_state_change(self):
        """Test that large state changes have higher loss."""
        loss_fn = StabilityLoss()

        # High state change
        s_T_high = torch.randn(4, 32)
        s_T_1_high = torch.randn(4, 32) * 10  # Large difference
        stability_pair_high = (s_T_high, s_T_1_high)

        # Low state change
        s_T_low = torch.randn(4, 32)
        s_T_1_low = s_T_low + torch.randn(4, 32) * 0.01  # Small difference
        stability_pair_low = (s_T_low, s_T_1_low)

        loss_high, _ = loss_fn.compute_loss(stability_pair_high)
        loss_low, _ = loss_fn.compute_loss(stability_pair_low)

        assert loss_high > loss_low

    def test_gradient_flow(self):
        """Test that gradients flow through stability loss."""
        loss_fn = StabilityLoss()
        s_T = torch.randn(4, 32, requires_grad=True)
        s_T_1 = torch.randn(4, 32, requires_grad=True)
        stability_pair = (s_T, s_T_1)

        loss, _ = loss_fn.compute_loss(stability_pair)
        loss.backward()

        assert s_T.grad is not None
        assert s_T_1.grad is not None

    def test_with_santana_logging(self):
        """Test that santana is used for logging only."""
        loss_fn = StabilityLoss()
        s_T = torch.randn(4, 32)
        s_T_1 = torch.randn(4, 32)
        stability_pair = (s_T, s_T_1)

        # Create santana with energies for logging
        santana = SantanaLog()
        for i in range(5):
            state = torch.randn(4, 32)
            energy = 1.0 / (i + 1)
            santana.add(state, energy=energy)

        loss, components = loss_fn.compute_loss(stability_pair, santana)

        # Loss should be based on stability_pair, not santana energies
        assert components["num_steps"] == 5
        assert components["total_energy"] > 0
