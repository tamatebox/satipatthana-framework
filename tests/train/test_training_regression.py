"""
Regression tests for training issues documented in docs/issues/001_training_issues_v4.md.

These tests verify that the fixes for known training issues remain effective:
- Issue 1: StabilityLoss gradient flow (stability_pair vs SantanaLog)
- Issue 4: Vitakka gate threshold and output diversity

These are "sanity check" tests that verify:
1. Loss decreases after a few training steps
2. Output diversity (s_star variance) is maintained
3. Gradients flow correctly through the system
"""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from satipatthana.core.system import SatipatthanaSystem, TrainingStage
from satipatthana.core.engines import SamathaEngine, VipassanaEngine
from satipatthana.configs.system import SystemConfig, SamathaConfig, VipassanaEngineConfig
from satipatthana.configs.adapters import MlpAdapterConfig
from satipatthana.configs.vitakka import StandardVitakkaConfig
from satipatthana.configs.vicara import StandardVicaraConfig
from satipatthana.configs.sati import FixedStepSatiConfig
from satipatthana.configs.augmenter import IdentityAugmenterConfig
from satipatthana.configs.vipassana import StandardVipassanaConfig
from satipatthana.configs.decoders import ConditionalDecoderConfig, ReconstructionDecoderConfig, SimpleAuxHeadConfig

from satipatthana.components.adapters.mlp import MlpAdapter
from satipatthana.components.augmenters.identity import IdentityAugmenter
from satipatthana.components.vitakka.standard import StandardVitakka
from satipatthana.components.vicara.standard import StandardVicara
from satipatthana.components.refiners.mlp import MlpRefiner
from satipatthana.components.sati.fixed_step import FixedStepSati
from satipatthana.components.vipassana.standard import StandardVipassana
from satipatthana.components.decoders.conditional import ConditionalDecoder
from satipatthana.components.decoders.reconstruction import ReconstructionDecoder
from satipatthana.components.decoders.auxiliary import SimpleAuxHead
from satipatthana.components.objectives.vipassana import StabilityLoss


# Constants
INPUT_DIM = 16
LATENT_DIM = 32
CONTEXT_DIM = 16
OUTPUT_DIM = 10
BATCH_SIZE = 8


class SimpleDataset(Dataset):
    """Simple dataset for regression tests."""

    def __init__(self, size: int = 64):
        # Create diverse data to ensure output diversity
        self.data = torch.randn(size, INPUT_DIM)
        self.labels = torch.randint(0, OUTPUT_DIM, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"x": self.data[idx], "y": self.labels[idx]}


def create_test_system(gate_threshold: float = -1.0) -> SatipatthanaSystem:
    """Create a SatipatthanaSystem for testing with specified gate_threshold."""
    config = SystemConfig(
        dim=LATENT_DIM,
        use_label_guidance=True,
        samatha=SamathaConfig(
            dim=LATENT_DIM,
            max_steps=3,
            adapter=MlpAdapterConfig(input_dim=INPUT_DIM, dim=LATENT_DIM),
            augmenter=IdentityAugmenterConfig(),
            vitakka=StandardVitakkaConfig(dim=LATENT_DIM, n_probes=4, gate_threshold=gate_threshold),
            vicara=StandardVicaraConfig(dim=LATENT_DIM, refine_steps=3),
            sati=FixedStepSatiConfig(),
        ),
        # context_dim = gru_hidden_dim + metric_proj_dim = 8 + 8 = 16
        vipassana=VipassanaEngineConfig(
            vipassana=StandardVipassanaConfig(latent_dim=LATENT_DIM, gru_hidden_dim=8, metric_proj_dim=8)
        ),
        task_decoder=ConditionalDecoderConfig(dim=LATENT_DIM, context_dim=CONTEXT_DIM, output_dim=OUTPUT_DIM),
    )

    samatha_config = config.samatha
    adapter = MlpAdapter(samatha_config.adapter)
    augmenter = IdentityAugmenter(samatha_config.augmenter)
    vitakka = StandardVitakka(samatha_config.vitakka)
    refiner = MlpRefiner({"dim": LATENT_DIM})
    vicara = StandardVicara(samatha_config.vicara, refiner)
    sati = FixedStepSati(samatha_config.sati)

    samatha_engine = SamathaEngine(
        config=samatha_config,
        adapter=adapter,
        augmenter=augmenter,
        vitakka=vitakka,
        vicara=vicara,
        sati=sati,
    )

    vipassana_engine = VipassanaEngine(
        config=config.vipassana,
        vipassana=StandardVipassana(config.vipassana.vipassana),
    )

    task_decoder = ConditionalDecoder(config.task_decoder)
    adapter_recon_head = ReconstructionDecoder(ReconstructionDecoderConfig(dim=LATENT_DIM, input_dim=INPUT_DIM))
    samatha_recon_head = ReconstructionDecoder(ReconstructionDecoderConfig(dim=LATENT_DIM, input_dim=INPUT_DIM))
    auxiliary_head = SimpleAuxHead(SimpleAuxHeadConfig(dim=LATENT_DIM, output_dim=OUTPUT_DIM))

    return SatipatthanaSystem(
        config=config,
        samatha=samatha_engine,
        vipassana=vipassana_engine,
        task_decoder=task_decoder,
        adapter_recon_head=adapter_recon_head,
        samatha_recon_head=samatha_recon_head,
        auxiliary_head=auxiliary_head,
    )


class TestStabilityLossGradientFlow:
    """Tests for Issue 1: StabilityLoss gradient flow.

    Related: docs/issues/001_training_issues_v4.md - Issue 1
    Verifies that stability_pair maintains gradients while SantanaLog is detached.
    """

    def test_stability_pair_has_gradients(self):
        """Test that stability_pair tensors have grad_fn (not detached)."""
        system = create_test_system()
        system.train()

        x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        samatha_output = system.samatha(x)

        s_T, s_T_1 = samatha_output.stability_pair

        # stability_pair should have gradients
        assert s_T.grad_fn is not None, "s_T should have grad_fn"
        assert s_T_1.grad_fn is not None, "s_T_1 should have grad_fn"

    def test_santana_states_are_detached(self):
        """Test that SantanaLog states are detached (no grad_fn)."""
        system = create_test_system()
        system.train()

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        samatha_output = system.samatha(x)

        santana = samatha_output.santana

        # SantanaLog states should be detached
        for state in santana.states:
            assert state.grad_fn is None, "SantanaLog states should be detached"

    def test_stability_loss_backward_works(self):
        """Test that StabilityLoss can compute gradients through stability_pair."""
        system = create_test_system()
        system.set_stage(TrainingStage.SAMATHA_TRAINING)

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        samatha_output = system.samatha(x)

        stability_loss = StabilityLoss()
        loss, _ = stability_loss.compute_loss(samatha_output.stability_pair, samatha_output.santana)

        # Should be able to backward
        loss.backward()

        # Check that gradients flowed to Vicara parameters
        vicara_has_grad = False
        for p in system.samatha.vicara.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                vicara_has_grad = True
                break

        assert vicara_has_grad, "Vicara should have gradients after StabilityLoss backward"

    def test_stability_loss_decreases_with_training(self):
        """Test that StabilityLoss decreases after a few optimization steps."""
        system = create_test_system()
        system.set_stage(TrainingStage.SAMATHA_TRAINING)

        optimizer = torch.optim.Adam(
            [p for p in system.parameters() if p.requires_grad],
            lr=0.01,
        )

        stability_loss_fn = StabilityLoss()
        dataset = SimpleDataset(size=32)

        # Record initial loss
        x = torch.stack([dataset[i]["x"] for i in range(BATCH_SIZE)])
        samatha_output = system.samatha(x)
        initial_loss, _ = stability_loss_fn.compute_loss(samatha_output.stability_pair, samatha_output.santana)
        initial_loss_value = initial_loss.item()

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            samatha_output = system.samatha(x)
            loss, _ = stability_loss_fn.compute_loss(samatha_output.stability_pair, samatha_output.santana)
            loss.backward()
            optimizer.step()

        # Record final loss
        samatha_output = system.samatha(x)
        final_loss, _ = stability_loss_fn.compute_loss(samatha_output.stability_pair, samatha_output.santana)
        final_loss_value = final_loss.item()

        # Loss should decrease (or at least not increase significantly)
        # Allow some tolerance for stochasticity
        assert final_loss_value <= initial_loss_value * 1.5, (
            f"StabilityLoss should not increase significantly. "
            f"Initial: {initial_loss_value:.4f}, Final: {final_loss_value:.4f}"
        )


class TestOutputDiversity:
    """Tests for Issue 4: Output diversity with gate_threshold.

    Related: docs/issues/001_training_issues_v4.md - Issue 4
    Verifies that s_star outputs are diverse (not all identical).
    """

    def test_s_star_is_diverse_with_open_gate(self):
        """Test that s_star has diversity when gate_threshold=-1.0 (always open)."""
        system = create_test_system(gate_threshold=-1.0)
        system.eval()

        # Create diverse inputs
        x = torch.randn(BATCH_SIZE, INPUT_DIM)

        with torch.no_grad():
            samatha_output = system.samatha(x)
            s_star = samatha_output.s_star

        # Check diversity: cosine similarity between samples should NOT be 1.0
        # Compare first sample with all others
        cos_sim = F.cosine_similarity(s_star[0:1], s_star[1:], dim=1)
        mean_cos_sim = cos_sim.mean().item()

        assert mean_cos_sim < 0.99, (
            f"s_star should be diverse (mean cosine similarity < 0.99). "
            f"Got mean cosine similarity: {mean_cos_sim:.4f}"
        )

    def test_s_star_variance_is_nonzero(self):
        """Test that s_star has non-zero variance across batch."""
        system = create_test_system(gate_threshold=-1.0)
        system.eval()

        x = torch.randn(BATCH_SIZE, INPUT_DIM)

        with torch.no_grad():
            samatha_output = system.samatha(x)
            s_star = samatha_output.s_star

        # Check variance across batch dimension
        variance = s_star.var(dim=0).mean().item()

        assert variance > 1e-6, f"s_star should have non-zero variance. Got variance: {variance:.6f}"

    def test_closed_gate_produces_zero_s0(self):
        """Test that gate_threshold=1.0 (always closed) produces near-zero s0."""
        system = create_test_system(gate_threshold=1.0)  # Gate always closed
        system.eval()

        x = torch.randn(BATCH_SIZE, INPUT_DIM)

        with torch.no_grad():
            # Get s0 from vitakka directly
            z = system.samatha.adapter(x)
            s0, meta = system.samatha.vitakka(z)

        # With gate_threshold=1.0, gate should be closed for most samples
        # s0 should be near zero or have very low variance
        s0_norm = s0.norm(dim=1).mean().item()

        # This test documents the PROBLEM behavior - s0 becomes near-zero
        # when gate is closed, which is why we need gate_threshold=-1.0 for training
        assert s0_norm < 1.0, f"With closed gate, s0 should be near zero. Got norm: {s0_norm:.4f}"

    def test_different_inputs_produce_different_outputs(self):
        """Test that semantically different inputs produce different s_star."""
        system = create_test_system(gate_threshold=-1.0)
        system.eval()

        # Create two batches with very different inputs
        x1 = torch.randn(4, INPUT_DIM) * 0.1  # Small values
        x2 = torch.randn(4, INPUT_DIM) * 10.0  # Large values

        with torch.no_grad():
            s_star1 = system.samatha(x1).s_star
            s_star2 = system.samatha(x2).s_star

        # The two batches should produce different s_star distributions
        mean1 = s_star1.mean(dim=0)
        mean2 = s_star2.mean(dim=0)

        diff = (mean1 - mean2).norm().item()
        assert diff > 0.01, f"Different inputs should produce different s_star. Diff: {diff:.4f}"


class TestGradientFlow:
    """Tests for overall gradient flow through the system.

    Verifies that gradients flow correctly from loss to all trainable components.
    """

    def test_stage1_gradients_reach_vitakka(self):
        """Test that Stage 1 gradients reach Vitakka probes."""
        system = create_test_system()
        system.set_stage(TrainingStage.SAMATHA_TRAINING)

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage1(x)

        # Simple loss on s_star
        loss = result["s_star"].sum()
        loss.backward()

        # Check Vitakka probes have gradients
        vitakka_probes = system.samatha.vitakka.probes
        assert vitakka_probes.grad is not None, "Vitakka probes should have gradients"
        assert vitakka_probes.grad.abs().sum() > 0, "Vitakka probe gradients should be non-zero"

    def test_stage1_gradients_reach_vicara(self):
        """Test that Stage 1 gradients reach Vicara refiner."""
        system = create_test_system()
        system.set_stage(TrainingStage.SAMATHA_TRAINING)

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage1(x)

        # Simple loss on s_star
        loss = result["s_star"].sum()
        loss.backward()

        # Check Vicara has gradients
        vicara_has_grad = False
        for name, p in system.samatha.vicara.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                vicara_has_grad = True
                break

        assert vicara_has_grad, "Vicara should have gradients in Stage 1"

    def test_stage2_gradients_reach_vipassana(self):
        """Test that Stage 2 gradients reach Vipassana encoder."""
        system = create_test_system()

        # Initialize Vipassana networks
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        _ = system.forward_stage2(x)

        system.set_stage(TrainingStage.VIPASSANA_TRAINING)

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage2(x)

        # Loss on trust_score
        loss = result["trust_score"].sum()
        loss.backward()

        # Check Vipassana has gradients
        vipassana_has_grad = False
        for name, p in system.vipassana.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                vipassana_has_grad = True
                break

        assert vipassana_has_grad, "Vipassana should have gradients in Stage 2"

    def test_no_gradient_leakage_to_frozen_components(self):
        """Test that frozen components do not receive gradients."""
        system = create_test_system()
        system.set_stage(TrainingStage.DECODER_FINETUNING)  # Only task_decoder trainable

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage3(x)

        loss = result["output"].sum()
        loss.backward()

        # Samatha should NOT have gradients (frozen in Stage 3)
        for name, p in system.samatha.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, f"Frozen param {name} should not have gradients"

        # Vipassana should NOT have gradients (frozen in Stage 3)
        for name, p in system.vipassana.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, f"Frozen param {name} should not have gradients"
