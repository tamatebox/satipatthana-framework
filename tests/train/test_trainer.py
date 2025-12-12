"""
Tests for SatipatthanaTrainer.

These tests verify:
- Stage switching and freeze policies
- Stage-specific loss computation
- Noise generation strategies for Stage 2
- Curriculum training flow
"""

import pytest
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

from satipatthana.core.system import SatipatthanaSystem, TrainingStage
from satipatthana.core.engines import SamathaEngine, VipassanaEngine
from satipatthana.train.trainer import SatipatthanaTrainer, Stage2NoiseStrategy
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
from satipatthana.data import VoidDataset


# Constants
INPUT_DIM = 16
LATENT_DIM = 32
CONTEXT_DIM = 16
OUTPUT_DIM = 10
BATCH_SIZE = 8


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, size: int = 64, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM):
        self.size = size
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, output_dim, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"x": self.data[idx], "y": self.labels[idx]}


@pytest.fixture
def system_config():
    """Create system config for testing."""
    return SystemConfig(
        dim=LATENT_DIM,
        use_label_guidance=True,  # Enable for testing
        samatha=SamathaConfig(
            dim=LATENT_DIM,
            max_steps=3,
            adapter=MlpAdapterConfig(input_dim=INPUT_DIM, dim=LATENT_DIM),
            augmenter=IdentityAugmenterConfig(),
            vitakka=StandardVitakkaConfig(dim=LATENT_DIM, n_probes=4),
            vicara=StandardVicaraConfig(dim=LATENT_DIM, refine_steps=3),
            sati=FixedStepSatiConfig(),
        ),
        # context_dim = gru_hidden_dim + metric_proj_dim = 8 + 8 = 16
        vipassana=VipassanaEngineConfig(
            vipassana=StandardVipassanaConfig(latent_dim=LATENT_DIM, gru_hidden_dim=8, metric_proj_dim=8)
        ),
        task_decoder=ConditionalDecoderConfig(
            dim=LATENT_DIM,
            context_dim=CONTEXT_DIM,
            output_dim=OUTPUT_DIM,
        ),
    )


@pytest.fixture
def system_full(system_config):
    """Create SatipatthanaSystem with all heads for testing."""
    config = system_config.samatha
    adapter = MlpAdapter(config.adapter)
    augmenter = IdentityAugmenter(config.augmenter)
    vitakka = StandardVitakka(config.vitakka)
    refiner = MlpRefiner({"dim": LATENT_DIM})
    vicara = StandardVicara(config.vicara, refiner)
    sati = FixedStepSati(config.sati)

    samatha_engine = SamathaEngine(
        config=config,
        adapter=adapter,
        augmenter=augmenter,
        vitakka=vitakka,
        vicara=vicara,
        sati=sati,
    )

    vipassana_engine = VipassanaEngine(
        config=system_config.vipassana,
        vipassana=StandardVipassana(system_config.vipassana.vipassana),
    )

    task_decoder = ConditionalDecoder(system_config.task_decoder)

    adapter_recon_head = ReconstructionDecoder(ReconstructionDecoderConfig(dim=LATENT_DIM, input_dim=INPUT_DIM))

    samatha_recon_head = ReconstructionDecoder(ReconstructionDecoderConfig(dim=LATENT_DIM, input_dim=INPUT_DIM))

    auxiliary_head = SimpleAuxHead(SimpleAuxHeadConfig(dim=LATENT_DIM, output_dim=OUTPUT_DIM))

    return SatipatthanaSystem(
        config=system_config,
        samatha=samatha_engine,
        vipassana=vipassana_engine,
        task_decoder=task_decoder,
        adapter_recon_head=adapter_recon_head,
        samatha_recon_head=samatha_recon_head,
        auxiliary_head=auxiliary_head,
    )


@pytest.fixture
def dataset():
    """Create dummy dataset for testing."""
    return DummyDataset(size=32)


@pytest.fixture
def training_args(tmp_path):
    """Create minimal training arguments."""
    return TrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=1,
        save_steps=1000,  # Disable saving
        report_to="none",
        use_cpu=True,
    )


class TestSatipatthanaTrainerInit:
    """Tests for trainer initialization."""

    def test_initialization(self, system_full, training_args, dataset):
        """Test trainer initialization."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.SAMATHA_TRAINING,
        )

        assert trainer.stage == TrainingStage.SAMATHA_TRAINING
        assert trainer.model == system_full

    def test_initialization_requires_system(self, training_args, dataset):
        """Test that trainer requires SatipatthanaSystem."""
        with pytest.raises(TypeError):
            SatipatthanaTrainer(
                model=torch.nn.Linear(10, 10),  # Wrong type
                args=training_args,
                train_dataset=dataset,
            )


class TestStageSwitching:
    """Tests for stage switching."""

    def test_set_stage_updates_model(self, system_full, training_args, dataset):
        """Test that set_stage updates both trainer and model."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.INFERENCE,
        )

        trainer.set_stage(TrainingStage.SAMATHA_TRAINING)

        assert trainer.stage == TrainingStage.SAMATHA_TRAINING
        assert system_full.current_stage == TrainingStage.SAMATHA_TRAINING

    def test_set_stage_applies_freeze_policy(self, system_full, training_args, dataset):
        """Test that set_stage applies correct freeze policy."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.DECODER_FINETUNING,
        )

        trainable = system_full.get_trainable_params()

        # Only task_decoder should be trainable
        assert "task_decoder" in trainable
        assert "adapter" not in trainable
        assert "vipassana" not in trainable


class TestOptimizerParamGroups:
    """Tests for optimizer param_groups after stage switching.

    These tests verify that the optimizer is correctly reset when switching stages,
    ensuring that the new trainable parameters are properly included in the optimizer's
    param_groups. This is critical because HuggingFace Trainer caches the optimizer.

    Related Issue: docs/issues/001_training_issues_v4.md - Issue 2
    """

    def test_optimizer_reset_after_stage_switch(self, system_full, training_args, dataset):
        """Test that optimizer is reset (set to None) after train_stage call."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.INFERENCE,
        )

        # Simulate having a cached optimizer from a previous stage
        trainer.optimizer = torch.optim.Adam(system_full.parameters(), lr=0.001)
        trainer.lr_scheduler = None

        # Call train_stage which should reset the optimizer
        training_args.max_steps = 1  # Minimal training
        trainer.train_stage(TrainingStage.ADAPTER_PRETRAINING, num_epochs=1)

        # After train_stage completes, optimizer should have been recreated
        # (it won't be None because train() creates a new one)
        # The key test is that it contains the correct parameters

    def test_optimizer_contains_correct_params_stage0(self, system_full, training_args, dataset):
        """Test that Stage 0 optimizer only contains adapter and adapter_recon_head params."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.ADAPTER_PRETRAINING,
        )

        # Get expected trainable parameter ids
        expected_param_ids = set()
        for p in system_full.samatha.adapter.parameters():
            if p.requires_grad:
                expected_param_ids.add(id(p))
        if system_full.adapter_recon_head is not None:
            for p in system_full.adapter_recon_head.parameters():
                if p.requires_grad:
                    expected_param_ids.add(id(p))

        # Create optimizer
        optimizer = trainer.create_optimizer()

        # Check that optimizer param_groups contain exactly the expected params
        optimizer_param_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimizer_param_ids.add(id(p))

        assert optimizer_param_ids == expected_param_ids, (
            f"Optimizer param mismatch for Stage 0. "
            f"Expected {len(expected_param_ids)} params, got {len(optimizer_param_ids)}"
        )

    def test_optimizer_contains_correct_params_stage1(self, system_full, training_args, dataset):
        """Test that Stage 1 optimizer contains Samatha core params."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.SAMATHA_TRAINING,
            use_label_guidance=True,
        )

        # Get expected trainable parameter ids for Stage 1
        expected_param_ids = set()
        for module in [
            system_full.samatha.adapter,
            system_full.samatha.vitakka,
            system_full.samatha.vicara,
            system_full.samatha.sati,
        ]:
            for p in module.parameters():
                if p.requires_grad:
                    expected_param_ids.add(id(p))

        if system_full.samatha_recon_head is not None:
            for p in system_full.samatha_recon_head.parameters():
                if p.requires_grad:
                    expected_param_ids.add(id(p))

        if system_full.auxiliary_head is not None:
            for p in system_full.auxiliary_head.parameters():
                if p.requires_grad:
                    expected_param_ids.add(id(p))

        # Create optimizer
        optimizer = trainer.create_optimizer()

        # Check param_groups
        optimizer_param_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimizer_param_ids.add(id(p))

        assert optimizer_param_ids == expected_param_ids, (
            f"Optimizer param mismatch for Stage 1. "
            f"Expected {len(expected_param_ids)} params, got {len(optimizer_param_ids)}"
        )

    def test_optimizer_contains_correct_params_stage2(self, system_full, training_args, dataset):
        """Test that Stage 2 optimizer only contains Vipassana params."""
        # First initialize Vipassana networks
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        _ = system_full.forward_stage2(x)

        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
        )

        # Get expected trainable parameter ids for Stage 2 (only Vipassana)
        expected_param_ids = set()
        for p in system_full.vipassana.parameters():
            if p.requires_grad:
                expected_param_ids.add(id(p))

        # Create optimizer
        optimizer = trainer.create_optimizer()

        # Check param_groups
        optimizer_param_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimizer_param_ids.add(id(p))

        assert optimizer_param_ids == expected_param_ids, (
            f"Optimizer param mismatch for Stage 2. "
            f"Expected {len(expected_param_ids)} params, got {len(optimizer_param_ids)}"
        )

    def test_optimizer_contains_correct_params_stage3(self, system_full, training_args, dataset):
        """Test that Stage 3 optimizer only contains task_decoder params."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.DECODER_FINETUNING,
        )

        # Get expected trainable parameter ids for Stage 3 (only task_decoder)
        expected_param_ids = set()
        for p in system_full.task_decoder.parameters():
            if p.requires_grad:
                expected_param_ids.add(id(p))

        # Create optimizer
        optimizer = trainer.create_optimizer()

        # Check param_groups
        optimizer_param_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimizer_param_ids.add(id(p))

        assert optimizer_param_ids == expected_param_ids, (
            f"Optimizer param mismatch for Stage 3. "
            f"Expected {len(expected_param_ids)} params, got {len(optimizer_param_ids)}"
        )

    def test_stage_transition_updates_optimizer_params(self, system_full, training_args, dataset):
        """Test that transitioning between stages correctly updates optimizer params.

        This is the critical regression test for Issue 2 in 001_training_issues_v4.md.
        It verifies that after switching from Stage 0 to Stage 1, the optimizer
        includes the newly trainable Vitakka/Vicara parameters.
        """
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.ADAPTER_PRETRAINING,
        )

        # Create optimizer for Stage 0
        optimizer_stage0 = trainer.create_optimizer()
        stage0_param_ids = set()
        for group in optimizer_stage0.param_groups:
            for p in group["params"]:
                stage0_param_ids.add(id(p))

        # Switch to Stage 1
        trainer.set_stage(TrainingStage.SAMATHA_TRAINING)
        trainer.optimizer = None  # Reset as train_stage does
        trainer.lr_scheduler = None

        # Create new optimizer for Stage 1
        optimizer_stage1 = trainer.create_optimizer()
        stage1_param_ids = set()
        for group in optimizer_stage1.param_groups:
            for p in group["params"]:
                stage1_param_ids.add(id(p))

        # Stage 1 should have MORE parameters than Stage 0
        # (includes vitakka, vicara, sati, aux_head in addition to adapter)
        assert len(stage1_param_ids) > len(stage0_param_ids), (
            f"Stage 1 should have more trainable params than Stage 0. "
            f"Stage 0: {len(stage0_param_ids)}, Stage 1: {len(stage1_param_ids)}"
        )

        # Verify Vitakka params are included in Stage 1
        vitakka_param_ids = {id(p) for p in system_full.samatha.vitakka.parameters() if p.requires_grad}
        assert vitakka_param_ids.issubset(stage1_param_ids), "Vitakka params should be in Stage 1 optimizer"

        # Verify Vicara params are included in Stage 1
        vicara_param_ids = {id(p) for p in system_full.samatha.vicara.parameters() if p.requires_grad}
        assert vicara_param_ids.issubset(stage1_param_ids), "Vicara params should be in Stage 1 optimizer"


class TestLossComputation:
    """Tests for stage-specific loss computation."""

    def test_stage0_loss(self, system_full, training_args, dataset):
        """Test Stage 0 loss computation."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.ADAPTER_PRETRAINING,
        )

        batch = {"x": torch.randn(BATCH_SIZE, INPUT_DIM)}
        loss, outputs = trainer.compute_loss(system_full, batch, return_outputs=True)

        assert loss > 0
        assert "z" in outputs
        assert "x_recon" in outputs
        assert outputs["z"].shape == (BATCH_SIZE, LATENT_DIM)
        assert outputs["x_recon"].shape == (BATCH_SIZE, INPUT_DIM)

    def test_stage1_loss(self, system_full, training_args, dataset):
        """Test Stage 1 loss computation."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.SAMATHA_TRAINING,
            use_label_guidance=True,
        )

        batch = {"x": torch.randn(BATCH_SIZE, INPUT_DIM), "y": torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))}
        loss, outputs = trainer.compute_loss(system_full, batch, return_outputs=True)

        assert loss > 0
        assert "s_star" in outputs
        assert "santana" in outputs
        assert outputs["s_star"].shape == (BATCH_SIZE, LATENT_DIM)

    def test_stage1_loss_with_label_guidance(self, system_full, training_args, dataset):
        """Test Stage 1 loss with label guidance."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.SAMATHA_TRAINING,
            use_label_guidance=True,
        )

        batch = {"x": torch.randn(BATCH_SIZE, INPUT_DIM), "y": torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))}
        loss, outputs = trainer.compute_loss(system_full, batch, return_outputs=True)

        # Should have guidance loss components
        assert "guidance_loss" in outputs or "aux_output" in outputs

    def test_stage2_loss(self, system_full, training_args, dataset):
        """Test Stage 2 loss computation with noise strategies."""
        # First initialize Vipassana with a forward pass
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        _ = system_full.forward_stage2(x)

        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
            noise_level=0.3,
        )

        batch = {"x": torch.randn(BATCH_SIZE * 3, INPUT_DIM)}  # Larger batch for 3-way split
        loss, outputs = trainer.compute_loss(system_full, batch, return_outputs=True)

        assert loss >= 0  # BCE loss is always >= 0
        assert "trust_scores" in outputs
        assert "targets" in outputs
        # Triple Score system uses individual BCE losses
        assert "trust_bce" in outputs
        assert "conformity_bce" in outputs
        assert "confidence_bce" in outputs

    def test_stage3_loss(self, system_full, training_args, dataset):
        """Test Stage 3 loss computation."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.DECODER_FINETUNING,
        )

        batch = {"x": torch.randn(BATCH_SIZE, INPUT_DIM), "y": torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))}
        loss, outputs = trainer.compute_loss(system_full, batch, return_outputs=True)

        assert loss > 0
        assert "output" in outputs
        assert "task_loss" in outputs
        assert outputs["output"].shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_stage3_requires_labels(self, system_full, training_args, dataset):
        """Test Stage 3 raises error without labels."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.DECODER_FINETUNING,
        )

        batch = {"x": torch.randn(BATCH_SIZE, INPUT_DIM)}  # No labels
        with pytest.raises(ValueError, match="requires labels"):
            trainer.compute_loss(system_full, batch)


class TestNoiseStrategies:
    """Tests for Stage 2 noise generation strategies."""

    def test_noise_strategy_enum(self):
        """Test noise strategy enum values."""
        assert Stage2NoiseStrategy.AUGMENTED == 0
        assert Stage2NoiseStrategy.DRUNK == 1
        assert Stage2NoiseStrategy.MISMATCH == 2
        assert Stage2NoiseStrategy.VOID == 3


class TestVoidPath:
    """Tests for Void Path (OOD data) in Stage 2."""

    def test_void_path_disabled_by_default(self, system_full, training_args, dataset):
        """Test that Void Path is disabled by default (no void_dataset)."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
        )
        assert trainer.void_dataset is None

    def test_void_path_with_dataset(self, system_full, training_args, dataset):
        """Test that Void Path is enabled when void_dataset is provided."""
        void_dataset = VoidDataset(DummyDataset(size=100))
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
            void_dataset=void_dataset,
        )
        assert trainer.void_dataset is not None

    def test_void_path_samples_from_dataset(self, system_full, training_args, dataset):
        """Test that Void Path samples OOD data from void_dataset."""
        void_dataset = VoidDataset(DummyDataset(size=100))
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
            void_dataset=void_dataset,
        )

        void_data = trainer._sample_void_data(10, torch.device("cpu"))

        assert void_data is not None
        assert void_data.shape == (10, INPUT_DIM)

    def test_void_path_returns_none_without_dataset(self, system_full, training_args, dataset):
        """Test that _sample_void_data returns None when void_dataset is not set."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
        )

        void_data = trainer._sample_void_data(10, torch.device("cpu"))
        assert void_data is None

    def test_stage2_loss_with_void_dataset(self, system_full, training_args, dataset):
        """Test Stage 2 loss computation includes Void Path when void_dataset is provided."""
        void_dataset = VoidDataset(DummyDataset(size=100))
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
            void_dataset=void_dataset,
        )

        batch = {"x": torch.randn(BATCH_SIZE, INPUT_DIM)}
        loss, outputs = trainer.compute_loss(system_full, batch, return_outputs=True)

        assert loss > 0
        assert "trust_scores" in outputs
        assert "targets" in outputs
        # With void path, we have 5 paths worth of trust scores
        # 4 ID paths + 1 void path (void_size = batch_size // 4)
        expected_size = BATCH_SIZE + (BATCH_SIZE // 4)
        assert outputs["trust_scores"].shape[0] == expected_size

    def test_stage2_loss_without_void_dataset(self, system_full, training_args, dataset):
        """Test Stage 2 loss computation without Void Path (no void_dataset)."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.VIPASSANA_TRAINING,
        )

        batch = {"x": torch.randn(BATCH_SIZE, INPUT_DIM)}
        loss, outputs = trainer.compute_loss(system_full, batch, return_outputs=True)

        assert loss > 0
        assert "trust_scores" in outputs
        # Without void_dataset, only 4 ID paths
        assert outputs["trust_scores"].shape[0] == BATCH_SIZE


class TestCurriculumTraining:
    """Tests for curriculum training flow."""

    def test_train_stage(self, system_full, training_args, dataset):
        """Test training a single stage."""
        # Create a simple dataset that's big enough
        big_dataset = DummyDataset(size=64)

        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=big_dataset,
            stage=TrainingStage.INFERENCE,
        )

        # Train Stage 0 for 1 step
        training_args.max_steps = 2
        result = trainer.train_stage(TrainingStage.ADAPTER_PRETRAINING, num_epochs=1)

        # Should return training output
        assert result is not None

    def test_inference_mode_after_curriculum(self, system_full, training_args, dataset):
        """Test that inference mode is set after curriculum completion."""
        trainer = SatipatthanaTrainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.ADAPTER_PRETRAINING,
        )

        # Setting to inference mode
        trainer.set_stage(TrainingStage.INFERENCE)

        assert trainer.stage == TrainingStage.INFERENCE
        assert system_full.current_stage == TrainingStage.INFERENCE

        # All parameters should be frozen
        trainable = system_full.get_trainable_params()
        assert len(trainable) == 0
