"""
Tests for SamadhiV4Trainer.

These tests verify:
- Stage switching and freeze policies
- Stage-specific loss computation
- Noise generation strategies for Stage 2
- Curriculum training flow
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments

from samadhi.core.system import SamadhiSystem, TrainingStage
from samadhi.core.engines import SamathaEngine, VipassanaEngine
from samadhi.train.v4_trainer import SamadhiV4Trainer, Stage2NoiseStrategy
from samadhi.configs.system import SystemConfig, SamathaConfig, VipassanaEngineConfig
from samadhi.configs.adapters import MlpAdapterConfig
from samadhi.configs.vitakka import StandardVitakkaConfig
from samadhi.configs.vicara import StandardVicaraConfig
from samadhi.configs.sati import FixedStepSatiConfig
from samadhi.configs.augmenter import IdentityAugmenterConfig
from samadhi.configs.vipassana import StandardVipassanaConfig
from samadhi.configs.decoders import ConditionalDecoderConfig, ReconstructionDecoderConfig, SimpleAuxHeadConfig

from samadhi.components.adapters.mlp import MlpAdapter
from samadhi.components.augmenters.identity import IdentityAugmenter
from samadhi.components.vitakka.standard import StandardVitakka
from samadhi.components.vicara.standard import StandardVicara
from samadhi.components.refiners.mlp import MlpRefiner
from samadhi.components.sati.fixed_step import FixedStepSati
from samadhi.components.vipassana.standard import StandardVipassana
from samadhi.components.decoders.conditional import ConditionalDecoder
from samadhi.components.decoders.reconstruction import ReconstructionDecoder
from samadhi.components.decoders.auxiliary import SimpleAuxHead


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
        vipassana=VipassanaEngineConfig(vipassana=StandardVipassanaConfig(context_dim=CONTEXT_DIM, hidden_dim=32)),
        task_decoder=ConditionalDecoderConfig(
            dim=LATENT_DIM,
            context_dim=CONTEXT_DIM,
            output_dim=OUTPUT_DIM,
        ),
    )


@pytest.fixture
def system_full(system_config):
    """Create SamadhiSystem with all heads for testing."""
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

    return SamadhiSystem(
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


class TestSamadhiV4TrainerInit:
    """Tests for trainer initialization."""

    def test_initialization(self, system_full, training_args, dataset):
        """Test trainer initialization."""
        trainer = SamadhiV4Trainer(
            model=system_full,
            args=training_args,
            train_dataset=dataset,
            stage=TrainingStage.SAMATHA_TRAINING,
        )

        assert trainer.stage == TrainingStage.SAMATHA_TRAINING
        assert trainer.model == system_full

    def test_initialization_requires_system(self, training_args, dataset):
        """Test that trainer requires SamadhiSystem."""
        with pytest.raises(TypeError):
            SamadhiV4Trainer(
                model=torch.nn.Linear(10, 10),  # Wrong type
                args=training_args,
                train_dataset=dataset,
            )


class TestStageSwitching:
    """Tests for stage switching."""

    def test_set_stage_updates_model(self, system_full, training_args, dataset):
        """Test that set_stage updates both trainer and model."""
        trainer = SamadhiV4Trainer(
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
        trainer = SamadhiV4Trainer(
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


class TestLossComputation:
    """Tests for stage-specific loss computation."""

    def test_stage0_loss(self, system_full, training_args, dataset):
        """Test Stage 0 loss computation."""
        trainer = SamadhiV4Trainer(
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
        trainer = SamadhiV4Trainer(
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
        trainer = SamadhiV4Trainer(
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

        trainer = SamadhiV4Trainer(
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
        assert "bce_loss" in outputs

    def test_stage3_loss(self, system_full, training_args, dataset):
        """Test Stage 3 loss computation."""
        trainer = SamadhiV4Trainer(
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
        trainer = SamadhiV4Trainer(
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


class TestCurriculumTraining:
    """Tests for curriculum training flow."""

    def test_train_stage(self, system_full, training_args, dataset):
        """Test training a single stage."""
        # Create a simple dataset that's big enough
        big_dataset = DummyDataset(size=64)

        trainer = SamadhiV4Trainer(
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
        trainer = SamadhiV4Trainer(
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
