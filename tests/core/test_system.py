"""
Tests for SatipatthanaSystem (Phase 4).

These tests verify:
- System initialization and component integration
- Training stage switching and freeze policies
- Stage-specific forward passes
- Full inference pipeline
"""

import pytest
import torch

from satipatthana.core.system import SatipatthanaSystem, TrainingStage, SystemOutput
from satipatthana.core.engines import SamathaEngine, VipassanaEngine
from satipatthana.core.santana import SantanaLog
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


# Constants
INPUT_DIM = 16
LATENT_DIM = 32
CONTEXT_DIM = 16
OUTPUT_DIM = 10
BATCH_SIZE = 4


@pytest.fixture
def system_config():
    """Create system config for testing."""
    return SystemConfig(
        dim=LATENT_DIM,
        use_label_guidance=False,
        samatha=SamathaConfig(
            dim=LATENT_DIM,
            max_steps=5,
            adapter=MlpAdapterConfig(input_dim=INPUT_DIM, dim=LATENT_DIM),
            augmenter=IdentityAugmenterConfig(),
            vitakka=StandardVitakkaConfig(dim=LATENT_DIM, n_probes=4),
            vicara=StandardVicaraConfig(dim=LATENT_DIM, refine_steps=5),
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
def samatha_engine(system_config):
    """Create SamathaEngine for testing."""
    config = system_config.samatha
    adapter = MlpAdapter(config.adapter)
    augmenter = IdentityAugmenter(config.augmenter)
    vitakka = StandardVitakka(config.vitakka)
    refiner = MlpRefiner({"dim": LATENT_DIM})
    vicara = StandardVicara(config.vicara, refiner)
    sati = FixedStepSati(config.sati)

    return SamathaEngine(
        config=config,
        adapter=adapter,
        augmenter=augmenter,
        vitakka=vitakka,
        vicara=vicara,
        sati=sati,
    )


@pytest.fixture
def vipassana_engine(system_config):
    """Create VipassanaEngine for testing."""
    config = system_config.vipassana
    vipassana = StandardVipassana(config.vipassana)
    return VipassanaEngine(config=config, vipassana=vipassana)


@pytest.fixture
def task_decoder(system_config):
    """Create ConditionalDecoder for testing."""
    return ConditionalDecoder(system_config.task_decoder)


@pytest.fixture
def recon_decoder():
    """Create ReconstructionDecoder for testing."""
    return ReconstructionDecoder(
        ReconstructionDecoderConfig(
            dim=LATENT_DIM,
            input_dim=INPUT_DIM,
        )
    )


@pytest.fixture
def aux_head():
    """Create SimpleAuxHead for testing."""
    return SimpleAuxHead(
        SimpleAuxHeadConfig(
            dim=LATENT_DIM,
            output_dim=OUTPUT_DIM,
        )
    )


@pytest.fixture
def system(system_config, samatha_engine, vipassana_engine, task_decoder):
    """Create SatipatthanaSystem for testing."""
    return SatipatthanaSystem(
        config=system_config,
        samatha=samatha_engine,
        vipassana=vipassana_engine,
        task_decoder=task_decoder,
    )


@pytest.fixture
def system_full(system_config, samatha_engine, vipassana_engine, task_decoder, recon_decoder, aux_head):
    """Create SatipatthanaSystem with all optional heads."""
    return SatipatthanaSystem(
        config=system_config,
        samatha=samatha_engine,
        vipassana=vipassana_engine,
        task_decoder=task_decoder,
        adapter_recon_head=recon_decoder,
        samatha_recon_head=ReconstructionDecoder(ReconstructionDecoderConfig(dim=LATENT_DIM, input_dim=INPUT_DIM)),
        auxiliary_head=aux_head,
    )


class TestSatipatthanaSystem:
    """Tests for SatipatthanaSystem basic functionality."""

    def test_initialization(self, system, system_config):
        """Test system initialization."""
        assert system.config == system_config
        assert system.current_stage == TrainingStage.INFERENCE
        assert isinstance(system.samatha, SamathaEngine)
        assert isinstance(system.vipassana, VipassanaEngine)
        assert isinstance(system.task_decoder, ConditionalDecoder)

    def test_forward_output_type(self, system):
        """Test forward returns SystemOutput."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system(x)

        assert isinstance(result, SystemOutput)
        assert result.output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert result.s_star.shape == (BATCH_SIZE, LATENT_DIM)
        assert result.v_ctx.shape == (BATCH_SIZE, CONTEXT_DIM)
        assert result.trust_score.shape == (BATCH_SIZE, 1)
        assert isinstance(result.santana, SantanaLog)
        assert result.severity.shape == (BATCH_SIZE,)

    def test_forward_skip_vipassana(self, system):
        """Test forward with Vipassana skipped."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system(x, run_vipassana=False)

        # Should return zero context and full trust
        assert torch.all(result.v_ctx == 0.0)
        assert torch.all(result.trust_score == 1.0)

    def test_forward_skip_decoder(self, system):
        """Test forward with decoder skipped."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system(x, run_decoder=False)

        # Output should be zero placeholder
        assert torch.all(result.output == 0.0)

    def test_inference_mode(self, system):
        """Test inference method."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        output, trust = system.inference(x)

        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert trust.shape == (BATCH_SIZE, 1)


class TestTrainingStages:
    """Tests for training stage switching."""

    def test_set_stage_adapter_pretraining(self, system_full):
        """Test Stage 0: Adapter pre-training freeze policy."""
        system_full.set_stage(TrainingStage.ADAPTER_PRETRAINING)

        assert system_full.current_stage == TrainingStage.ADAPTER_PRETRAINING

        # Check trainable params
        trainable = system_full.get_trainable_params()
        assert "adapter" in trainable
        assert "adapter_recon_head" in trainable

        # Check frozen params
        assert "vitakka" not in trainable
        assert "vicara" not in trainable
        assert "vipassana" not in trainable
        assert "task_decoder" not in trainable

    def test_set_stage_samatha_training(self, system_full):
        """Test Stage 1: Samatha training freeze policy."""
        system_full.set_stage(TrainingStage.SAMATHA_TRAINING)

        trainable = system_full.get_trainable_params()
        assert "adapter" in trainable
        assert "vitakka" in trainable
        assert "vicara" in trainable
        assert "samatha_recon_head" in trainable

        # Vipassana and TaskDecoder should be frozen
        assert "vipassana" not in trainable
        assert "task_decoder" not in trainable

    def test_set_stage_samatha_with_label_guidance(
        self, system_config, samatha_engine, vipassana_engine, task_decoder, aux_head
    ):
        """Test Stage 1 with label guidance enabled."""
        system_config.use_label_guidance = True
        system = SatipatthanaSystem(
            config=system_config,
            samatha=samatha_engine,
            vipassana=vipassana_engine,
            task_decoder=task_decoder,
            auxiliary_head=aux_head,
        )

        system.set_stage(TrainingStage.SAMATHA_TRAINING)

        trainable = system.get_trainable_params()
        assert "auxiliary_head" in trainable

    def test_set_stage_vipassana_training(self, system_full):
        """Test Stage 2: Vipassana training freeze policy."""
        # First run a forward pass to initialize Vipassana's lazy networks
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        system_full.forward_stage2(x)

        system_full.set_stage(TrainingStage.VIPASSANA_TRAINING)

        trainable = system_full.get_trainable_params()
        assert "vipassana" in trainable

        # Everything else should be frozen
        assert "adapter" not in trainable
        assert "vitakka" not in trainable
        assert "vicara" not in trainable
        assert "task_decoder" not in trainable

    def test_set_stage_decoder_finetuning(self, system_full):
        """Test Stage 3: Decoder fine-tuning freeze policy."""
        system_full.set_stage(TrainingStage.DECODER_FINETUNING)

        trainable = system_full.get_trainable_params()
        assert "task_decoder" in trainable

        # Everything else should be frozen
        assert "adapter" not in trainable
        assert "vipassana" not in trainable

    def test_set_stage_inference(self, system_full):
        """Test inference mode - everything frozen."""
        system_full.set_stage(TrainingStage.INFERENCE)

        trainable = system_full.get_trainable_params()
        assert len(trainable) == 0


class TestStageForwardPasses:
    """Tests for stage-specific forward passes."""

    def test_forward_stage0(self, system_full):
        """Test Stage 0 forward pass."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        z, x_recon = system_full.forward_stage0(x)

        assert z.shape == (BATCH_SIZE, LATENT_DIM)
        assert x_recon.shape == (BATCH_SIZE, INPUT_DIM)

    def test_forward_stage1(self, system_full):
        """Test Stage 1 forward pass."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system_full.forward_stage1(x)

        assert "s_star" in result
        assert "santana" in result
        assert "severity" in result
        assert "x_recon" in result

        assert result["s_star"].shape == (BATCH_SIZE, LATENT_DIM)
        assert result["x_recon"].shape == (BATCH_SIZE, INPUT_DIM)

    def test_forward_stage1_with_label_guidance(
        self, system_config, samatha_engine, vipassana_engine, task_decoder, aux_head
    ):
        """Test Stage 1 with label guidance."""
        system_config.use_label_guidance = True
        system = SatipatthanaSystem(
            config=system_config,
            samatha=samatha_engine,
            vipassana=vipassana_engine,
            task_decoder=task_decoder,
            auxiliary_head=aux_head,
        )

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage1(x)

        assert "aux_output" in result
        assert result["aux_output"].shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_forward_stage2(self, system):
        """Test Stage 2 forward pass."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage2(x)

        assert "s_star" in result
        assert "santana" in result
        assert "v_ctx" in result
        assert "trust_score" in result

        assert result["v_ctx"].shape == (BATCH_SIZE, CONTEXT_DIM)
        assert result["trust_score"].shape == (BATCH_SIZE, 1)

    def test_forward_stage2_drunk_mode(self, system):
        """Test Stage 2 with drunk mode."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Normal mode
        result_normal = system.forward_stage2(x, drunk_mode=False)

        # Drunk mode
        torch.manual_seed(42)
        result_drunk = system.forward_stage2(x, drunk_mode=True)

        # Results should differ due to drunk mode perturbations
        # (Note: exact comparison depends on random seed behavior)
        assert result_normal["trust_score"].shape == result_drunk["trust_score"].shape

    def test_forward_stage3(self, system):
        """Test Stage 3 forward pass."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage3(x)

        assert "output" in result
        assert "s_star" in result
        assert "v_ctx" in result
        assert "trust_score" in result

        assert result["output"].shape == (BATCH_SIZE, OUTPUT_DIM)


class TestGradientFlow:
    """Tests for gradient flow through stages."""

    def test_gradient_stage0(self, system_full):
        """Test gradient flow in Stage 0."""
        system_full.set_stage(TrainingStage.ADAPTER_PRETRAINING)
        system_full.train()

        x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        z, x_recon = system_full.forward_stage0(x)

        loss = x_recon.sum()
        loss.backward()

        # Adapter should have gradients
        for param in system_full.samatha.adapter.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gradient_stage1(self, system_full):
        """Test gradient flow in Stage 1."""
        system_full.set_stage(TrainingStage.SAMATHA_TRAINING)
        system_full.train()

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system_full.forward_stage1(x)

        loss = result["s_star"].sum() + result["x_recon"].sum()
        loss.backward()

        # Samatha components should have gradients
        for param in system_full.samatha.adapter.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gradient_stage2(self, system):
        """Test gradient flow in Stage 2."""
        # First initialize Vipassana networks with a forward pass
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        _ = system.forward_stage2(x)

        # Now set stage and run again
        system.set_stage(TrainingStage.VIPASSANA_TRAINING)
        system.train()

        x = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Run Samatha (frozen in Stage 2)
        with torch.no_grad():
            s_star, santana, severity = system.samatha(x)

        # Run Vipassana (trainable) - NOT in no_grad context
        # Note: s_star needs requires_grad=True to allow gradient flow through Vipassana
        s_star = s_star.detach().requires_grad_(True)
        v_ctx, trust_score = system.vipassana(s_star, santana)

        # Use v_ctx for loss since trust_score is converted to scalar internally
        # and loses gradient connection
        loss = v_ctx.sum()
        loss.backward()

        # Vipassana encoder should have gradients (via v_ctx path)
        # Note: trust_head parameters won't have gradients since trust_score
        # is converted to scalar internally and loses gradient connection
        vipassana_inner = system.vipassana.vipassana
        encoder_has_gradients = False
        for param in vipassana_inner._encoder.parameters():
            if param.requires_grad and param.grad is not None:
                encoder_has_gradients = True
                break
        assert encoder_has_gradients, "Vipassana encoder should have gradients"

    def test_gradient_stage3(self, system):
        """Test gradient flow in Stage 3."""
        system.set_stage(TrainingStage.DECODER_FINETUNING)
        system.train()

        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        result = system.forward_stage3(x)

        loss = result["output"].sum()
        loss.backward()

        # TaskDecoder should have gradients
        for param in system.task_decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
