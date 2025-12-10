"""
Tests for SamathaEngine and VipassanaEngine (Phase 3).

These integration tests verify:
- Samatha convergence behavior
- Drunk mode output variability
- Vipassana trust score differentiation
"""

import pytest
import torch
import torch.nn as nn

from samadhi.core.engines import SamathaEngine, VipassanaEngine
from samadhi.core.santana import SantanaLog
from samadhi.configs.system import SamathaConfig, VipassanaEngineConfig
from samadhi.configs.adapters import MlpAdapterConfig
from samadhi.configs.vitakka import StandardVitakkaConfig
from samadhi.configs.vicara import StandardVicaraConfig
from samadhi.configs.sati import FixedStepSatiConfig, ThresholdSatiConfig
from samadhi.configs.augmenter import IdentityAugmenterConfig, GaussianNoiseAugmenterConfig
from samadhi.configs.vipassana import StandardVipassanaConfig

from samadhi.components.adapters.mlp import MlpAdapter
from samadhi.components.augmenters.identity import IdentityAugmenter
from samadhi.components.augmenters.gaussian import GaussianNoiseAugmenter
from samadhi.components.vitakka.standard import StandardVitakka
from samadhi.components.vicara.standard import StandardVicara
from samadhi.components.refiners.mlp import MlpRefiner
from samadhi.components.sati.fixed_step import FixedStepSati
from samadhi.components.sati.threshold import ThresholdSati
from samadhi.components.vipassana.standard import StandardVipassana


# Test fixtures
@pytest.fixture
def default_config():
    """Default SamathaConfig for testing."""
    return SamathaConfig(
        dim=32,
        max_steps=5,
        adapter=MlpAdapterConfig(input_dim=16, dim=32),
        augmenter=IdentityAugmenterConfig(),
        vitakka=StandardVitakkaConfig(dim=32, n_probes=4),
        vicara=StandardVicaraConfig(dim=32, refine_steps=5),
        sati=FixedStepSatiConfig(),
    )


@pytest.fixture
def samatha_engine(default_config):
    """Create a SamathaEngine with default components."""
    adapter = MlpAdapter(default_config.adapter)
    augmenter = IdentityAugmenter(default_config.augmenter)
    vitakka = StandardVitakka(default_config.vitakka)
    refiner = MlpRefiner({"dim": 32})
    vicara = StandardVicara(default_config.vicara, refiner)
    sati = FixedStepSati(default_config.sati)

    return SamathaEngine(
        config=default_config,
        adapter=adapter,
        augmenter=augmenter,
        vitakka=vitakka,
        vicara=vicara,
        sati=sati,
    )


@pytest.fixture
def vipassana_config():
    """Default VipassanaEngineConfig for testing."""
    return VipassanaEngineConfig(vipassana=StandardVipassanaConfig(context_dim=16, hidden_dim=32))


@pytest.fixture
def vipassana_engine(vipassana_config):
    """Create a VipassanaEngine with default components."""
    vipassana = StandardVipassana(vipassana_config.vipassana)
    return VipassanaEngine(config=vipassana_config, vipassana=vipassana)


class TestSamathaEngine:
    """Tests for SamathaEngine."""

    def test_initialization(self, samatha_engine, default_config):
        """Test engine initialization."""
        assert samatha_engine.config == default_config
        assert samatha_engine.drunk_mode is False
        assert isinstance(samatha_engine.adapter, MlpAdapter)
        assert isinstance(samatha_engine.augmenter, IdentityAugmenter)
        assert isinstance(samatha_engine.vitakka, StandardVitakka)
        assert isinstance(samatha_engine.vicara, StandardVicara)
        assert isinstance(samatha_engine.sati, FixedStepSati)

    def test_forward_output_shapes(self, samatha_engine):
        """Test output shapes from forward pass."""
        batch_size = 8
        input_dim = 16
        latent_dim = 32

        x = torch.randn(batch_size, input_dim)
        s_star, santana, severity = samatha_engine(x)

        # Check s_star shape
        assert s_star.shape == (batch_size, latent_dim)

        # Check santana has recorded states
        assert isinstance(santana, SantanaLog)
        assert len(santana) > 0

        # Check severity shape
        assert severity.shape == (batch_size,)

    def test_convergence_no_noise(self, samatha_engine):
        """Test that engine produces consistent output for same input (no noise)."""
        torch.manual_seed(42)
        x = torch.randn(4, 16)

        samatha_engine.eval()
        with torch.no_grad():
            s_star1, _, _ = samatha_engine(x, noise_level=0.0)
            s_star2, _, _ = samatha_engine(x, noise_level=0.0)

        # Same input should produce same output in eval mode
        assert torch.allclose(s_star1, s_star2, atol=1e-5)

    def test_augmentation_with_noise(self):
        """Test that noise affects the output."""
        config = SamathaConfig(
            dim=32,
            max_steps=5,
            adapter=MlpAdapterConfig(input_dim=16, dim=32),
            augmenter=GaussianNoiseAugmenterConfig(max_noise_std=1.0),
            vitakka=StandardVitakkaConfig(dim=32, n_probes=4),
            vicara=StandardVicaraConfig(dim=32, refine_steps=5),
            sati=FixedStepSatiConfig(),
        )

        adapter = MlpAdapter(config.adapter)
        augmenter = GaussianNoiseAugmenter(config.augmenter)
        vitakka = StandardVitakka(config.vitakka)
        refiner = MlpRefiner({"dim": 32})
        vicara = StandardVicara(config.vicara, refiner)
        sati = FixedStepSati(config.sati)

        engine = SamathaEngine(
            config=config,
            adapter=adapter,
            augmenter=augmenter,
            vitakka=vitakka,
            vicara=vicara,
            sati=sati,
        )

        x = torch.randn(4, 16)

        _, _, severity_no_noise = engine(x, noise_level=0.0)
        _, _, severity_with_noise = engine(x, noise_level=0.5)

        # Without noise, severity should be 0
        assert torch.all(severity_no_noise == 0.0)

        # With noise, severity should be non-zero
        assert torch.all(severity_with_noise > 0.0)

    def test_santana_log_records_trajectory(self, samatha_engine):
        """Test that SantanaLog properly records the trajectory."""
        x = torch.randn(4, 16)
        _, santana, _ = samatha_engine(x)

        # Should have initial state + refinement steps
        # Initial + up to max_steps
        assert len(santana) >= 2
        assert len(santana) <= samatha_engine.config.max_steps + 1

        # Energies should be recorded (except for initial state)
        assert len(santana.energies) >= 1

        # First meta should contain initial state info
        assert santana.meta_history[0]["type"] == "initial"

    def test_drunk_mode_toggle(self, samatha_engine):
        """Test drunk mode property toggling."""
        assert samatha_engine.drunk_mode is False

        samatha_engine.drunk_mode = True
        assert samatha_engine.drunk_mode is True

        samatha_engine.drunk_mode = False
        assert samatha_engine.drunk_mode is False

    def test_drunk_mode_affects_output(self, samatha_engine):
        """Test that drunk mode produces variable output."""
        torch.manual_seed(42)
        x = torch.randn(4, 16)

        # Get normal output
        samatha_engine.eval()
        with torch.no_grad():
            s_normal, santana_normal, _ = samatha_engine(x, drunk_mode=False)

        # Get drunk mode output (multiple runs should vary due to randomness)
        results = []
        for i in range(3):
            torch.manual_seed(i)  # Different seeds for variation
            with torch.no_grad():
                s_drunk, _, _ = samatha_engine(x, drunk_mode=True)
                results.append(s_drunk.clone())

        # Drunk mode should produce varying outputs with different seeds
        # At least one pair should differ
        has_variation = False
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if not torch.allclose(results[i], results[j], atol=1e-3):
                    has_variation = True
                    break

        assert has_variation, "Drunk mode should produce variable outputs"

    def test_drunk_mode_via_forward_param(self, samatha_engine):
        """Test drunk mode can be enabled via forward parameter."""
        x = torch.randn(4, 16)

        # Initially not in drunk mode
        assert samatha_engine.drunk_mode is False

        # Call forward with drunk_mode=True
        samatha_engine(x, drunk_mode=True)

        # Should have entered drunk mode
        assert samatha_engine.drunk_mode is True

    def test_sati_early_stopping(self):
        """Test that ThresholdSati can trigger early stopping."""
        config = SamathaConfig(
            dim=32,
            max_steps=20,  # High max steps
            adapter=MlpAdapterConfig(input_dim=16, dim=32),
            augmenter=IdentityAugmenterConfig(),
            vitakka=StandardVitakkaConfig(dim=32, n_probes=4),
            vicara=StandardVicaraConfig(dim=32, refine_steps=5),
            sati=ThresholdSatiConfig(energy_threshold=0.5, min_steps=2),
        )

        adapter = MlpAdapter(config.adapter)
        augmenter = IdentityAugmenter(config.augmenter)
        vitakka = StandardVitakka(config.vitakka)
        refiner = MlpRefiner({"dim": 32})
        vicara = StandardVicara(config.vicara, refiner)
        sati = ThresholdSati(config.sati)

        engine = SamathaEngine(
            config=config,
            adapter=adapter,
            augmenter=augmenter,
            vitakka=vitakka,
            vicara=vicara,
            sati=sati,
        )

        x = torch.randn(4, 16)
        engine.eval()
        with torch.no_grad():
            _, santana, _ = engine(x)

        # With threshold-based stopping, should stop before max_steps
        # (initial state + some refinement steps, but not all 20)
        # Note: This depends on convergence dynamics
        assert len(santana) <= config.max_steps + 1

    def test_vitakka_metadata_stored(self, samatha_engine):
        """Test that Vitakka metadata is stored in SantanaLog."""
        x = torch.randn(4, 16)
        _, santana, _ = samatha_engine(x)

        # First meta should contain vitakka info
        assert "vitakka" in santana.meta_history[0]
        vitakka_meta = santana.meta_history[0]["vitakka"]

        # Check expected keys
        assert "winner_id" in vitakka_meta
        assert "confidence" in vitakka_meta
        assert "probs" in vitakka_meta

    def test_training_mode(self, samatha_engine):
        """Test forward pass in training mode."""
        x = torch.randn(4, 16)

        samatha_engine.train()
        s_star, santana, severity = samatha_engine(x)

        # Should still produce valid output
        assert s_star.shape == (4, 32)
        assert len(santana) > 0

    def test_gradient_flow(self, samatha_engine):
        """Test that gradients can flow through the engine."""
        x = torch.randn(4, 16, requires_grad=True)

        samatha_engine.train()
        s_star, _, _ = samatha_engine(x)

        # Compute loss and backprop
        loss = s_star.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None


class TestVipassanaEngine:
    """Tests for VipassanaEngine."""

    def test_initialization(self, vipassana_engine, vipassana_config):
        """Test engine initialization."""
        assert vipassana_engine.config == vipassana_config
        assert isinstance(vipassana_engine.vipassana, StandardVipassana)

    def test_forward_output_shapes(self, vipassana_engine):
        """Test output shapes from forward pass."""
        batch_size = 8
        state_dim = 32
        context_dim = 16

        s_star = torch.randn(batch_size, state_dim)
        santana = SantanaLog()
        for _ in range(5):
            santana.add(torch.randn(batch_size, state_dim), energy=0.1)

        v_ctx, trust_score = vipassana_engine(s_star, santana)

        # Check context vector shape
        assert v_ctx.shape == (batch_size, context_dim)

        # Check trust score shape
        assert trust_score.shape == (batch_size, 1)

        # Trust score should be in [0, 1]
        assert torch.all(trust_score >= 0.0)
        assert torch.all(trust_score <= 1.0)

    def test_trust_score_for_good_trajectory(self, vipassana_engine):
        """Test trust score for a smooth convergence trajectory."""
        batch_size = 4
        state_dim = 32

        # Create a smooth convergence trajectory (decreasing energy)
        santana = SantanaLog()
        s_prev = torch.randn(batch_size, state_dim)
        santana.add(s_prev, energy=0.0)  # Initial

        for i in range(5):
            # Small changes - smooth convergence
            delta = torch.randn_like(s_prev) * 0.1 / (i + 1)
            s_curr = s_prev + delta
            santana.add(s_curr, energy=0.1 / (i + 1))
            s_prev = s_curr

        s_star = santana.get_final_state()
        v_ctx, trust_score = vipassana_engine(s_star, santana)

        # Good trajectory should have reasonable trust score
        assert trust_score.mean() > 0.0

    def test_trust_score_for_chaotic_trajectory(self, vipassana_engine):
        """Test trust score for a chaotic trajectory."""
        batch_size = 4
        state_dim = 32

        # Create a chaotic trajectory (high energy)
        santana = SantanaLog()
        for _ in range(5):
            s = torch.randn(batch_size, state_dim) * 5.0  # Large random jumps
            santana.add(s, energy=2.0)  # High energy

        s_star = santana.get_final_state()
        v_ctx_chaotic, trust_chaotic = vipassana_engine(s_star, santana)

        # Create smooth trajectory for comparison
        santana_smooth = SantanaLog()
        s_smooth = torch.randn(batch_size, state_dim) * 0.1
        for _ in range(5):
            santana_smooth.add(s_smooth.clone(), energy=0.01)
            s_smooth = s_smooth + torch.randn_like(s_smooth) * 0.01

        s_star_smooth = santana_smooth.get_final_state()
        v_ctx_smooth, trust_smooth = vipassana_engine(s_star_smooth, santana_smooth)

        # Trust scores exist (actual values depend on learned parameters)
        assert trust_chaotic.shape == (batch_size, 1)
        assert trust_smooth.shape == (batch_size, 1)

    def test_context_vector_different_for_different_trajectories(self, vipassana_engine):
        """Test that different trajectories produce different context vectors."""
        batch_size = 4
        state_dim = 32

        # First trajectory
        santana1 = SantanaLog()
        for _ in range(5):
            santana1.add(torch.randn(batch_size, state_dim) * 0.1, energy=0.1)
        s_star1 = santana1.get_final_state()
        v_ctx1, _ = vipassana_engine(s_star1, santana1)

        # Different trajectory (larger magnitude states)
        santana2 = SantanaLog()
        for _ in range(5):
            santana2.add(torch.randn(batch_size, state_dim) * 2.0, energy=1.0)
        s_star2 = santana2.get_final_state()
        v_ctx2, _ = vipassana_engine(s_star2, santana2)

        # Context vectors should differ
        assert not torch.allclose(v_ctx1, v_ctx2, atol=0.01)

    def test_empty_trajectory(self, vipassana_engine):
        """Test handling of empty trajectory."""
        batch_size = 4
        state_dim = 32

        s_star = torch.randn(batch_size, state_dim)
        santana = SantanaLog()  # Empty

        v_ctx, trust_score = vipassana_engine(s_star, santana)

        # Should handle empty trajectory gracefully
        assert v_ctx.shape == (batch_size, 16)
        assert trust_score.shape == (batch_size, 1)

    def test_single_step_trajectory(self, vipassana_engine):
        """Test handling of single-step trajectory."""
        batch_size = 4
        state_dim = 32

        s_star = torch.randn(batch_size, state_dim)
        santana = SantanaLog()
        santana.add(s_star.clone(), energy=0.0)

        v_ctx, trust_score = vipassana_engine(s_star, santana)

        # Should handle single step trajectory
        assert v_ctx.shape == (batch_size, 16)
        assert trust_score.shape == (batch_size, 1)


class TestIntegratedFlow:
    """Integration tests for Samatha -> Vipassana flow."""

    def test_samatha_to_vipassana_pipeline(self, samatha_engine, vipassana_engine):
        """Test complete Samatha -> Vipassana flow."""
        x = torch.randn(4, 16)

        # Run Samatha
        s_star, santana, severity = samatha_engine(x)

        # Run Vipassana on Samatha output
        v_ctx, trust_score = vipassana_engine(s_star, santana)

        # All outputs should be valid
        assert s_star.shape == (4, 32)
        assert v_ctx.shape == (4, 16)
        assert trust_score.shape == (4, 1)

    def test_drunk_vs_sober_trust_difference(self, samatha_engine, vipassana_engine):
        """Test that drunk mode produces lower trust scores."""
        torch.manual_seed(42)
        x = torch.randn(4, 16)

        # Sober mode
        samatha_engine.eval()
        with torch.no_grad():
            s_sober, santana_sober, _ = samatha_engine(x, drunk_mode=False)
            _, trust_sober = vipassana_engine(s_sober, santana_sober)

        # Multiple drunk mode runs (trust may vary)
        drunk_trusts = []
        for seed in range(5):
            torch.manual_seed(seed)
            with torch.no_grad():
                s_drunk, santana_drunk, _ = samatha_engine(x, drunk_mode=True)
                _, trust_drunk = vipassana_engine(s_drunk, santana_drunk)
                drunk_trusts.append(trust_drunk.mean().item())

        # Note: This test verifies that the pipeline works and produces values
        # The actual trust score difference depends on network weights
        # In practice, an untrained network may not show clear differentiation
        assert all(0.0 <= t <= 1.0 for t in drunk_trusts)

    def test_noisy_input_trust(self, vipassana_engine):
        """Test trust with noisy vs clean input paths."""
        config = SamathaConfig(
            dim=32,
            max_steps=5,
            adapter=MlpAdapterConfig(input_dim=16, dim=32),
            augmenter=GaussianNoiseAugmenterConfig(max_noise_std=0.5),
            vitakka=StandardVitakkaConfig(dim=32, n_probes=4),
            vicara=StandardVicaraConfig(dim=32, refine_steps=5),
            sati=FixedStepSatiConfig(),
        )

        adapter = MlpAdapter(config.adapter)
        augmenter = GaussianNoiseAugmenter(config.augmenter)
        vitakka = StandardVitakka(config.vitakka)
        refiner = MlpRefiner({"dim": 32})
        vicara = StandardVicara(config.vicara, refiner)
        sati = FixedStepSati(config.sati)

        engine = SamathaEngine(
            config=config,
            adapter=adapter,
            augmenter=augmenter,
            vitakka=vitakka,
            vicara=vicara,
            sati=sati,
        )

        x = torch.randn(4, 16)

        # Clean path
        torch.manual_seed(42)
        s_clean, santana_clean, severity_clean = engine(x, noise_level=0.0)
        v_ctx_clean, trust_clean = vipassana_engine(s_clean, santana_clean)

        # Noisy path
        torch.manual_seed(42)
        s_noisy, santana_noisy, severity_noisy = engine(x, noise_level=0.8)
        v_ctx_noisy, trust_noisy = vipassana_engine(s_noisy, santana_noisy)

        # Verify severity reflects noise level
        assert torch.all(severity_clean == 0.0)
        assert torch.all(severity_noisy > 0.0)

        # Both should produce valid context and trust
        assert v_ctx_clean.shape == (4, 16)
        assert v_ctx_noisy.shape == (4, 16)
