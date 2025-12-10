"""
Engines: Core processing engines for Samadhi Framework v4.0.

This module contains:
    - SamathaEngine: The "Meditator" - convergence engine
    - VipassanaEngine: The "Observer" - meta-cognition engine
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from samadhi.core.santana import SantanaLog
from samadhi.components.adapters.base import BaseAdapter
from samadhi.components.augmenters.base import BaseAugmenter
from samadhi.components.vitakka.base import BaseVitakka
from samadhi.components.vicara.base import BaseVicara
from samadhi.components.sati.base import BaseSati
from samadhi.components.vipassana.base import BaseVipassana
from samadhi.configs.system import SamathaConfig, VipassanaEngineConfig
from samadhi.utils.logger import get_logger

logger = get_logger(__name__)


class SamathaEngine(nn.Module):
    """
    Samatha Engine (The Meditator) - World Model.

    Converges any input to a "meaningful point" through iterative refinement.
    This engine is task-agnostic and focuses solely on structure extraction.

    Components:
        - Adapter: Raw input -> Latent space
        - Augmenter: Environmental noise injection
        - Vitakka: Search/probing for intentions
        - Vicara: Iterative refinement
        - Sati: Convergence monitoring/gating

    Output:
        - S*: Converged state
        - SantanaLog: Thinking trajectory log
        - severity: Noise intensity (from Augmenter)
    """

    def __init__(
        self,
        config: SamathaConfig,
        adapter: BaseAdapter,
        augmenter: BaseAugmenter,
        vitakka: BaseVitakka,
        vicara: BaseVicara,
        sati: BaseSati,
    ):
        super().__init__()
        self.config = config
        self.adapter = adapter
        self.augmenter = augmenter
        self.vitakka = vitakka
        self.vicara = vicara
        self.sati = sati

        # Drunk mode state
        self._drunk_mode = False
        self._original_dropout_rates: Dict[str, float] = {}

    @property
    def drunk_mode(self) -> bool:
        """Get current drunk mode state."""
        return self._drunk_mode

    @drunk_mode.setter
    def drunk_mode(self, value: bool):
        """Set drunk mode and apply/revert component modifications."""
        if value == self._drunk_mode:
            return

        self._drunk_mode = value
        if value:
            self._enter_drunk_mode()
        else:
            self._exit_drunk_mode()

    def _enter_drunk_mode(self):
        """
        Apply drunk mode modifications to internal components.

        Modifications include:
        - Increased dropout rates
        - Added noise to probe selection
        - Random perturbations to refinement
        """
        logger.debug("Entering drunk mode")

        # Store and increase dropout rates
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                self._original_dropout_rates[name] = module.p
                # Increase dropout to 0.5 or higher
                module.p = max(0.5, module.p * 2.0)

    def _exit_drunk_mode(self):
        """Restore original component states."""
        logger.debug("Exiting drunk mode")

        # Restore original dropout rates
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout) and name in self._original_dropout_rates:
                module.p = self._original_dropout_rates[name]
        self._original_dropout_rates.clear()

    def forward(
        self,
        x: torch.Tensor,
        noise_level: float = 0.0,
        run_augmenter: bool = True,
        drunk_mode: bool = False,
    ) -> Tuple[torch.Tensor, SantanaLog, torch.Tensor]:
        """
        Samatha forward pass: Convergence process.

        Flow: x -> Augment -> Adapt -> Vitakka -> Vicara loop (w/ Sati) -> S*

        Args:
            x: Raw input tensor (Batch, *)
            noise_level: Noise intensity for Augmenter (0.0-1.0)
            run_augmenter: Whether to run augmentation
            drunk_mode: Enable internal dysfunction mode

        Returns:
            s_star: Converged state (Batch, Dim)
            santana: SantanaLog containing thinking trajectory
            severity: Per-sample noise intensity (Batch,)
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        # Set drunk mode for this forward pass
        if drunk_mode != self._drunk_mode:
            self.drunk_mode = drunk_mode

        # 1. Augmentation (environmental noise)
        if run_augmenter and noise_level > 0.0:
            x_aug, severity = self.augmenter(x, noise_level)
        else:
            x_aug = x
            severity = torch.zeros(batch_size, device=device, dtype=dtype)

        # 2. Adaptation: Raw -> Latent space
        z = self.adapter(x_aug)

        # 3. Vitakka: Search for intentions
        s0, vitakka_meta = self.vitakka(z)

        # Add drunk mode perturbation to initial state
        if self._drunk_mode:
            s0 = s0 + torch.randn_like(s0) * 0.1

        # 4. Vicara loop with Sati monitoring
        s_star, santana = self._run_vicara_loop(s0, vitakka_meta)

        # Store initial state info in santana metadata
        if santana.meta_history:
            santana.meta_history[0]["vitakka"] = vitakka_meta

        return s_star, santana, severity

    def _run_vicara_loop(self, s0: torch.Tensor, vitakka_meta: Dict[str, Any]) -> Tuple[torch.Tensor, SantanaLog]:
        """
        Execute Vicara loop with Sati monitoring.

        Args:
            s0: Initial state from Vitakka (Batch, Dim)
            vitakka_meta: Metadata from Vitakka (for ProbeSpecificVicara)

        Returns:
            s_star: Converged state (Batch, Dim)
            santana: Complete trajectory log
        """
        santana = SantanaLog()
        s_t = s0

        # Record initial state
        santana.add(s_t, meta={"step": 0, "type": "initial"})

        for step in range(self.config.max_steps):
            s_prev = s_t.clone()

            # Drunk mode: random skip updates
            if self._drunk_mode and torch.rand(1).item() < 0.2:
                # Skip this step randomly
                energy = 0.0
            else:
                # Single step update via Vicara
                s_t = self.vicara.step(s_t, context=vitakka_meta)

                # Drunk mode: add perturbation after update
                if self._drunk_mode:
                    s_t = s_t + torch.randn_like(s_t) * 0.05

                # Compute energy (state change magnitude)
                energy = torch.norm(s_t - s_prev, dim=1).mean().item()

            # Record state
            santana.add(
                s_t,
                meta={"step": step + 1, "type": "refinement"},
                energy=energy,
            )

            # Check stopping condition via Sati
            should_stop, sati_info = self.sati(s_t, santana)
            if should_stop:
                logger.debug(f"Sati stopped at step {step + 1}: {sati_info.get('reason', 'unknown')}")
                break

        return s_t, santana


class VipassanaEngine(nn.Module):
    """
    Vipassana Engine (The Observer) - Meta-cognition.

    Analyzes the thinking process (SantanaLog) from Samatha to determine
    the quality and confidence of the convergence.

    Output:
        - V_ctx: Context vector (embedding of "doubt/ambiguity")
        - trust_score: Scalar confidence score (0.0-1.0)
    """

    def __init__(
        self,
        config: VipassanaEngineConfig,
        vipassana: BaseVipassana,
    ):
        super().__init__()
        self.config = config
        self.vipassana = vipassana

    def forward(
        self,
        s_star: torch.Tensor,
        santana: SantanaLog,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vipassana forward pass: Introspection.

        Analyzes the converged state and thinking trajectory to produce
        confidence metrics.

        Args:
            s_star: Converged state from Samatha (Batch, Dim)
            santana: SantanaLog containing thinking trajectory

        Returns:
            v_ctx: Context vector (Batch, context_dim) - embedding of "doubt"
            trust_score: Confidence tensor (Batch, 1) for external control
        """
        batch_size = s_star.size(0)
        device = s_star.device
        dtype = s_star.dtype

        # Run Vipassana analysis
        v_ctx, trust_score = self.vipassana(s_star, santana)

        # Ensure trust_score is (Batch, 1) tensor
        # StandardVipassana now returns tensor directly
        if not isinstance(trust_score, torch.Tensor):
            # Backward compatibility: convert scalar to tensor if needed
            trust_score = torch.full(
                (batch_size, 1),
                trust_score,
                device=device,
                dtype=dtype,
            )
        elif trust_score.dim() == 0:
            # Handle 0-d tensor (scalar tensor)
            trust_score = trust_score.expand(batch_size, 1)

        return v_ctx, trust_score


__all__ = ["SamathaEngine", "VipassanaEngine"]
