"""
SatipatthanaSystem: Top-level system orchestrator for Satipatthana Framework v4.0.

This module contains the SatipatthanaSystem class which integrates:
    - SamathaEngine: Convergence/thinking
    - VipassanaEngine: Meta-cognition/introspection
    - ConditionalDecoder: Task-specific output
    - Reconstruction Heads: Training stabilization
    - AuxiliaryHead: Label guidance (optional)
"""

from enum import IntEnum
from typing import Tuple, Dict, Any, Optional, NamedTuple
import torch
import torch.nn as nn

from satipatthana.core.santana import SantanaLog
from satipatthana.core.engines import SamathaEngine, VipassanaEngine
from satipatthana.configs.system import SystemConfig
from satipatthana.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingStage(IntEnum):
    """Training stages for the 4-stage curriculum."""

    ADAPTER_PRETRAINING = 0  # Stage 0: Adapter pre-training
    SAMATHA_TRAINING = 1  # Stage 1: Samatha training
    VIPASSANA_TRAINING = 2  # Stage 2: Vipassana training
    DECODER_FINETUNING = 3  # Stage 3: Decoder fine-tuning
    INFERENCE = -1  # Inference mode (no training)


class SystemOutput(NamedTuple):
    """Output from SatipatthanaSystem forward pass."""

    output: torch.Tensor  # Task output (Batch, output_dim)
    s_star: torch.Tensor  # Converged state (Batch, dim)
    v_ctx: torch.Tensor  # Context vector (Batch, context_dim)
    trust_score: torch.Tensor  # Trust score (Batch, 1)
    santana: SantanaLog  # Thinking trajectory
    severity: torch.Tensor  # Noise severity (Batch,)
    aux_output: Optional[torch.Tensor] = None  # Auxiliary head output
    recon_adapter: Optional[torch.Tensor] = None  # Adapter reconstruction
    recon_samatha: Optional[torch.Tensor] = None  # Samatha reconstruction


class SatipatthanaSystem(nn.Module):
    """
    Satipatthana System - Top-level orchestrator for v4.0 architecture.

    Integrates all engines and decoders, managing the complete flow:
    1. Samatha: Input -> Convergence -> S*, SantanaLog
    2. Vipassana: S*, SantanaLog -> V_ctx, trust_score
    3. Decoder: S* + V_ctx -> Task output

    Supports 4-stage training curriculum with appropriate freezing.
    """

    def __init__(
        self,
        config: SystemConfig,
        samatha: SamathaEngine,
        vipassana: VipassanaEngine,
        task_decoder: nn.Module,
        adapter_recon_head: Optional[nn.Module] = None,
        samatha_recon_head: Optional[nn.Module] = None,
        auxiliary_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # Core engines
        self.samatha = samatha
        self.vipassana = vipassana

        # Decoders
        self.task_decoder = task_decoder
        self.adapter_recon_head = adapter_recon_head
        self.samatha_recon_head = samatha_recon_head
        self.auxiliary_head = auxiliary_head

        # Training stage (default to inference)
        self._current_stage = TrainingStage.INFERENCE

    @property
    def current_stage(self) -> TrainingStage:
        """Get current training stage."""
        return self._current_stage

    def set_stage(self, stage: TrainingStage) -> None:
        """
        Set training stage and apply appropriate freeze/unfreeze policy.

        Args:
            stage: Target training stage
        """
        self._current_stage = stage
        self._apply_freeze_policy(stage)
        logger.info(f"Training stage set to: {stage.name}")

    def _apply_freeze_policy(self, stage: TrainingStage) -> None:
        """
        Apply freeze/unfreeze policy based on training stage.

        Freeze policy from spec:
        - Stage 0: Train Adapter + adapter_recon_head
        - Stage 1: Train Adapter + Samatha Core + samatha_recon_head + (conditional) AuxHead
        - Stage 2: Train Vipassana only
        - Stage 3: Train TaskDecoder only
        """
        # First, freeze everything
        self._freeze_all()

        if stage == TrainingStage.ADAPTER_PRETRAINING:
            # Stage 0: Train Adapter + adapter_recon_head
            self._unfreeze_module(self.samatha.adapter)
            if self.adapter_recon_head is not None:
                self._unfreeze_module(self.adapter_recon_head)

        elif stage == TrainingStage.SAMATHA_TRAINING:
            # Stage 1: Train Adapter + Samatha Core + samatha_recon_head + (conditional) AuxHead
            self._unfreeze_module(self.samatha.adapter)
            self._unfreeze_module(self.samatha.vitakka)
            self._unfreeze_module(self.samatha.vicara)
            self._unfreeze_module(self.samatha.sati)
            if self.samatha_recon_head is not None:
                self._unfreeze_module(self.samatha_recon_head)
            if self.config.use_label_guidance and self.auxiliary_head is not None:
                self._unfreeze_module(self.auxiliary_head)

        elif stage == TrainingStage.VIPASSANA_TRAINING:
            # Stage 2: Train Vipassana only
            self._unfreeze_module(self.vipassana)

        elif stage == TrainingStage.DECODER_FINETUNING:
            # Stage 3: Train TaskDecoder only
            self._unfreeze_module(self.task_decoder)

        # INFERENCE: everything frozen

    def _freeze_all(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module: nn.Module) -> None:
        """Unfreeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        noise_level: float = 0.0,
        drunk_mode: bool = False,
        run_vipassana: bool = True,
        run_decoder: bool = True,
    ) -> SystemOutput:
        """
        Full system forward pass.

        Args:
            x: Raw input (Batch, *)
            noise_level: Augmentation noise level (0.0-1.0)
            drunk_mode: Enable Samatha drunk mode
            run_vipassana: Whether to run Vipassana (can skip for Stage 0/1)
            run_decoder: Whether to run task decoder

        Returns:
            SystemOutput containing all outputs and intermediate states
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        # 1. Samatha: Convergence
        samatha_output = self.samatha(x, noise_level=noise_level, drunk_mode=drunk_mode)
        s_star = samatha_output.s_star
        santana = samatha_output.santana
        severity = samatha_output.severity

        # 2. Vipassana: Introspection (optional)
        if run_vipassana:
            v_ctx, trust_score = self.vipassana(s_star, santana)
        else:
            # Return dummy context and full trust
            context_dim = self.config.vipassana.vipassana.context_dim
            v_ctx = torch.zeros(batch_size, context_dim, device=device, dtype=dtype)
            trust_score = torch.ones(batch_size, 1, device=device, dtype=dtype)

        # 3. Task Decoder (optional)
        if run_decoder:
            # ConditionalDecoder expects concatenated input
            s_and_ctx = torch.cat([s_star, v_ctx], dim=1)
            output = self.task_decoder(s_and_ctx)
        else:
            output = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # 4. Auxiliary outputs (for training)
        aux_output = None
        if self.auxiliary_head is not None and self.training:
            aux_output = self.auxiliary_head(s_star)

        recon_adapter = None
        if self.adapter_recon_head is not None and self.training:
            # Adapter reconstruction from latent z (before Vitakka)
            z = self.samatha.adapter(x)
            recon_adapter = self.adapter_recon_head(z)

        recon_samatha = None
        if self.samatha_recon_head is not None and self.training:
            recon_samatha = self.samatha_recon_head(s_star)

        return SystemOutput(
            output=output,
            s_star=s_star,
            v_ctx=v_ctx,
            trust_score=trust_score,
            santana=santana,
            severity=severity,
            aux_output=aux_output,
            recon_adapter=recon_adapter,
            recon_samatha=recon_samatha,
        )

    def forward_stage0(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 0: Adapter pre-training forward.

        Only runs Adapter -> Reconstruction.

        Args:
            x: Raw input (Batch, *)

        Returns:
            z: Latent representation (Batch, dim)
            x_recon: Reconstructed input (Batch, *)
        """
        z = self.samatha.adapter(x)

        if self.adapter_recon_head is not None:
            x_recon = self.adapter_recon_head(z)
        else:
            x_recon = z  # No reconstruction head

        return z, x_recon

    def forward_stage1(
        self,
        x: torch.Tensor,
        noise_level: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Stage 1: Samatha training forward.

        Runs full Samatha but skips Vipassana and TaskDecoder.

        Args:
            x: Raw input (Batch, *)
            noise_level: Augmentation noise level

        Returns:
            Dictionary with s_star, santana, severity, stability_pair, x_recon, aux_output (if applicable)
        """
        # Run Samatha - returns SamathaOutput with stability_pair
        samatha_output = self.samatha(x, noise_level=noise_level)

        result = {
            "s_star": samatha_output.s_star,
            "santana": samatha_output.santana,
            "severity": samatha_output.severity,
            "stability_pair": samatha_output.stability_pair,
        }

        # Samatha reconstruction
        if self.samatha_recon_head is not None:
            result["x_recon"] = self.samatha_recon_head(samatha_output.s_star)

        # Auxiliary head for label guidance
        if self.config.use_label_guidance and self.auxiliary_head is not None:
            result["aux_output"] = self.auxiliary_head(samatha_output.s_star)

        return result

    def forward_stage2(
        self,
        x: torch.Tensor,
        noise_level: float = 0.0,
        drunk_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Stage 2: Vipassana training forward.

        Runs Samatha (frozen) + Vipassana.

        Args:
            x: Raw input (Batch, *)
            noise_level: Augmentation noise level
            drunk_mode: Enable drunk mode for negative samples

        Returns:
            Dictionary with s_star, santana, v_ctx, trust_score
        """
        # Run Samatha (frozen in Stage 2)
        with torch.no_grad():
            samatha_output = self.samatha(x, noise_level=noise_level, drunk_mode=drunk_mode)
            s_star_detached = samatha_output.s_star
            santana = samatha_output.santana
            severity = samatha_output.severity

        # Clone s_star and enable gradients for Vipassana's input
        # This allows Vipassana to compute gradients through its own parameters
        s_star = s_star_detached.clone().requires_grad_(True)

        # Run Vipassana (trainable)
        v_ctx, trust_score = self.vipassana(s_star, santana)

        return {
            "s_star": s_star,
            "santana": santana,
            "severity": severity,
            "v_ctx": v_ctx,
            "trust_score": trust_score,
        }

    def forward_stage3(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Stage 3: Decoder fine-tuning forward.

        Runs full pipeline but only TaskDecoder is trainable.

        Args:
            x: Raw input (Batch, *)

        Returns:
            Dictionary with output, s_star, v_ctx, trust_score
        """
        # Run Samatha (frozen)
        with torch.no_grad():
            samatha_output = self.samatha(x)
            s_star = samatha_output.s_star
            santana = samatha_output.santana

        # Run Vipassana (frozen)
        with torch.no_grad():
            v_ctx, trust_score = self.vipassana(s_star, santana)

        # Run TaskDecoder (trainable)
        s_and_ctx = torch.cat([s_star, v_ctx], dim=1)
        output = self.task_decoder(s_and_ctx)

        return {
            "output": output,
            "s_star": s_star,
            "v_ctx": v_ctx,
            "trust_score": trust_score,
        }

    def inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode forward pass.

        Args:
            x: Raw input (Batch, *)

        Returns:
            output: Task output (Batch, output_dim)
            trust_score: Confidence score (Batch, 1)
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(x)
        return result.output, result.trust_score

    def get_trainable_params(self) -> Dict[str, int]:
        """Get count of trainable parameters by component."""
        counts = {}

        def count_params(module, name):
            total = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if total > 0:
                counts[name] = total

        count_params(self.samatha.adapter, "adapter")
        count_params(self.samatha.vitakka, "vitakka")
        count_params(self.samatha.vicara, "vicara")
        count_params(self.samatha.sati, "sati")
        count_params(self.vipassana, "vipassana")
        count_params(self.task_decoder, "task_decoder")

        if self.adapter_recon_head is not None:
            count_params(self.adapter_recon_head, "adapter_recon_head")
        if self.samatha_recon_head is not None:
            count_params(self.samatha_recon_head, "samatha_recon_head")
        if self.auxiliary_head is not None:
            count_params(self.auxiliary_head, "auxiliary_head")

        return counts


__all__ = ["SatipatthanaSystem", "TrainingStage", "SystemOutput"]
