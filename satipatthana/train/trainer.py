"""
SatipatthanaTrainer v4: 4-Stage Curriculum Trainer.

This trainer implements the 4-stage curriculum learning for Satipatthana v4.0:
- Stage 0: Adapter Pre-training (Reconstruction)
- Stage 1: Samatha Training (Stability + optional Label Guidance)
- Stage 2: Vipassana Training (Contrastive Learning)
- Stage 3: Decoder Fine-tuning (Task-specific)

Each stage has specific freeze/unfreeze policies and objectives.
"""

from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from enum import IntEnum
import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from satipatthana.core.system import SatipatthanaSystem, TrainingStage
from satipatthana.core.santana import SantanaLog
from satipatthana.data import VoidDataset
from satipatthana.components.objectives.vipassana import (
    VipassanaObjective,
    GuidanceLoss,
    ProbeDiversityLoss,
    StabilityLoss,
)
from satipatthana.utils.logger import get_logger

logger = get_logger(__name__)


class Stage2NoiseStrategy(IntEnum):
    """Noise generation strategies for Stage 2 Vipassana Training."""

    AUGMENTED = 0  # Environmental noise via Augmenter
    DRUNK = 1  # Internal dysfunction via drunk_mode
    MISMATCH = 2  # Logical inconsistency via batch shuffling
    VOID = 3  # Out-of-distribution data (unknown territory)


class SatipatthanaTrainer(Trainer):
    """
    Hugging Face Trainer adapted for Satipatthana v4.0 4-stage curriculum.

    This trainer manages:
    - Stage switching and freeze/unfreeze policies
    - Stage-specific forward passes
    - Noise generation for Stage 2 (Augmented, Drunk, Mismatch)
    - Loss computation per stage

    Args:
        model: SatipatthanaSystem instance
        args: HuggingFace TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        stage: Training stage (from TrainingStage enum)
        noise_level: Noise intensity for Augmenter (Stage 2)
        use_label_guidance: Whether to use label guidance in Stage 1
        task_type: "classification" or "regression" for GuidanceLoss
        vipassana_margin: Margin for contrastive loss (Stage 2)
        stability_weight: Weight for stability loss (Stage 1)
        guidance_weight: Weight for guidance loss (Stage 1)
        recon_weight: Weight for reconstruction loss (Stage 0, 1)
        callbacks: Optional list of callbacks
        void_dataset: Optional VoidDataset for OOD data in Void Path (Stage 2).
                     Use VoidDataset from satipatthana.data to wrap any data source.
                     If None, Void Path is disabled.
        class_weight: Optional class weights for CrossEntropyLoss (Stage 3).
                     Tensor of shape (n_classes,) for handling imbalanced datasets.
    """

    def __init__(
        self,
        model: SatipatthanaSystem = None,
        args: TrainingArguments = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        processing_class: Optional[Any] = None,
        model_init: Optional[Any] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Any] = None,
        # Stage-specific parameters
        stage: TrainingStage = TrainingStage.INFERENCE,
        noise_level: float = 0.3,
        use_label_guidance: bool = False,
        task_type: str = "classification",
        vipassana_margin: float = 0.5,
        stability_weight: float = 0.1,
        guidance_weight: float = 1.0,
        recon_weight: float = 1.0,
        diversity_weight: float = 0.1,
        label_key: str = "y",
        void_dataset: Optional[VoidDataset] = None,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not isinstance(model, SatipatthanaSystem):
            raise TypeError("SatipatthanaTrainer requires model to be a SatipatthanaSystem instance.")

        # Store stage-specific configuration
        self._stage = stage
        self.noise_level = noise_level
        self.use_label_guidance = use_label_guidance
        self.vipassana_margin = vipassana_margin
        self.stability_weight = stability_weight
        self.guidance_weight = guidance_weight
        self.recon_weight = recon_weight
        self.diversity_weight = diversity_weight
        self.label_key = label_key

        # Initialize objectives
        self.vipassana_objective = VipassanaObjective()
        self.guidance_loss = GuidanceLoss(task_type=task_type)
        self.stability_loss = StabilityLoss()
        self.diversity_loss = ProbeDiversityLoss()
        self.recon_loss_fn = nn.MSELoss()
        self.class_weight = class_weight
        if task_type == "classification":
            self.task_loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.task_loss_fn = nn.MSELoss()
        self.task_type = task_type

        # Void Path configuration for Stage 2
        self.void_dataset = void_dataset
        self._void_indices: Optional[torch.Tensor] = None  # For random sampling from void_dataset

        # Set initial stage
        self.set_stage(stage)

        logger.info(f"Initialized SatipatthanaTrainer for stage: {stage.name}")

    @property
    def stage(self) -> TrainingStage:
        """Get current training stage."""
        return self._stage

    def set_stage(self, stage: TrainingStage):
        """
        Switch to a new training stage.

        This will:
        1. Update the current stage
        2. Apply freeze/unfreeze policies via SatipatthanaSystem

        Args:
            stage: Target training stage
        """
        self._stage = stage
        self.model.set_stage(stage)

        # Log trainable parameters
        trainable = self.model.get_trainable_params()
        logger.info(f"Stage {stage.name}: Trainable components: {list(trainable.keys())}")

    def _extract_inputs(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract x and y from input dictionary."""
        if isinstance(inputs, dict):
            # Check each key explicitly to avoid tensor boolean evaluation
            x = inputs.get("x")
            if x is None:
                x = inputs.get("input_values")
            if x is None:
                x = inputs.get("pixel_values")

            y = inputs.get(self.label_key)
            if y is None:
                y = inputs.get("labels")
        else:
            x = inputs[0]
            y = inputs[1] if len(inputs) > 1 else None

        if x is None:
            raise ValueError("Could not extract input 'x' from inputs")

        x = x.to(self.args.device)
        if y is not None:
            y = y.to(self.args.device)

        return x, y

    def compute_loss(
        self,
        model: SatipatthanaSystem,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss based on current training stage.

        Dispatches to stage-specific loss computation methods.
        """
        x, y = self._extract_inputs(inputs)

        if self._stage == TrainingStage.ADAPTER_PRETRAINING:
            loss, outputs = self._compute_stage0_loss(model, x)
        elif self._stage == TrainingStage.SAMATHA_TRAINING:
            loss, outputs = self._compute_stage1_loss(model, x, y)
        elif self._stage == TrainingStage.VIPASSANA_TRAINING:
            loss, outputs = self._compute_stage2_loss(model, x)
        elif self._stage == TrainingStage.DECODER_FINETUNING:
            loss, outputs = self._compute_stage3_loss(model, x, y)
        else:
            raise ValueError(f"Cannot compute loss for stage: {self._stage}")

        if return_outputs:
            return loss, outputs
        return loss

    def _compute_stage0_loss(self, model: SatipatthanaSystem, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Stage 0: Adapter Pre-training.

        Loss: Reconstruction (x -> Adapter -> z -> ReconHead -> x_hat)
        """
        z, x_recon = model.forward_stage0(x)

        loss = self.recon_loss_fn(x_recon, x)

        outputs = {
            "z": z,
            "x_recon": x_recon,
            "recon_loss": loss.item(),
        }

        logger.debug(f"Stage 0 loss: {loss.item():.4f}")
        return loss, outputs

    def _compute_stage1_loss(
        self, model: SatipatthanaSystem, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Stage 1: Samatha Training.

        Loss: Stability + Reconstruction + (optional) Label Guidance
        """
        result = model.forward_stage1(x)

        s_star = result["s_star"]
        santana = result["santana"]
        stability_pair = result["stability_pair"]
        x_recon = result.get("x_recon")
        aux_output = result.get("aux_output")

        loss_components = {}
        total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        # Stability loss (now differentiable via stability_pair)
        stability_loss, stability_components = self.stability_loss.compute_loss(stability_pair, santana)
        loss_components.update(stability_components)
        total_loss = total_loss + self.stability_weight * stability_loss

        # Reconstruction loss (if reconstruction head is available)
        if x_recon is not None:
            recon_loss = self.recon_loss_fn(x_recon, x)
            loss_components["recon_loss"] = recon_loss.item()
            total_loss = total_loss + self.recon_weight * recon_loss

        # Label guidance loss (if enabled and labels available)
        if self.use_label_guidance and y is not None and aux_output is not None:
            guidance_loss, guidance_components = self.guidance_loss.compute_loss(aux_output, y)
            loss_components.update(guidance_components)
            total_loss = total_loss + self.guidance_weight * guidance_loss

        # Probe diversity loss (prevent mode collapse)
        if self.diversity_weight > 0:
            probes = model.samatha.vitakka.probes
            diversity_loss, diversity_components = self.diversity_loss.compute_loss(probes)
            loss_components.update(diversity_components)
            total_loss = total_loss + self.diversity_weight * diversity_loss

        outputs = {
            "s_star": s_star,
            "santana": santana,
            "x_recon": x_recon,
            "aux_output": aux_output,
            **loss_components,
        }

        logger.debug(f"Stage 1 loss: {total_loss.item():.4f}, components: {loss_components}")
        return total_loss, outputs

    def _sample_void_data(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Sample OOD data from void_dataset for Void Path.

        Args:
            batch_size: Number of OOD samples to sample
            device: Target device for the tensor

        Returns:
            OOD tensor of shape (batch_size, *input_shape), or None if void_dataset is not set
        """
        if self.void_dataset is None:
            return None

        # Random sampling from VoidDataset (always returns {"x": tensor})
        dataset_len = len(self.void_dataset)
        indices = torch.randint(0, dataset_len, (batch_size,))

        samples = [self.void_dataset[idx.item()]["x"] for idx in indices]
        return torch.stack(samples).to(device)

    def _compute_stage2_loss(self, model: SatipatthanaSystem, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Stage 2: Vipassana Training with Triple Score.

        Uses up to 5 noise strategies to generate contrastive pairs:
        1. Clean path (No noise) - target: 1.0
        2. Augmented path (Environmental Ambiguity) - target: 1.0 - severity
        3. Drunk path (Internal Dysfunction) - target: 0.0
        4. Mismatch path (Logical Inconsistency) - target: 0.0
        5. Void path (OOD data from void_dataset) - target: 0.0 (only if void_dataset is set)

        All three scores (trust, conformity, confidence) are trained with the same targets.
        """
        batch_size = x.size(0)
        device = x.device

        # Split ID data into 4 parts (Clean, Augmented, Drunk, Mismatch)
        # Void data is sampled separately from void_dataset
        num_id_paths = 4
        split_size = batch_size // num_id_paths
        remainder = batch_size % num_id_paths
        id_sizes = [split_size + (1 if i < remainder else 0) for i in range(num_id_paths)]

        # Void size matches average ID split size (only if void_dataset is set)
        void_size = split_size if self.void_dataset is not None else 0

        x_splits = torch.split(x, id_sizes)  # Split ID data

        all_trust_scores = []
        all_conformity_scores = []
        all_confidence_scores = []
        all_targets = []

        # 1. Clean Path (No noise - baseline for high trust)
        if id_sizes[0] > 0:
            x_clean = x_splits[0]
            result_clean = model.forward_stage2(x_clean, noise_level=0.0, drunk_mode=False)
            # Target: 1.0 (clean input should produce high scores)
            target_clean = torch.ones_like(result_clean["trust_score"])

            all_trust_scores.append(result_clean["trust_score"])
            all_conformity_scores.append(result_clean["conformity_score"])
            all_confidence_scores.append(result_clean["confidence_score"])
            all_targets.append(target_clean)

        # 2. Augmented Path (Environmental Ambiguity)
        if id_sizes[1] > 0:
            x_aug = x_splits[1]
            result_aug = model.forward_stage2(x_aug, noise_level=self.noise_level, drunk_mode=False)
            # Target: 1.0 - severity (higher noise = lower trust target)
            severity_aug = result_aug.get("severity", torch.zeros(id_sizes[1], device=device))
            target_aug = 1.0 - severity_aug.unsqueeze(-1)

            all_trust_scores.append(result_aug["trust_score"])
            all_conformity_scores.append(result_aug["conformity_score"])
            all_confidence_scores.append(result_aug["confidence_score"])
            all_targets.append(target_aug)

        # 3. Drunk Path (Internal Dysfunction)
        if id_sizes[2] > 0:
            x_drunk = x_splits[2]
            result_drunk = model.forward_stage2(x_drunk, noise_level=0.0, drunk_mode=True)
            # Target: 0.0 (drunk mode should produce low scores)
            target_drunk = torch.zeros_like(result_drunk["trust_score"])

            all_trust_scores.append(result_drunk["trust_score"])
            all_conformity_scores.append(result_drunk["conformity_score"])
            all_confidence_scores.append(result_drunk["confidence_score"])
            all_targets.append(target_drunk)

        # 4. Mismatch Path (Logical Inconsistency)
        if id_sizes[3] > 0:
            x_mismatch = x_splits[3]
            # First get normal S* and SantanaLog
            with torch.no_grad():
                samatha_output = model.samatha(x_mismatch)
                s_star_normal = samatha_output.s_star
                santana_normal = samatha_output.santana

            # Shuffle S* within batch to create mismatch
            perm = torch.randperm(id_sizes[3], device=device)
            s_star_shuffled = s_star_normal[perm]

            # Pass mismatched (S*, SantanaLog) to Vipassana
            # Get probes for semantic feature computation
            probes = model.samatha.vitakka.probes
            vipassana_output = model.vipassana(s_star_shuffled, santana_normal, probes=probes)

            # Target: 0.0 (mismatch should produce low scores)
            target_mismatch = torch.zeros_like(vipassana_output.trust_score)

            all_trust_scores.append(vipassana_output.trust_score)
            all_conformity_scores.append(vipassana_output.conformity_score)
            all_confidence_scores.append(vipassana_output.confidence_score)
            all_targets.append(target_mismatch)

        # 5. Void Path (OOD data from void_dataset)
        if void_size > 0:
            x_void = self._sample_void_data(void_size, device)
            if x_void is not None:
                result_void = model.forward_stage2(x_void, noise_level=0.0, drunk_mode=False)
                # Target: 0.0 (OOD input should produce low scores)
                target_void = torch.zeros_like(result_void["trust_score"])

                all_trust_scores.append(result_void["trust_score"])
                all_conformity_scores.append(result_void["conformity_score"])
                all_confidence_scores.append(result_void["confidence_score"])
                all_targets.append(target_void)

        # Concatenate all results
        trust_scores = torch.cat(all_trust_scores, dim=0)
        conformity_scores = torch.cat(all_conformity_scores, dim=0)
        confidence_scores = torch.cat(all_confidence_scores, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Compute Triple Score BCE loss
        loss, loss_components = self.vipassana_objective.compute_loss(
            trust_scores, targets, conformity_scores, confidence_scores
        )

        outputs = {
            "trust_scores": trust_scores,
            "conformity_scores": conformity_scores,
            "confidence_scores": confidence_scores,
            "targets": targets,
            **loss_components,
        }

        logger.debug(f"Stage 2 loss: {loss.item():.4f}, components: {loss_components}")
        return loss, outputs

    def _compute_stage3_loss(
        self, model: SatipatthanaSystem, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Stage 3: Decoder Fine-tuning.

        Loss: Task-specific (CrossEntropy for classification, MSE for regression)
        """
        if y is None:
            raise ValueError("Stage 3 requires labels (y)")

        result = model.forward_stage3(x)
        output = result["output"]

        # Compute task loss
        if self.task_type == "classification":
            loss = self.task_loss_fn(output, y)
        else:
            # Regression: reshape targets if needed
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            loss = self.task_loss_fn(output, y)

        # Compute accuracy/metrics for logging
        if self.task_type == "classification":
            preds = output.argmax(dim=-1)
            accuracy = (preds == y).float().mean().item()
        else:
            accuracy = None

        outputs = {
            "output": output,
            "s_star": result["s_star"],
            "v_ctx": result["v_ctx"],
            "trust_score": result["trust_score"],
            "task_loss": loss.item(),
        }
        if accuracy is not None:
            outputs["accuracy"] = accuracy

        logger.debug(f"Stage 3 loss: {loss.item():.4f}")
        return loss, outputs

    def train_stage(self, stage: TrainingStage, num_epochs: int = 1, **kwargs):
        """
        Convenience method to train a specific stage.

        Args:
            stage: Stage to train
            num_epochs: Number of epochs for this stage
            **kwargs: Additional arguments passed to train()
        """
        # For Stage 2 (Vipassana Training), we need to initialize the lazy networks
        # before setting the stage, otherwise the parameters won't exist yet
        if stage == TrainingStage.VIPASSANA_TRAINING:
            self._initialize_vipassana_networks()

        self.set_stage(stage)

        # Reset optimizer and scheduler to use new trainable parameters
        # This is critical when switching stages as different parameters are trainable
        self.optimizer = None
        self.lr_scheduler = None

        # Update training arguments for this stage
        original_epochs = self.args.num_train_epochs
        self.args.num_train_epochs = num_epochs

        logger.info(f"Starting {stage.name} training for {num_epochs} epochs")

        try:
            result = self.train(**kwargs)
        finally:
            # Restore original epochs
            self.args.num_train_epochs = original_epochs

        return result

    def _initialize_vipassana_networks(self):
        """
        Initialize Vipassana's lazy networks by running a dummy forward pass.

        StandardVipassana uses lazy initialization - the encoder and trust_head
        networks are only created on the first forward pass when the state
        dimension is known. This method ensures they exist before set_stage()
        is called, so the parameters can be properly unfrozen.
        """
        # Get a sample batch from the dataset to determine dimensions
        sample = self.train_dataset[0]
        if isinstance(sample, dict):
            x = sample.get("x")
            if x is None:
                x = sample.get("input_values")
            if x is None:
                x = sample.get("pixel_values")
        else:
            x = sample[0]

        if x is None:
            logger.warning("Could not get sample input for Vipassana initialization")
            return

        # Add batch dimension if needed
        if x.dim() == len(x.shape):
            x = x.unsqueeze(0)

        x = x.to(self.args.device)

        # Run a forward pass to initialize Vipassana networks
        with torch.no_grad():
            try:
                self.model.forward_stage2(x, noise_level=0.0, drunk_mode=False)
                logger.info("Vipassana networks initialized via forward pass")
            except Exception as e:
                logger.warning(f"Failed to initialize Vipassana networks: {e}")

    def run_curriculum(
        self, stage0_epochs: int = 0, stage1_epochs: int = 10, stage2_epochs: int = 5, stage3_epochs: int = 5, **kwargs
    ):
        """
        Run the full 4-stage curriculum.

        Args:
            stage0_epochs: Epochs for Adapter Pre-training (0 to skip)
            stage1_epochs: Epochs for Samatha Training
            stage2_epochs: Epochs for Vipassana Training
            stage3_epochs: Epochs for Decoder Fine-tuning
            **kwargs: Additional arguments passed to train()

        Returns:
            Dict with results from each stage
        """
        results = {}

        if stage0_epochs > 0:
            logger.info("=== Stage 0: Adapter Pre-training ===")
            results["stage0"] = self.train_stage(TrainingStage.ADAPTER_PRETRAINING, num_epochs=stage0_epochs, **kwargs)

        if stage1_epochs > 0:
            logger.info("=== Stage 1: Samatha Training ===")
            results["stage1"] = self.train_stage(TrainingStage.SAMATHA_TRAINING, num_epochs=stage1_epochs, **kwargs)

        if stage2_epochs > 0:
            logger.info("=== Stage 2: Vipassana Training ===")
            results["stage2"] = self.train_stage(TrainingStage.VIPASSANA_TRAINING, num_epochs=stage2_epochs, **kwargs)

        if stage3_epochs > 0:
            logger.info("=== Stage 3: Decoder Fine-tuning ===")
            results["stage3"] = self.train_stage(TrainingStage.DECODER_FINETUNING, num_epochs=stage3_epochs, **kwargs)

        # Set to inference mode after training
        self.set_stage(TrainingStage.INFERENCE)

        logger.info("=== Curriculum Training Complete ===")
        return results


__all__ = ["SatipatthanaTrainer", "Stage2NoiseStrategy"]
