from typing import Dict, Any, Optional, Union, Tuple, List
import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
from samadhi.train.objectives.base_objective import BaseObjective
from samadhi.utils.logger import get_logger
from samadhi.core.engine import SamadhiEngine  # Import SamadhiEngine
from samadhi.configs.main import SamadhiConfig  # Import SamadhiConfig

logger = get_logger(__name__)


class SamadhiTrainer(Trainer):
    """
    Hugging Face Trainer adapter for Samadhi Model.
    Delegates loss calculation to a generic Objective component.
    """

    def __init__(
        self,
        model: SamadhiEngine = None,  # Changed type hint to SamadhiEngine
        args: Any = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        processing_class: Optional[Any] = None,
        model_init: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Any] = None,
        objective: Optional[BaseObjective] = None,
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
        logger.info(f"Initializing SamadhiTrainer with objective: {objective.__class__.__name__}")
        if objective is None:
            raise ValueError("An 'objective' instance must be provided to SamadhiTrainer.")
        self.objective = objective

        # Ensure model has a SamadhiConfig. This should be set by SamadhiBuilder.
        if not isinstance(model.config, SamadhiConfig):
            raise TypeError("SamadhiTrainer expects model.config to be an instance of SamadhiConfig.")

    def compute_loss(self, model: SamadhiEngine, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overridden compute_loss to use Samadhi's Objective component.
        Dynamically controls the forward pass based on objective's needs_vitakka and needs_vicara flags.
        """
        # HF Trainer passes inputs as a dictionary usually.
        # Samadhi models expect tensor inputs.
        # We assume the collator produces a dict with 'input_values' or similar,
        # or we handle generic keys. For now, let's assume standard (x, y) or just (x) are needed.

        # Extract inputs
        if isinstance(inputs, dict):
            # Try common keys safely (avoiding boolean evaluation of tensors)
            x = inputs.get("input_values")
            if x is None:
                x = inputs.get("pixel_values")
            if x is None:
                x = inputs.get("x")

            y = inputs.get("labels")
            if y is None:
                y = inputs.get("y")

            logger.debug(
                f"Extracted inputs from dict: x={x.shape if x is not None else None}, y={y.shape if y is not None else None}"
            )
        else:
            # Fallback if inputs is tuple/list (less common in HF Trainer but possible with custom collator)
            x = inputs[0]
            y = inputs[1] if len(inputs) > 1 else None
            logger.debug(
                f"Extracted inputs from tuple/list: x={x.shape if x is not None else None}, y={y.shape if y is not None else None}"
            )

        if x is None:
            # If we can't find 'x' by name, and inputs is a dict, maybe the first value is x?
            # This is risky but a common fallback.
            if isinstance(inputs, dict) and len(inputs) > 0:
                x = list(inputs.values())[0]
                if x is None:
                    logger.warning("Input 'x' could not be extracted from inputs.")
            else:
                logger.warning("Input 'x' could not be extracted from inputs (not a dict or empty).")

        # Ensure x is on device
        x = x.to(self.args.device)
        if y is not None:
            y = y.to(self.args.device)

        # --- Forward Pass (using SamadhiEngine's dynamic path selection) ---
        logger.debug(
            f"Forward pass with run_vitakka={self.objective.needs_vitakka}, run_vicara={self.objective.needs_vicara}"
        )
        # Assume 'model' passed here is the SamadhiEngine (or compatible) with dynamic forward.
        output_from_decoder, s_final, metadata = model.forward(
            x, run_vitakka=self.objective.needs_vitakka, run_vicara=self.objective.needs_vicara
        )

        # --- Loss Calculation ---
        # num_refine_steps is only relevant if Vicara was run
        num_refine_steps = (
            model.config.vicara.refine_steps if self.objective.needs_vicara else 0
        )  # Changed config access

        total_loss, loss_components = self.objective.compute_loss(
            x=x,
            y=y,
            s0=metadata.get("s0", s_final),  # If Vitakka is skipped, s0 might not be in meta. Use s_final instead.
            s_final=s_final,
            decoded_s_final=output_from_decoder,
            metadata=metadata,
            num_refine_steps=num_refine_steps,
        )
        logger.debug(f"Total loss: {total_loss.item():.4f}, Components: {loss_components}")

        # Log custom metrics via self.log() if in the main process (for more detailed logging, requires custom callback)
        # For now, just return loss.

        if return_outputs:
            return total_loss, output_from_decoder

        return total_loss
