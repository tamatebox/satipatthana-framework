from typing import List, Union
import torch.nn as nn
from satipatthana.utils.logger import get_logger

logger = get_logger(__name__)


def freeze_components(model: nn.Module, components_to_freeze: Union[List[str], str]):
    """
    Freezes the parameters of specified components within a model.

    Args:
        model (nn.Module): The model containing components to freeze.
        components_to_freeze (Union[List[str], str]): A list of component names or a single component name to freeze.
                                                      Example: ["adapter", "vitakka", "vicara"]
    """
    if isinstance(components_to_freeze, str):
        components_to_freeze = [components_to_freeze]

    for name, module in model.named_children():
        if name in components_to_freeze:
            logger.info(f"Freezing component: {name}")
            for param in module.parameters():
                param.requires_grad = False
        else:
            logger.info(f"Keeping component unfrozen: {name}")
            for param in module.parameters():
                param.requires_grad = True


def unfreeze_components(model: nn.Module, components_to_unfreeze: Union[List[str], str]):
    """
    Unfreezes the parameters of specified components within a model.

    Args:
        model (nn.Module): The model containing components to unfreeze.
        components_to_unfreeze (Union[List[str], str]): A list of component names or a single component name to unfreeze.
    """
    if isinstance(components_to_unfreeze, str):
        components_to_unfreeze = [components_to_unfreeze]

    for name, module in model.named_children():
        if name in components_to_unfreeze:
            logger.info(f"Unfreezing component: {name}")
            for param in module.parameters():
                param.requires_grad = True


def check_frozen_status(model: nn.Module):
    """
    Displays the frozen status (requires_grad) of each component in the model.
    """
    logger.info("--- Component Freezing Status ---")
    for name, module in model.named_children():
        all_frozen = True
        all_unfrozen = True
        for i, param in enumerate(module.parameters()):
            if i == 0:  # Check only the first parameter to represent component status
                if param.requires_grad:
                    all_frozen = False
                else:
                    all_unfrozen = False
            # If there are mixed states within a module, it's more complex.
            # For simplicity, we assume all params in a module are either frozen or unfrozen by `freeze_components`.

        if not list(module.parameters()):  # Handle modules with no parameters
            logger.info(f"Component '{name}': No trainable parameters.")
        elif all_frozen:
            logger.info(f"Component '{name}': FROZEN")
        elif all_unfrozen:
            logger.info(f"Component '{name}': UNLOCKED")
        else:
            logger.warning(f"Component '{name}': MIXED STATE (investigate manually)")
    logger.info("-------------------------------")
