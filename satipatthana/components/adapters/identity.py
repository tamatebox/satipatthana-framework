import torch
from satipatthana.components.adapters.base import BaseAdapter
from satipatthana.configs.adapters import IdentityAdapterConfig


class IdentityAdapter(BaseAdapter):
    """
    Identity Adapter: passes input through unchanged.

    Use this when the input is already in a meaningful feature space
    and no transformation is needed (e.g., low-dimensional point data).

    Args:
        config: Configuration with input_dim (dim is set automatically).
    """

    def __init__(self, config: IdentityAdapterConfig):
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through unchanged.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Same tensor unchanged, shape (batch, dim) where dim == input_dim.
        """
        return x
