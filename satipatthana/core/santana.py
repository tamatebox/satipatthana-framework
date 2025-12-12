"""
SantanaLog: Trajectory logging for Satipatthana Framework.

Santāna (Stream of Consciousness) represents the continuous flow of mental states.
This class captures the trajectory of states during the Vicara refinement loop,
enabling Vipassana to analyze the thinking process.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import torch


@dataclass
class SantanaLog:
    """
    Trajectory log for Vicara refinement process.

    Captures the sequence of states, metadata, and energies during convergence.
    Used by Vipassana to assess the quality of the thinking process.

    Attributes:
        states: List of state tensors at each step (Batch, Dim)
        meta_history: List of metadata dictionaries at each step
        energies: List of energy values (stability loss) at each step
    """

    states: List[torch.Tensor] = field(default_factory=list)
    meta_history: List[Dict[str, Any]] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    convergence_steps: Optional[torch.Tensor] = None  # (Batch,) per-sample convergence step

    def add(
        self,
        state: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        energy: Optional[float] = None,
    ) -> None:
        """
        Add a state snapshot to the log.

        Args:
            state: Current state tensor (Batch, Dim)
            meta: Optional metadata dictionary for this step
            energy: Optional energy value (stability loss) for this step
        """
        # Store detached clone to avoid memory issues
        self.states.append(state.detach().clone())

        if meta is not None:
            self.meta_history.append(meta)
        else:
            self.meta_history.append({})

        if energy is not None:
            self.energies.append(energy)

    def to_tensor(self) -> torch.Tensor:
        """
        Convert states list to a single tensor.

        Returns:
            Tensor of shape (num_steps, batch_size, dim) or (num_steps, dim) if batch_size=1
            Returns empty tensor if no states recorded.
        """
        if not self.states:
            return torch.tensor([])

        # Stack along new dimension: (num_steps, batch_size, dim)
        return torch.stack(self.states, dim=0)

    def __len__(self) -> int:
        """Return the number of recorded states."""
        return len(self.states)

    def get_final_state(self) -> Optional[torch.Tensor]:
        """Get the final converged state, or None if empty."""
        if not self.states:
            return None
        return self.states[-1]

    def get_initial_state(self) -> Optional[torch.Tensor]:
        """Get the initial state (s0), or None if empty."""
        if not self.states:
            return None
        return self.states[0]

    def get_total_energy(self) -> float:
        """Get the sum of all recorded energies."""
        return sum(self.energies)

    def get_final_energy(self) -> Optional[float]:
        """Get the final energy value, or None if no energies recorded."""
        if not self.energies:
            return None
        return self.energies[-1]

    def clear(self) -> None:
        """Clear all recorded data."""
        self.states.clear()
        self.meta_history.clear()
        self.energies.clear()

    def to_batch_list(self, batch_size: int) -> List["SantanaLog"]:
        """
        Split batched SantanaLog into individual logs per sample.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            List of SantanaLog objects, one per sample
        """
        result = []
        for b in range(batch_size):
            log = SantanaLog()
            for state in self.states:
                # Extract single sample from batch
                log.states.append(state[b : b + 1])
            log.meta_history = self.meta_history.copy()
            log.energies = self.energies.copy()
            result.append(log)
        return result

    def get_padded_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert trajectory to padded tensor with valid lengths for GRU processing.

        Returns:
            padded_states: (Steps, Batch, Dim) trajectory tensor
            lengths: (Batch,) valid step count per sample (CPU LongTensor for pack_padded_sequence)
        """
        if not self.states:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        # Stack states: (Steps, Batch, Dim)
        trajectory = torch.stack(self.states, dim=0)
        steps, batch_size, _ = trajectory.shape

        if self.convergence_steps is not None:
            # Use recorded convergence steps (requires CPU for pack_padded_sequence)
            lengths = self.convergence_steps.cpu().long()
        else:
            # Fallback: assume all samples use full steps
            lengths = torch.full((batch_size,), steps, dtype=torch.long)

        return trajectory, lengths
