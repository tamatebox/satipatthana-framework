from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from src.components.adapters.base import BaseAdapter
from src.components.decoders.base import BaseDecoder
from src.components.vitakka import Vitakka
from src.components.vicara import VicaraBase


class SamadhiEngine(nn.Module):
    """
    Samadhi Engine (The Core Container).

    Orchestrates the convergence process:
    Raw Input -> Adapter -> [Vitakka -> Sati -> Vicara] -> Decoder -> Output
    """

    def __init__(
        self, adapter: BaseAdapter, vitakka: Vitakka, vicara: VicaraBase, decoder: BaseDecoder, config: Dict[str, Any]
    ):
        super().__init__()
        self.adapter = adapter
        self.vitakka = vitakka
        self.vicara = vicara
        self.decoder = decoder
        self.config = config

        # History Log (Citta-santāna)
        self.history_log = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Standard forward pass (Processing a batch).

        Args:
            x: Raw input (Batch, *)

        Returns:
            output: Decoder output (Batch, OutDim)
            s_final: Purified state (Batch, Dim)
            meta: Metadata from Vitakka (gate, winner, etc.)
        """
        # 1. Adapter: Raw -> Latent Space
        z = self.adapter(x)

        # 2. Vitakka: Search & Initial State
        # Vitakka returns (s0, meta)
        s0, meta = self.vitakka(z)

        # 3. Vicara: Refinement (Purification)
        # Vicara returns (s_final, trajectory, energies)
        # We pass 'meta' as context (e.g. for weighted refinement)
        s_final, _, _ = self.vicara(s0, context=meta)

        # 4. Decoder: Latent -> Output
        output = self.decoder(s_final)

        return output, s_final, meta

    def forward_step(self, x_input: torch.Tensor, step_idx: int) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Single Time-Step Execution (Search -> Refine -> Log).
        Maintained for compatibility with existing demo logic / iterative inference.

        Args:
            x_input: (1, *) - Single input
            step_idx: Current step index

        Returns:
            s_final: Converged state
            full_log: Logs
        """
        # 1. Adapter
        z = self.adapter(x_input)

        # 2. Vitakka
        s0, meta = self.vitakka(z)

        # Gate Check (assuming batch_size=1)
        is_gate_open = meta["gate_open"]
        if isinstance(is_gate_open, torch.Tensor):
            is_gate_open = is_gate_open.item()

        if not is_gate_open:
            return None  # Gate Closed

        # 3. Vicara
        s_final, _, energies = self.vicara(s0, context=meta)

        # 4. Logging (Meta-Cognition)
        # Extract single item from batch metadata for logging
        winner_id = meta["winner_id"].item() if isinstance(meta["winner_id"], torch.Tensor) else meta["winner_id"]

        # Label resolution
        labels = self.config.get("labels", [])
        winner_label = labels[winner_id] if isinstance(winner_id, int) and winner_id < len(labels) else str(winner_id)

        probe_log = {
            "winner_id": winner_id,
            "winner_label": winner_label,
            "confidence": (
                meta["confidence"].item() if isinstance(meta["confidence"], torch.Tensor) else meta["confidence"]
            ),
            "raw_score": (
                meta["raw_score"].item() if isinstance(meta["raw_score"], torch.Tensor) else meta["raw_score"]
            ),
            "gate_open": is_gate_open,
        }

        dynamics = self._compute_dynamics(probe_log)

        full_log = {
            "step": step_idx,
            "probe_log": probe_log,
            "dynamics": dynamics,
            "energies": energies,
            "s_norm": torch.norm(s_final).item(),
        }

        self.history_log.append(full_log)

        return s_final, full_log

    def _compute_dynamics(self, current_log: Dict) -> Optional[Dict]:
        """Compute state transition dynamics."""
        if not self.history_log:
            return None

        prev_log = self.history_log[-1]["probe_log"]

        if current_log["winner_id"] == prev_log["winner_id"]:
            trans_type = "Sustain"
        else:
            trans_type = "Shift"

        labels = self.config.get("labels", [])
        curr_label = current_log.get("winner_label", str(current_log["winner_id"]))
        prev_label = prev_log.get("winner_label", str(prev_log["winner_id"]))

        return {
            "from": prev_label,
            "to": curr_label,
            "type": trans_type,
            "confidence_delta": current_log["confidence"] - prev_log["confidence"],
        }
