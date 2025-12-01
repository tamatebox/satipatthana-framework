from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from samadhi.components.adapters.base import BaseAdapter
from samadhi.components.decoders.base import BaseDecoder
from samadhi.components.vitakka.base import BaseVitakka
from samadhi.components.vicara.base import BaseVicara
from samadhi.utils.logger import get_logger
from samadhi.configs.main import SamadhiConfig  # Import SamadhiConfig

logger = get_logger(__name__)


class SamadhiEngine(nn.Module):
    """
    Samadhi Engine (The Core Container).

    Orchestrates the convergence process:
    Raw Input -> Adapter -> [Vitakka -> Sati -> Vicara] -> Decoder -> Output
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        vitakka: BaseVitakka,
        vicara: BaseVicara,
        decoder: BaseDecoder,
        config: SamadhiConfig,  # Changed type hint
    ):
        super().__init__()
        self.adapter = adapter
        self.vitakka = vitakka
        self.vicara = vicara
        self.decoder = decoder
        self.config = config

        # History Log (Citta-santāna)
        self.history_log = []

    def forward(
        self, x: torch.Tensor, run_vitakka: bool = True, run_vicara: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Standard forward pass (Processing a batch).
        Allows dynamic skipping of Vitakka and Vicara components.

        Args:
            x: Raw input (Batch, *)
            run_vitakka (bool): If True, execute the Vitakka (Search) component.
            run_vicara (bool): If True, execute the Vicara (Refinement) component.

        Returns:
            output: Decoder output (Batch, OutDim)
            s_final: Purified state (Batch, Dim) or initial latent state if Vicara is skipped
            meta: Metadata from Vitakka (gate, winner, etc.) or empty dict if Vitakka is skipped
        """
        # 1. Adapter: Raw -> Latent Space
        z = self.adapter(x)

        s: torch.Tensor  # Current latent state
        meta: Dict[str, Any] = {}

        # 2. Vitakka: Search & Initial State
        if run_vitakka:
            # Vitakka returns (s0, meta)
            s, meta = self.vitakka(z)  # Pass adapted latent state
            logger.debug(f"Vitakka run: meta keys={list(meta.keys())}")
        else:
            s = z  # Adapter output directly becomes the latent state

        # 3. Vicara: Refinement (Purification)
        s_final: torch.Tensor
        if run_vicara and run_vitakka:  # Vicara needs Vitakka's output (s0) and meta as context
            s_final, _, _ = self.vicara(s, context=meta)
        elif run_vicara and not run_vitakka:  # If Vitakka is skipped but Vicara is run, use z as s0
            # This case means Vicara is running on raw adapter output. Metadata might be empty.
            s_final, _, _ = self.vicara(s, context={})
            logger.debug("Vicara run without Vitakka context.")
        else:
            s_final = s  # Vicara skipped, final state is the state before Vicara

        # 4. Decoder: Latent -> Output
        output = self.decoder(s_final)

        return output, s_final, meta

    def forward_step(
        self, x_input: torch.Tensor, step_idx: int, run_vitakka: bool = True, run_vicara: bool = True
    ) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Single Time-Step Execution (Search -> Refine -> Log).
        Maintained for compatibility with existing demo logic / iterative inference.
        Allows dynamic skipping of Vitakka and Vicara components.

        Args:
            x_input: (1, *) - Single input
            step_idx: Current step index
            run_vitakka (bool): If True, execute the Vitakka (Search) component.
            run_vicara (bool): If True, execute the Vicara (Refinement) component.

        Returns:
            s_final: Converged state
            full_log: Logs
        """
        # 1. Adapter
        z = self.adapter(x_input)

        s: torch.Tensor
        meta: Dict[str, Any] = {}

        # 2. Vitakka
        if run_vitakka:
            s, meta = self.vitakka(z)  # Pass adapted latent state
            # Gate Check (assuming batch_size=1)
            is_gate_open = meta["gate_open"]
            if isinstance(is_gate_open, torch.Tensor):
                is_gate_open = is_gate_open.item()
            if not is_gate_open:
                logger.info(f"Step {step_idx}: Gate Closed (Input rejected).")
                return None  # Gate Closed
        else:
            s = z  # Adapter output directly becomes the latent state
            # No gate check if Vitakka is skipped
            meta = {"gate_open": True, "winner_id": -1, "confidence": 1.0, "raw_score": 1.0}

        # 3. Vicara
        s_final: torch.Tensor
        if run_vicara and run_vitakka:  # Vicara needs Vitakka's output (s) and meta as context
            s_final, _, energies = self.vicara(s, context=meta)
        elif run_vicara and not run_vitakka:  # If Vitakka skipped but Vicara runs, use z as s0
            s_final, _, energies = self.vicara(s, context={})
        else:
            s_final = s  # Vicara skipped
            energies = {}  # No energies if skipped

        # 4. Logging (Meta-Cognition)
        # Only log meaningful Vitakka metadata if Vitakka was run
        probe_log = {
            "winner_id": meta.get("winner_id", -1),
            "winner_label": meta.get("winner_label", "N/A"),
            "confidence": meta.get("confidence", 1.0),
            "raw_score": meta.get("raw_score", 1.0),
            "gate_open": is_gate_open if run_vitakka else True,
        }

        logger.debug(f"Step {step_idx}: Winner={probe_log['winner_label']}, Conf={probe_log['confidence'].item():.4f}")

        # Skip dynamics if Vitakka was skipped and no meaningful probe_log exists
        dynamics = self._compute_dynamics(probe_log) if run_vitakka else None

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
        """
        Compute state transition dynamics.
        """
        if not self.history_log:
            return None

        prev_log = self.history_log[-1]["probe_log"]

        if current_log["winner_id"] == prev_log["winner_id"]:
            trans_type = "Sustain"
        else:
            trans_type = "Shift"

        labels = self.config.labels  # Changed from .get("labels", [])
        curr_label = current_log.get("winner_label", str(current_log["winner_id"]))
        prev_label = prev_log.get("winner_label", str(prev_log["winner_id"]))

        return {
            "from": prev_label,
            "to": curr_label,
            "type": trans_type,
            "confidence_delta": current_log["confidence"] - prev_log["confidence"],
        }
