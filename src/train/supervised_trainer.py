from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.train.base_trainer import BaseSamadhiTrainer


class SupervisedSamadhiTrainer(BaseSamadhiTrainer):
    """
    Supervised learning trainer for the Samadhi Model.
    Trains using pairs of input data (x) and target data (y).
    Characterized by the use of Reconstruction Loss.
    """

    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> float:
        """
        Executes a single training step for one batch.
        Args:
            x (torch.Tensor): Noisy input data (Batch, Dim)
            y (torch.Tensor): Clean target data (Batch, Dim)
        """
        if y is None:
            raise ValueError("Target 'y' cannot be None for SupervisedSamadhiTrainer.")

        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        # ====================================================
        # 2. Forward Pass
        # ====================================================

        # --- A. Search (Vitakka) ---
        # Get s0 (assuming batch support)
        s0, metadata = self.model.vitakka_search(x)

        # Calculate probability distribution for Entropy Loss
        # Performs the same calculation as inside Vitakka to quantify the 'indecision' of probe selection.
        probs = metadata["probs"]

        # --- B. Refine (Vicara) ---
        # Execute while maintaining gradients for training.
        s_t = s0
        batch_stability_loss = torch.tensor(0.0, device=self.device)
        num_steps = self.model.config["refine_steps"]

        if num_steps > 0:
            for _ in range(num_steps):
                s_prev = s_t
                residual = self.model.vicara.refine_step(s_t, metadata)
                # Inertial update (Delegates to centralized logic)
                s_t = self.model.vicara.update_state(s_t, residual)

                # Sum of change amount (L2 norm) for each sample in the batch
                batch_stability_loss += torch.norm(s_t - s_prev, p=2, dim=1).sum()

        s_final = s_t
        decoded_s_final = self.model.decoder(s_final)

        # ====================================================
        # 3. Loss Calculation
        # ====================================================

        # (1) Reconstruction Loss: Has it approached the correct answer?
        # The core of supervised learning: aims for the purified result to match the target.
        recon_loss = nn.MSELoss()(decoded_s_final, y)

        # (2) Stability Loss: Has the mind become unmoving?
        if num_steps > 0:
            # Normalize by batch size and number of steps
            batch_stability_loss = batch_stability_loss / (len(x) * num_steps)

        # (3) Entropy Loss: Was the selection made without hesitation?
        entropy_loss = self._compute_entropy(probs)

        # (4) Load Balancing Loss: Promote diverse probe usage
        balance_loss = self._compute_load_balance_loss(probs)

        # --- Total Loss ---
        # Get coefficients from Config (or default values if not present)
        stability_coeff = self.model.config.get("stability_coeff", 0.01)
        entropy_coeff = self.model.config.get("entropy_coeff", 0.1)
        balance_coeff = self.model.config.get("balance_coeff", 0.001)

        total_loss = (
            recon_loss
            + (stability_coeff * batch_stability_loss)
            + (entropy_coeff * entropy_loss)
            + (balance_coeff * balance_loss)
        )

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def fit(self, dataloader, epochs: int = 5):
        """
        Executes supervised learning for a specified number of epochs.
        The dataloader is expected to return (x, y) pairs.
        """
        self.model.train()

        loss_history = []

        print(f"\n--- Start Supervised Training ({epochs} epochs) ---")
        print(f"Device: {self.device}")
        print(
            f"Params: Stability={self.model.config.get('stability_coeff', 0.01)}, Entropy={self.model.config.get('entropy_coeff', 0.1)}, Balance={self.model.config.get('balance_coeff', 0.001)}"
        )

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for batch_idx, batch_data in enumerate(dataloader):
                # DataLoader format handling: (x, y, ...)
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    x_batch = batch_data[0]
                    y_batch = batch_data[1]
                else:
                    # Error if DataLoader returns a single tensor (since it's supervised learning)
                    raise ValueError("DataLoader must return (input, target) pairs for SupervisedSamadhiTrainer.")

                loss = self.train_step(x_batch, y_batch)
                total_loss += loss
                count += 1

                if batch_idx % 50 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss:.4f}", end="")

            avg_loss = total_loss / count
            loss_history.append(avg_loss)
            print(f"\nEpoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")

        return loss_history
