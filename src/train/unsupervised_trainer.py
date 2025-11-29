from typing import Optional
import torch
import torch.nn.functional as F
from src.train.base_trainer import BaseSamadhiTrainer


class UnsupervisedSamadhiTrainer(BaseSamadhiTrainer):
    """
    Unsupervised learning trainer for the Samadhi Model.
    Trains using only input data (x), ignoring target data (y).
    Minimizes only Stability Loss and Entropy Loss.
    This allows the model to self-organize "spontaneously stable attractors (concepts)".
    """

    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> float:
        """
        Executes a single training step for one batch.
        Args:
            x (torch.Tensor): Input data (Batch, Dim)
            y (torch.Tensor, Optional): Ignored.
        """
        x = x.to(self.device)
        # y is ignored

        self.optimizer.zero_grad()

        # ====================================================
        # 2. Forward Pass
        # ====================================================

        # --- A. Search (Vitakka) ---
        # Get s0
        s0, metadata = self.model.vitakka_search(x)

        # Calculate probability distribution for Entropy Loss
        # Retrieved from metadata.
        probs = metadata["probs"]

        # --- B. Refine (Vicara) ---
        s_t = s0
        batch_stability_loss = torch.tensor(0.0, device=self.device)
        num_steps = self.model.config["refine_steps"]

        if num_steps > 0:
            for _ in range(num_steps):
                s_prev = s_t
                residual = self.model.vicara.refine_step(s_t, metadata)
                # Inertial update
                s_t = 0.7 * s_t + 0.3 * residual

                # Sum of change amount (L2 norm) for each sample in the batch
                batch_stability_loss += torch.norm(s_t - s_prev, p=2, dim=1).sum()

        # ====================================================
        # 3. Loss Calculation
        # ====================================================

        # (1) Reconstruction Loss is not used (as there is no target).

        # (2) Stability Loss: Has the mind become unmoving?
        if num_steps > 0:
            batch_stability_loss = batch_stability_loss / (len(x) * num_steps)

        # (3) Entropy Loss: Was the selection made without hesitation?
        entropy_loss = self._compute_entropy(probs)

        # (Optional) Sparsity Loss or Latent Regularization could be added here
        # to prevent mode collapse (e.g., all inputs mapping to same probe).
        # For now, we rely on Entropy Loss and diverse input data.

        # --- Total Loss ---
        stability_coeff = self.model.config.get("stability_coeff", 0.01)
        entropy_coeff = self.model.config.get("entropy_coeff", 0.1)

        # In unsupervised learning, the balance between Stability and Entropy is key.
        total_loss = (stability_coeff * batch_stability_loss) + (entropy_coeff * entropy_loss)

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def fit(self, dataloader, epochs: int = 5, attention_mode: str = "soft"):
        """
        Executes unsupervised learning for a specified number of epochs.
        The dataloader returns (x) or (x, y), but only x is used.
        """
        self.model.train()
        self.model.config["attention_mode"] = attention_mode

        loss_history = []

        print(f"\n--- Start Unsupervised Training ({epochs} epochs) ---")
        print(f"Device: {self.device}")
        print(
            f"Params: Stability={self.model.config.get('stability_coeff', 0.01)}, Entropy={self.model.config.get('entropy_coeff', 0.1)}"
        )

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for batch_idx, batch_data in enumerate(dataloader):
                # DataLoader format handling
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    x_batch = batch_data[0]
                    # y_batch (index 1) is ignored
                else:
                    x_batch = batch_data

                # Flatten handling
                if x_batch.dim() > 2:
                    x_batch = x_batch.view(x_batch.size(0), -1)

                loss = self.train_step(x_batch)  # y is optional
                total_loss += loss
                count += 1

                if batch_idx % 50 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss:.4f}", end="")

            avg_loss = total_loss / count
            loss_history.append(avg_loss)
            print(f"\nEpoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")

        return loss_history
