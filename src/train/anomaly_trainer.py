from typing import Optional
import torch
import torch.nn as nn
from src.train.base_trainer import BaseSamadhiTrainer


class AnomalySamadhiTrainer(BaseSamadhiTrainer):
    """
    Anomaly Detection Trainer with Margin Loss (Contrastive Learning).

    Teaches the model to:
    1. Reconstruct "Normal" data well (Attraction).
    2. Fail to reconstruct "Anomaly" data (Repulsion).

    Loss = Loss_Recon(Normal) + Weight * Max(0, Margin - Loss_Recon(Anomaly))
    """

    def __init__(self, model, optimizer, **kwargs):
        # Extract device from kwargs if present, otherwise None
        device = kwargs.get("device")
        super().__init__(model, optimizer, device=device)
        # Load hyperparams from config
        self.margin = self.model.config.get("anomaly_margin", 5.0)
        self.anomaly_weight = self.model.config.get("anomaly_weight", 1.0)

        # Per-sample loss for flexible weighting (reduction='none')
        self.mse_none = nn.MSELoss(reduction="none")

    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> float:
        """
        Executes a single training step.
        y is required here: 0 for Normal, 1 for Anomaly.
        """
        x = x.to(self.device)
        if y is None:
            # Fallback to unsupervised if no label provided (assume all normal)
            # This should generally be avoided in this Trainer.
            y = torch.zeros(x.size(0), device=self.device)
        else:
            y = y.to(self.device)

        self.optimizer.zero_grad()

        # ====================================================
        # 2. Forward Pass
        # ====================================================

        # --- A. Search (Vitakka) ---
        s0, metadata = self.model.vitakka_search(x)
        probs = metadata["probs"]

        # --- B. Refine (Vicara) ---
        s_t = s0
        batch_stability_loss = torch.tensor(0.0, device=self.device)
        num_steps = self.model.config.get("refine_steps", 0)

        if num_steps > 0:
            for _ in range(num_steps):
                s_prev = s_t
                residual = self.model.vicara.refine_step(s_t, metadata)
                s_t = self.model.vicara.update_state(s_t, residual)
                # Sum of change amount
                batch_stability_loss += torch.norm(s_t - s_prev, p=2, dim=1).sum()

        s_final = s_t
        decoded_s_final = self.model.decoder(s_final)

        # ====================================================
        # 3. Loss Calculation (Margin Loss)
        # ====================================================

        # (1) Reconstruction Loss (Per Sample)
        # Mean over features: (Batch, )
        # (Batch, Dim) -> (Batch, )
        recon_errors = torch.mean((decoded_s_final - x) ** 2, dim=1)

        # Normal Loss (y=0): Minimize Error
        # Masking: select only y==0
        normal_mask = y == 0
        if normal_mask.any():
            loss_normal = recon_errors[normal_mask].mean()
        else:
            loss_normal = torch.tensor(0.0, device=self.device)

        # Anomaly Loss (y=1): Maximize Error (Hinge)
        # Loss = max(0, margin - error)
        # If error > margin, loss is 0 (Good job, it's far enough)
        # If error < margin, loss increases (Push it away!)
        anomaly_mask = y == 1
        if anomaly_mask.any():
            dist_anomaly = recon_errors[anomaly_mask]
            loss_anomaly = torch.relu(self.margin - dist_anomaly).mean()
        else:
            loss_anomaly = torch.tensor(0.0, device=self.device)

        # (2) Stability Loss
        if num_steps > 0:
            batch_stability_loss = batch_stability_loss / (len(x) * num_steps)

        # (3) Entropy Loss
        entropy_loss = self._compute_entropy(probs)

        # (4) Load Balancing
        balance_loss = self._compute_load_balance_loss(probs)

        # --- Total Loss ---
        stability_coeff = self.model.config.get("stability_coeff", 0.01)
        entropy_coeff = self.model.config.get("entropy_coeff", 0.1)
        balance_coeff = self.model.config.get("balance_coeff", 0.001)

        # Combine
        total_loss = (
            loss_normal
            + (self.anomaly_weight * loss_anomaly)
            + (stability_coeff * batch_stability_loss)
            + (entropy_coeff * entropy_loss)
            + (balance_coeff * balance_loss)
        )

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def pretrain_autoencoder(self, dataloader, epochs: int = 3):
        """
        Pre-trains AE using data provided by the dataloader.
        It is assumed that the dataloader provides ONLY NORMAL data for this phase.
        """
        print(f"\n{'='*20} Starting Autoencoder Pre-training (Normal Only) {'='*20}")

        ae_optimizer = torch.optim.Adam(
            list(self.model.vitakka.adapter.parameters()) + list(self.model.decoder.parameters()),
            lr=self.optimizer.param_groups[0]["lr"],
        )
        recon_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            total_recon_loss = 0.0
            count = 0

            for batch_data in dataloader:
                # Assume dataloader provides (x, y) where y is for compatibility but not filtered here
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    x = batch_data[0]
                else:
                    x = batch_data  # For dataloaders that only yield x

                x = x.to(self.device)

                ae_optimizer.zero_grad()
                latent_x = self.model.vitakka.adapter(x)
                decoded_x = self.model.decoder(latent_x)
                loss = recon_loss_fn(decoded_x, x)
                loss.backward()
                ae_optimizer.step()

                total_recon_loss += loss.item()
                count += 1

            avg_recon_loss = total_recon_loss / max(count, 1)
            print(f"Pre-train Epoch {epoch+1}/{epochs}, Avg Recon Loss (Normal): {avg_recon_loss:.4f}")

        print(f"{ '='*20} Autoencoder Pre-training Complete {'='*20}\n")

    def fit(self, dataloader, epochs: int = 5):
        self.model.train()
        loss_history = []
        print(f"\n--- Start Anomaly Training (Margin Loss) ---")
        print(f"Device: {self.device}")
        print(f"Params: Margin={self.margin}, Anomaly Weight={self.anomaly_weight}")

        for epoch in range(epochs):
            total_loss = 0
            count = 0
            for batch_idx, batch_data in enumerate(dataloader):
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    x = batch_data[0]
                    y = batch_data[1]
                else:
                    raise ValueError("AnomalyTrainer requires labels (x, y) from dataloader.")

                loss = self.train_step(x, y)
                total_loss += loss
                count += 1

                if batch_idx % 50 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss:.4f}", end="")

            avg_loss = total_loss / count
            loss_history.append(avg_loss)
            print(f"\nEpoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")

        return loss_history
