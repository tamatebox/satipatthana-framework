import torch
import torch.optim as optim
from samadhi.model.mlp_samadhi import MlpSamadhiModel
from samadhi.train.anomaly_trainer import AnomalySamadhiTrainer
from dataset import get_dataloaders
import os
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    # Paths
    DATA_DIR = "kaggle_loan_payback/data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SUBMISSION_PATH = "kaggle_loan_payback/submission.csv"


import torch
import torch.optim as optim
from samadhi.model.mlp_samadhi import MlpSamadhiModel
from samadhi.train.anomaly_trainer import AnomalySamadhiTrainer
from dataset import get_dataloaders
import os
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    # Paths
    DATA_DIR = "kaggle_loan_payback/data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SUBMISSION_PATH = "kaggle_loan_payback/submission.csv"

    # Hyperparams
    BATCH_SIZE = 1024
    # PRETRAIAN_EPOCHS = 10 # Modified to 2
    # EPOCHS = 10           # Modified to 5
    LATENT_DIM = 64
    HIDDEN_DIM = 256
    LR = 1e-3
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    print(f"Using device: {DEVICE}")

    # Data
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(TRAIN_PATH, TEST_PATH, BATCH_SIZE)
    print(f"Input Feature Dim: {input_dim}")

    # Config
    config = {
        "dim": LATENT_DIM,
        "input_dim": input_dim,
        "adapter_hidden_dim": HIDDEN_DIM,
        "n_probes": 2,  # Adjusted to 2 as only 2 probes are actively used
        "probe_init": "he",
        "gate_threshold": 0.0,
        "refine_steps": 5,  # Keep at 5 for now
        "vicara_type": "standard",
        "attention_mode": "soft",
        "probe_trainable": True,
        # Loss params
        "anomaly_margin": 5.0,
        "anomaly_weight": 2.0,
        "stability_coeff": 0.01,
        "entropy_coeff": 0.05,
        "balance_coeff": 0.01,  # Reverted to 0.01
    }

    # Model
    model = MlpSamadhiModel(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Trainer
    trainer = AnomalySamadhiTrainer(model, optimizer, device=DEVICE)

    # 1. Pretrain Autoencoder
    trainer.pretrain_autoencoder(train_loader, epochs=2)

    # 2. Train Anomaly Detection
    loss_history = trainer.fit(train_loader, epochs=5)

    # 3. Evaluation
    print("\n--- Evaluation on Validation Set ---")
    model.eval()
    val_preds = []
    val_targets = []

    # Probe Analysis
    probe_stats = {i: {"count": 0, "sum_target": 0.0, "sum_error": 0.0} for i in range(config["n_probes"])}

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            # Reconstruct
            s0, meta = model.vitakka_search(x)
            s_t = s0
            for _ in range(config["refine_steps"]):
                residual = model.vicara.refine_step(s_t, meta)
                s_t = model.vicara.update_state(s_t, residual)

            decoded = model.decoder(s_t)

            # Error
            error = torch.mean((x - decoded) ** 2, dim=1)

            val_preds.extend(error.cpu().numpy())
            val_targets.extend(y.numpy())

            # Probe Analysis Collection
            # meta["winner_id"] is (Batch,) tensor of indices
            winners = meta["winner_id"]
            if isinstance(winners, torch.Tensor):
                winners = winners.cpu().numpy()

            batch_errors = error.cpu().numpy()
            batch_targets = y.numpy()

            for i, winner_id in enumerate(winners):
                probe_stats[winner_id]["count"] += 1
                probe_stats[winner_id]["sum_target"] += batch_targets[i]  # 1 is Default
                probe_stats[winner_id]["sum_error"] += batch_errors[i]

    auc = roc_auc_score(val_targets, val_preds)
    print(f"Validation ROC AUC: {auc:.4f}")

    print("\n--- Probe Analysis (Validation) ---")
    print(f"{'Probe ID':<10} | {'Count':<10} | {'Default Rate':<15} | {'Avg Error':<15}")
    print("-" * 60)
    for pid in sorted(probe_stats.keys()):
        stats = probe_stats[pid]
        count = stats["count"]
        if count > 0:
            default_rate = stats["sum_target"] / count
            avg_error = stats["sum_error"] / count
            print(
                f"{pid:<10} | {count:<10} | {default_rate:.4f}{' (High!)' if default_rate > 0.5 else '':<8} | {avg_error:.4f}"
            )
        else:
            print(f"{pid:<10} | {0:<10} | {'N/A':<15} | {'N/A':<15}")

    # 4. Inference on Test
    print("\n--- Inference on Test Set ---")
    test_preds = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(DEVICE)
            s0, meta = model.vitakka_search(x)
            s_t = s0
            for _ in range(config["refine_steps"]):
                residual = model.vicara.refine_step(s_t, meta)
                s_t = model.vicara.update_state(s_t, residual)
            decoded = model.decoder(s_t)
            error = torch.mean((x - decoded) ** 2, dim=1)
            test_preds.extend(error.cpu().numpy())

    # Create Submission
    # The competition target is 'loan_paid_back' (1 = Good, 0 = Bad).
    # Our model predicts Anomaly Score (Higher = Bad/Default).
    # So Probability(Paid Back) ~ 1 / (1 + Error) or just -Error (ranking).
    # AUC is rank-based.
    # If val_preds (Error) correlates with Default (0 -> 1),
    # then for "Paid Back" (1), we should invert the score.
    # Let's save the raw Error first, but formatted for submission.

    # "Predict the probability that a borrower will pay back their loan."
    # Target = 1 (Pay back).
    # Our Score = Likelihood of Default (Anomaly).
    # So we need to invert it.
    # Simple inversion for AUC: -1 * Error.

    # Or normalize to [0, 1].
    # Let's use MinMax or Sigmoid on -Error?
    # ROC AUC only cares about order.
    # Submission requires probability? Usually yes.
    # Let's convert Error to Prob(Default) -> Prob(Payback) = 1 - Prob(Default).
    # This is tricky without a calibrated classifier head.

    # Alternative: The Anomaly Score is just a score.
    # Let's verify correlation.
    # If AUC > 0.5, then Higher Score = Higher Target(1=Default in Val).
    # Wait, in Val: y=1 is Default.
    # So Higher Error predicts Default. Correct.
    # Competition Target: 1 = Paid Back. (Inverse of Default).
    # So Higher Error predicts Target=0.
    # So we should submit -Error (or 1/Error) to predict Target=1.

    submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # Scale scores to [0, 1] for "probability" look (though rank matters most)
    import numpy as np

    preds = np.array(test_preds)
    # Invert: We want High Score for Paid Back.
    # Currently High Score (Error) = Default.
    # So use -preds.
    inv_preds = -preds

    # MinMax Scale to [0, 1]
    min_p = inv_preds.min()
    max_p = inv_preds.max()
    scaled_preds = (inv_preds - min_p) / (max_p - min_p)

    submission["loan_paid_back"] = scaled_preds
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
