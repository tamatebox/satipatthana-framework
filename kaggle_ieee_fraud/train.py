import torch
import torch.optim as optim
from samadhi.model.mlp_samadhi import MlpSamadhiModel
from samadhi.train.anomaly_trainer import AnomalySamadhiTrainer
from dataset import get_dataloaders, IEEEFraudDataset
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import numpy as np
import gc
from torch.utils.data import DataLoader, TensorDataset


def main():
    # Paths
    DATA_DIR = "kaggle_ieee_fraud/data"
    SUBMISSION_PATH = "kaggle_ieee_fraud/submission.csv"

    # Hyperparams
    BATCH_SIZE = 1024
    PRETRAIN_EPOCHS = 10  # Increased from 2
    MAIN_EPOCHS = 10  # Increased from 5
    LATENT_DIM = 64
    HIDDEN_DIM = 256
    LR = 1e-4  # Decreased from 1e-3
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    print(f"Using device: {DEVICE}")

    # Data
    # debug=True loads only a small subset of data for quick testing
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(DATA_DIR, BATCH_SIZE, debug=False)
    print(f"Input Feature Dim: {input_dim}")

    # Config for MlpSamadhiModel
    config = {
        "dim": LATENT_DIM,
        "input_dim": input_dim,
        "adapter_hidden_dim": HIDDEN_DIM,
        "n_probes": 10,  # Reduced from 20
        "probe_init": "kmeans",  # This is just a flag for internal logic, actual kmeans init is below
        "gate_threshold": 0.0,  # Reverted to 0.0
        "refine_steps": 5,
        "vicara_type": "standard",
        "attention_mode": "soft",  # Soft for training, Hard for prediction
        "probe_trainable": True,
        # Loss params (tuned for anomaly detection)
        "anomaly_margin": 5.0,  # Reverted to 5.0
        "anomaly_weight": 2.0,
        "stability_coeff": 0.01,
        "entropy_coeff": 0.05,
        "balance_coeff": 0.01,
        "training_attention_mode": "soft",
        "prediction_attention_mode": "hard",
    }

    # Model
    model = MlpSamadhiModel(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Trainer
    trainer = AnomalySamadhiTrainer(model, optimizer, device=DEVICE)

    # 1. Pretrain Autoencoder on Normal Data Only
    print("\n--- Autoencoder Pre-training (Normal Only) ---")
    # Filter normal samples for autoencoder pretraining
    X_train_normal_list = []
    y_train_normal_list = []
    for x_batch, y_batch in train_loader:
        normal_mask = y_batch == 0
        X_train_normal_list.append(x_batch[normal_mask])
        y_train_normal_list.append(y_batch[normal_mask])

    X_train_normal = torch.cat(X_train_normal_list)
    y_train_normal = torch.cat(y_train_normal_list)

    # dataset.py から IEEEFraudDataset をインポート済み

    # オートエンコーダ用のデータセットを作成
    autoencoder_dataset = IEEEFraudDataset(X_train_normal, y_train_normal)
    train_autoencoder_loader = DataLoader(dataset=autoencoder_dataset, batch_size=BATCH_SIZE, shuffle=True)

    trainer.pretrain_autoencoder(train_autoencoder_loader, epochs=PRETRAIN_EPOCHS)
    del X_train_normal_list, y_train_normal_list, X_train_normal, y_train_normal, train_autoencoder_loader
    gc.collect()

    # 2. Initialize Probes with K-Means on Normal Data Latents
    print("\n--- Initializing Probes with K-Means on Normal Data Latents ---")
    model.eval()
    latents = []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            # Use only normal samples to initialize probes for normal behavior
            x_normal = x_batch[y_batch == 0]
            if len(x_normal) > 0:
                z = model.vitakka.adapter(x_normal)  # Extract latent features
                latents.append(z.cpu().numpy())

    latents = np.concatenate(latents, axis=0)

    # Subsample for KMeans if too large (e.g., > 100k samples)
    if len(latents) > 100000:
        indices = np.random.choice(len(latents), 100000, replace=False)
        latents_sub = latents[indices]
    else:
        latents_sub = latents

    print(f"Running KMeans with {len(latents_sub)} samples and {config['n_probes']} probes...")
    kmeans = KMeans(n_clusters=config["n_probes"], random_state=42, n_init=10)  # n_init for stability
    kmeans.fit(latents_sub)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(DEVICE)
    model.load_probes(centroids)
    print("Probes initialized with K-Means centroids.")
    del latents, latents_sub, kmeans, centroids
    gc.collect()

    # 3. Main Anomaly Training with Margin Loss
    print("\n--- Start Anomaly Training (Margin Loss) ---")
    trainer.fit(train_loader, epochs=MAIN_EPOCHS)

    # 4. Evaluation on Validation Set
    print("\n--- Evaluation on Validation Set ---")
    model.eval()
    val_anomaly_scores = []
    val_true_labels = []

    # Probe usage statistics
    probe_counts = np.zeros(config["n_probes"], dtype=int)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            # SamadhiModel forward pass for anomaly score
            s0, meta = model.vitakka_search(x)

            # Count winner probes
            winners = meta["winner_id"].cpu().numpy()
            for w in winners:
                probe_counts[w] += 1

            s_final, _, _ = model.vicara(s0, context=meta)
            x_recon = model.decoder(s_final)
            anomaly_score = torch.mean((x - x_recon) ** 2, dim=1)  # MSE as anomaly score

            val_anomaly_scores.extend(anomaly_score.cpu().numpy())
            val_true_labels.extend(y.cpu().numpy())

    auc = roc_auc_score(val_true_labels, val_anomaly_scores)
    print(f"Validation ROC AUC: {auc:.4f}")

    print("\n--- Probe Usage Statistics (Validation) ---")
    total_samples = np.sum(probe_counts)
    print(f"Total Samples: {total_samples}")
    for i, count in enumerate(probe_counts):
        percentage = count / total_samples if total_samples > 0 else 0
        print(f"Probe #{i:02d}: {count:5d} ({percentage:.2%})")

    # 5. Inference on Test Set and Submission Generation
    print("\n--- Inference on Test Set and Submission Generation ---")
    test_anomaly_scores = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(DEVICE)
            s0, meta = model.vitakka_search(x)
            s_final, _, _ = model.vicara(s0, context=meta)
            x_recon = model.decoder(s_final)
            anomaly_score = torch.mean((x - x_recon) ** 2, dim=1)
            test_anomaly_scores.extend(anomaly_score.cpu().numpy())

    # The competition target is 'isFraud' (1 = Fraud, 0 = Normal).
    # Our model predicts Anomaly Score (Higher = Fraud).
    # So, higher anomaly_score means higher probability of being fraud.

    submission = pd.DataFrame({"TransactionID": test_ids, "isFraud": test_anomaly_scores})

    # For ROC AUC, the raw score is fine, but for interpretation as "probability"
    # it's common to scale it. Given competition asks for "probability",
    # we can try a simple Min-Max scaling of anomaly scores to [0,1].
    # Note: This is NOT a calibrated probability, just a scaled score.
    min_score = np.min(test_anomaly_scores)
    max_score = np.max(test_anomaly_scores)
    scaled_fraud_prob = (np.array(test_anomaly_scores) - min_score) / (max_score - min_score)

    submission["isFraud"] = scaled_fraud_prob

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
