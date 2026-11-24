import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


class LoanDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


def load_and_preprocess_data(train_path, test_path):
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate target
    target = "loan_paid_back"
    y = train_df[target].values

    # 0 = Default (Anomaly), 1 = Paid (Normal)
    # The AnomalyTrainer expects 0=Normal, 1=Anomaly.
    # So we need to FLIP the labels for the Trainer.
    # Current: 1=Good, 0=Bad
    # Target: 0=Good, 1=Bad
    y_flipped = 1 - y

    # Drop target and ID
    X = train_df.drop([target, "id"], axis=1)
    X_test = test_df.drop(["id"], axis=1)

    # Combine for consistent preprocessing
    # (Be careful with data leakage in real competitions, but for baseline it's okay-ish)
    # Better: fit on train, transform test.

    # Identify Categorical and Numerical
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(include=["number"]).columns

    print(f"Categorical: {list(cat_cols)}")
    print(f"Numerical: {list(num_cols)}")

    # Handle Missing Values (if any - fill with mode/median)
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
        X_test[col] = X_test[col].fillna(X[col].mode()[0])

    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
        X_test[col] = X_test[col].fillna(X[col].median())

    # Encoding Categorical
    # Using LabelEncoder for simplicity with Tree models, but for Neural Nets, One-Hot is usually better.
    # Or Embeddings. For this MLP, let's use One-Hot (get_dummies).

    combined = pd.concat([X, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=True)  # dummy_na handles unseen NaNs

    # Split back
    X = combined.iloc[: len(X)]
    X_test = combined.iloc[len(X) :]

    # Scaling Numerical
    scaler = StandardScaler()
    # We must fit on Train only to avoid leakage
    # Ideally we should fit on Train ONLY, but we need to match columns of One-Hot.
    # The get_dummies above handled column matching.

    # Extract numerical columns again from the encoded dataframe?
    # Actually, scaling the boolean dummy columns is sometimes debated, but usually okay for Neural Nets.
    # Let's scale everything.
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    return X_scaled, y_flipped, X_test_scaled, scaler


def get_dataloaders(train_path, test_path, batch_size=256, val_size=0.2):
    X, y, X_test, scaler = load_and_preprocess_data(train_path, test_path)

    # Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42, stratify=y)

    train_dataset = LoanDataset(X_train, y_train)
    val_dataset = LoanDataset(X_val, y_val)
    # For submission
    test_dataset = LoanDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X.shape[1]
