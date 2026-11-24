import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import gc


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # Use float32 for torch compatibility
                else:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


class IEEEFraudDataset(Dataset):
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


def load_and_preprocess(data_dir, debug=False):
    print("Loading data...")
    if debug:
        nrows = 10000
    else:
        nrows = None

    train_transaction = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"), nrows=nrows)
    train_identity = pd.read_csv(os.path.join(data_dir, "train_identity.csv"), nrows=nrows)

    test_transaction = pd.read_csv(os.path.join(data_dir, "test_transaction.csv"), nrows=nrows)
    test_identity = pd.read_csv(os.path.join(data_dir, "test_identity.csv"), nrows=nrows)

    print("Merging data...")
    train = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
    test = pd.merge(test_transaction, test_identity, on="TransactionID", how="left")

    del train_transaction, train_identity, test_transaction, test_identity
    gc.collect()

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    y = train["isFraud"].copy()
    test_ids = test["TransactionID"].copy()

    # Combine for consistent feature engineering
    train_len = len(train)
    combined = pd.concat([train.drop(["isFraud"], axis=1), test], axis=0)

    del train, test
    gc.collect()

    # --- Feature Engineering ---
    print("Feature Engineering...")

    # 1. Time Features
    # TransactionDT is seconds.
    combined["hour"] = (combined["TransactionDT"] // (3600)) % 24
    combined["day_of_week"] = (combined["TransactionDT"] // (3600 * 24)) % 7

    # 2. Count Encoding
    for col in ["card1", "card2", "addr1", "addr2", "P_emaildomain"]:
        combined[f"{col}_count"] = combined[col].map(combined[col].value_counts(dropna=False))

    # 3. Aggregated Features (Mean/Std of TransactionAmt grouped by key cols)
    for group_col in ["card1", "card4", "addr1"]:
        combined[f"TransactionAmt_mean_{group_col}"] = combined.groupby(group_col)["TransactionAmt"].transform("mean")
        combined[f"TransactionAmt_std_{group_col}"] = combined.groupby(group_col)["TransactionAmt"].transform("std")

    # Drop TransactionDT and ID as they are raw keys/timestamps not suitable for direct input
    combined = combined.drop(["TransactionID", "TransactionDT"], axis=1)

    # Preprocessing
    print("Preprocessing...")

    # Select object columns from combined dataframe to catch all
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        combined[col] = combined[col].astype(str)
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])

    # Handle missing values (simple strategy: fill with -999)
    combined = combined.fillna(-999)

    # Reduce memory before scaling (optional but good for large data)
    combined = reduce_mem_usage(combined)

    # Standard Scaling
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    # Split back
    X = combined_scaled[:train_len]
    X_test = combined_scaled[train_len:]

    input_dim = X.shape[1]

    return X, y.values, X_test, test_ids, input_dim


def get_dataloaders(data_dir, batch_size=1024, debug=False):
    X, y, X_test, test_ids, input_dim = load_and_preprocess(data_dir, debug=debug)

    # Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_dataset = IEEEFraudDataset(X_train, y_train)
    val_dataset = IEEEFraudDataset(X_val, y_val)
    test_dataset = IEEEFraudDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, test_ids, input_dim
