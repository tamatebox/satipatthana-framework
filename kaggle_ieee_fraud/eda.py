import pandas as pd
import os

DATA_DIR = "kaggle_ieee_fraud/data"


def load_data():
    print("Loading data for EDA...")
    train_transaction = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"), nrows=10000)
    train_identity = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"), nrows=10000)
    print("Data loaded (first 10000 rows).")
    return train_transaction, train_identity


def analyze_data(df, name):
    print(f"\n--- Analysis of {name} ---")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values (Top 10):")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.head(10))

    print("\nData Types:")
    print(df.dtypes.value_counts())


def main():
    train_transaction, train_identity = load_data()

    analyze_data(train_transaction, "Train Transaction")
    analyze_data(train_identity, "Train Identity")

    # Check merge key
    print("\n--- Merge Check ---")
    print("Merging on 'TransactionID'...")
    merged = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
    print(f"Merged Shape: {merged.shape}")
    print("Merged columns sample:", merged.columns.tolist()[:20])


if __name__ == "__main__":
    main()
