import pandas as pd
import os

DATA_DIR = "kaggle_loan_payback/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


def analyze_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("\n--- Train Info ---")
    print(train_df.info())

    print("\n--- Missing Values (Train) ---")
    missing = train_df.isnull().sum()
    print(missing[missing > 0])

    print("\n--- Target Distribution ---")
    target_col = "loan_paid_back"  # Confirmed from previous run
    print(f"Target Column: '{target_col}'")
    print(f"  - 1.0: Loan was paid back (Normal/Majority class)")
    print(f"  - 0.0: Loan was defaulted (Anomaly/Minority class)")

    target_distribution = train_df[target_col].value_counts(normalize=True)
    print(f"\nDistribution of '{target_col}':")
    print(target_distribution)
    print(f"  Imbalance ratio (0.0 / 1.0): {target_distribution[0.0] / target_distribution[1.0]:.2f}")

    print("\n--- Numerical Features Summary ---")
    numerical_cols = train_df.select_dtypes(include=["number"]).columns.drop(["id", target_col], errors="ignore")
    print(train_df[numerical_cols].describe())

    print("\n--- Categorical Features Unique Values and Counts ---")
    categorical_cols = train_df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        print(f"\n--- Column: '{col}' ---")
        print(train_df[col].value_counts())


if __name__ == "__main__":
    analyze_data()
