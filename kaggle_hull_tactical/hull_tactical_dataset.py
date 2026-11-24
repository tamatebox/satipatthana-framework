import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import Optional, Dict


class HullInpaintingDataset(Dataset):
    def __init__(
        self, csv_path: str = "kaggle_hull_tactical/data/train.csv", mode: str = "train", mask_target: bool = True
    ):
        """
        Args:
            csv_path: Path to the dataset CSV file.
            mode: 'train' or 'test'. In 'train', target is available.
            mask_target: If True, masks the target column in the input 'x'.
        """
        self.mode = mode
        self.mask_target = mask_target

        # Load Data
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # Fallback for when data is missing (e.g. during code dev)
            print(f"Warning: {csv_path} not found. Using synthetic data.")
            df = self._create_synthetic_data()

        # Preprocessing
        # 1. Drop non-feature columns if they exist
        ignore_cols = ["date_id", "forward_returns", "risk_free_rate"]
        # target_col = "market_forward_excess_returns"
        # We need to ensure target is the LAST column for easier indexing

        self.target_col = "market_forward_excess_returns"

        # Select Features
        feature_cols = [c for c in df.columns if c not in ignore_cols and c != self.target_col]

        # Handling Missing Values (Simple fill 0 for now as per plan)
        df[feature_cols] = df[feature_cols].fillna(0.0)
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].fillna(0.0)

        # Normalize (Ideally should calculate mean/std from TRAIN split and apply to Test)
        # For simplicity here, we normalize the whole loaded df
        self.mean = (
            df[feature_cols + [self.target_col]].mean() if self.target_col in df.columns else df[feature_cols].mean()
        )
        self.std = (
            df[feature_cols + [self.target_col]].std() if self.target_col in df.columns else df[feature_cols].std()
        )

        # Avoid division by zero
        self.std = self.std.replace(0, 1.0)

        normalized_df = (
            df[feature_cols + ([self.target_col] if self.target_col in df.columns else [])] - self.mean
        ) / self.std

        # Convert to float32
        self.data_matrix = normalized_df.values.astype(np.float32)

        # Identify target index (last column)
        self.target_idx = self.data_matrix.shape[1] - 1

        if self.mode == "test" and self.target_col not in df.columns:
            # If test and no target, we might need to append a dummy column
            # But usually we just assume the target column exists in training data
            pass

    def _create_synthetic_data(self):
        # Create dummy data with 110 columns
        N = 100
        cols = [f"feat_{i}" for i in range(98)] + ["lagged_target", "market_forward_excess_returns"]
        data = np.random.randn(N, len(cols))
        return pd.DataFrame(data, columns=cols)

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 1. Full Vector (Ground Truth)
        full_vector = self.data_matrix[idx].copy()

        # 2. Masked Input
        masked_vector = full_vector.copy()
        if self.mask_target:
            masked_vector[self.target_idx] = 0.0

        return {
            "x": torch.from_numpy(masked_vector),  # Model Input
            "labels": torch.from_numpy(full_vector),  # Reconstruction Target (renamed 'target' to 'labels' for HF)
        }
