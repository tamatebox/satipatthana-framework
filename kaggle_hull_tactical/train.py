import os
import torch
import copy
from samadhi.core.builder import SamadhiBuilder
from samadhi.train.hf_trainer import SamadhiTrainer
from samadhi.train.objectives.supervised_regression import SupervisedRegressionObjective
from transformers import TrainingArguments
from hull_tactical_dataset import HullInpaintingDataset
from config import config  # Import the directly defined config


def main():
    # 1. Prepare Data
    print("Initializing Dataset...")
    dataset = HullInpaintingDataset(mode="train", mask_target=True)

    # Determine input dim dynamically from dataset (if config.TOTAL_DIM is not sufficient)
    # If the config specifies a fixed TOTAL_DIM, ensure it matches the dataset or adjust config.
    # For now, let's assume the config's TOTAL_DIM is the source of truth for model architecture.
    # We'll add a check to ensure dataset output matches config expected input_dim

    if dataset.data_matrix.shape[1] != config.adapter.input_dim:
        raise ValueError(
            f"Dataset output dimension ({dataset.data_matrix.shape[1]}) does not match "
            f"config.adapter.input_dim ({config.adapter.input_dim}). Please adjust your data or config."
        )

    print(f"Data Input Dimension: {config.adapter.input_dim}")

    # 2. Config is already created and imported as 'config'

    # 3. Build Model
    print("Building Samadhi Model...")
    builder = SamadhiBuilder(config)
    builder.set_adapter().set_vitakka().set_vicara().set_decoder()
    model = builder.build()

    # 4. Training Arguments
    # Using a temporary output directory
    output_dir = "kaggle_hull_tactical/checkpoints"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Short for demo/testing
        per_device_train_batch_size=32,
        logging_steps=500,
        overwrite_output_dir=True,
        remove_unused_columns=False,  # Important for custom dataset keys
    )

    # ==========================================
    # Phase 1: Pre-training (High Stability)
    # ==========================================
    print("\n=== Phase 1: Pre-training (High Stability) ===")

    # Modify config for stability
    # We clone the objective config to avoid mutating the original one permanently if we reused it
    # But here we modify the model's config directly.
    model.config.objective.stability_coeff = 0.2

    # Create Objective
    # We use SupervisedRegressionObjective because our dataset provides 'labels' (full vector)
    # and 'x' (masked vector). We want to map Masked -> Full.
    objective_p1 = SupervisedRegressionObjective(model.config)

    # Trainer
    trainer_p1 = SamadhiTrainer(model=model, args=training_args, train_dataset=dataset, objective=objective_p1)

    trainer_p1.train()
    print("Phase 1 Complete.")

    # Optional: Initialize Probes with K-Means (Conceptual step, skipping for now as it requires implementing K-Means on latent states)

    # ==========================================
    # Phase 2: Fine-tuning (Accuracy Focused)
    # ==========================================
    print("\n=== Phase 2: Fine-tuning (Accuracy Focused) ===")

    # Relax stability constraint
    model.config.objective.stability_coeff = 0.01

    # Create new Objective with updated config
    objective_p2 = SupervisedRegressionObjective(model.config)

    # Update Trainer's objective
    # We can create a new trainer or update the existing one. Creating new is cleaner.
    training_args_p2 = copy.deepcopy(training_args)
    training_args_p2.num_train_epochs = 10  # More epochs for fine-tuning

    trainer_p2 = SamadhiTrainer(model=model, args=training_args_p2, train_dataset=dataset, objective=objective_p2)

    trainer_p2.train()
    print("Phase 2 Complete.")

    # Save Final Model
    final_path = os.path.join(output_dir, "final_model")
    # Custom save might be needed if HF save_model doesn't serialize everything
    # But SamadhiEngine is a nn.Module, so it should work.
    trainer_p2.save_model(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
