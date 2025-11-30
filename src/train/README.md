# Training Module Documentation

This directory contains the logic for training the Samadhi Model. The framework uses a custom "Objective-Driven" approach, decoupling the model architecture from the loss calculation logic.

## Directory Structure

*   **`objectives/`**: Contains the implementations of different training objectives (loss functions). This is the primary extension point for defining *what* the model learns.
*   **`hf_trainer.py`**: A wrapper around Hugging Face's `Trainer`. It orchestrates the training loop and delegates loss calculation to the selected `Objective`.

---

## Objectives

In the Samadhi Framework, an **Objective** defines:
1.  **The Loss Function**: How the model's performance is measured (e.g., reconstruction error, classification accuracy, stability loss).
2.  **Execution Path**: Which parts of the model (Vitakka/Vicara) should run during the forward pass.

All objectives inherit from `src.train.objectives.base_objective.BaseObjective`.

### Key Properties

*   `needs_vitakka` (bool): If `True`, the model runs the Vitakka (Search) module. If `False`, the Adapter output is used directly as the initial state.
*   `needs_vicara` (bool): If `True`, the model runs the Vicara (Refinement) loop. If `False`, the refinement step is skipped.

### How to Add a New Objective

To define a new learning goal (e.g., a specific type of anomaly detection or a custom supervised task), create a new file in `src/train/objectives/` and inherit from `BaseObjective`.

```python
from src.train.objectives.base_objective import BaseObjective
import torch
import torch.nn.functional as F

class MyCustomObjective(BaseObjective):
    # Define which components are needed
    needs_vitakka = True
    needs_vicara = True

    def compute_loss(
        self,
        x,              # Input
        y,              # Target (optional)
        s0,             # Initial State
        s_final,        # Final Purified State
        decoded_s_final,# Output from Decoder
        metadata,       # Vitakka metadata (probe probs, etc.)
        num_refine_steps
    ):
        # 1. Calculate Primary Loss (e.g., MSE)
        task_loss = F.mse_loss(decoded_s_final, x)

        # 2. Calculate Stability Loss (if Vicara is used)
        # Penalize movement between s0 and s_final
        stability_loss = torch.norm(s_final - s0, dim=1).mean()

        # 3. Combine
        total_loss = task_loss + 0.1 * stability_loss

        # Return total loss and components for logging
        return total_loss, {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "stability_loss": stability_loss.item()
        }
```

### Common Objectives

*   **`AutoencoderObjective`**: Standard reconstruction loss. Skips Vitakka and Vicara (often used for pre-training Adapters/Decoders).
*   **`UnsupervisedObjective`**: The full Samadhi unsupervised learning process. Includes reconstruction loss, stability loss, and entropy regularization for Vitakka probes.
*   **`SupervisedRegressionObjective`**: Uses `y` labels to guide the latent state formation.

---

## Samadhi Trainer

`SamadhiTrainer` extends the Hugging Face `Trainer`.

*   **Role**: It overrides the `compute_loss` method to invoke `self.objective.compute_loss`.
*   **Usage**: You do not typically need to modify the Trainer. Instead, pass the desired `Objective` instance to the Trainer when initializing it.

```python
from src.train.hf_trainer import SamadhiTrainer
from src.train.objectives.unsupervised import UnsupervisedObjective

# ... model initialization ...

objective = UnsupervisedObjective(config)

trainer = SamadhiTrainer(
    model=model,
    objective=objective,
    # ... other args
)

trainer.train()
```

## Testing New Objectives

When adding a new Objective, add a corresponding test in `tests/train/objectives/`.
Ensure you test:
1.  That the loss is calculated correctly.
2.  That the `needs_vitakka` / `needs_vicara` flags are respected.
3.  That it handles edge cases (e.g., `y` being None if applicable).
