# Satipatthana Implementation Cookbook

A purely practical, code-centric guide. Copy, paste, and adapt.

> **For architecture and training strategy decisions**, see [training_strategy.md](training_strategy.md).

---

## 1. Directory Map

```text
satipatthana/
├── configs/           # Type-safe dataclass configs
│   ├── system.py      # SystemConfig (root)
│   ├── factory.py     # create_*_config() factories
│   ├── adapters.py    # MlpAdapterConfig, LstmAdapterConfig, etc.
│   ├── decoders.py    # ConditionalDecoderConfig, etc.
│   ├── vipassana.py   # StandardVipassanaConfig
│   ├── vicara.py      # VicaraConfig variants
│   ├── vitakka.py     # VitakkaConfig
│   ├── sati.py        # SatiConfig
│   ├── augmenter.py   # AugmenterConfig
│   └── enums.py       # AugmenterType, SatiType, etc.
├── core/              # Main system and engines
│   ├── system.py      # SatipatthanaSystem
│   ├── engines.py     # SamathaEngine, VipassanaEngine
│   └── santana.py     # SantanaLog (trajectory)
├── components/        # All modular components
│   ├── adapters/      # MLP, CNN (vision), LSTM/Transformer (sequence)
│   ├── augmenters/    # Identity, Gaussian
│   ├── vitakka/       # Semantic initialization
│   ├── vicara/        # Standard, Weighted, ProbeSpecific
│   ├── refiners/      # MLP refiner (used by Vicara)
│   ├── sati/          # FixedStep, Threshold
│   ├── vipassana/     # Meta-cognition (Triple Score)
│   ├── decoders/      # Reconstruction, Conditional, Auxiliary
│   └── objectives/    # Vipassana loss functions
├── train/             # Training utilities
│   └── trainer.py     # SatipatthanaTrainer
├── data/              # Dataset utilities
│   └── void_dataset.py # VoidDataset, FilteredNoiseVoid
└── utils/             # Utilities
    ├── logger.py      # Logging setup
    └── training.py    # Training helpers
```

---

## 2. Configuration Recipes

### Pattern A: Isolation (Supervised Anomaly with Guidance)

```python
from satipatthana.configs import (
    SystemConfig, SamathaConfig, VipassanaEngineConfig,
    create_adapter_config, create_vicara_config,
    AugmenterConfig, VitakkaConfig, SatiConfig,
    StandardVipassanaConfig, ConditionalDecoderConfig,
)
from satipatthana.configs.enums import AugmenterType, SatiType

config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=64,
        adapter=create_adapter_config("mlp", input_dim=30, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.3),
        vitakka=VitakkaConfig(num_probes=16, temperature=0.2),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
        max_steps=10,
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32, latent_dim=64),
    ),
    decoder=ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=2),
    use_label_guidance=True,  # Guidance ON
    seed=42,
)
```

### Pattern C: Standard Rejection (Unsupervised with Synthetic Noise)

```python
config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=64,
        adapter=create_adapter_config("mlp", input_dim=30, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.3),
        vitakka=VitakkaConfig(num_probes=16),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32, latent_dim=64),
    ),
    use_label_guidance=False,  # Guidance OFF
)
# VoidDataset: Use FilteredNoiseVoid (synthetic noise)
```

### Pattern E: Classification (Multi-class)

```python
config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=64,
        adapter=create_adapter_config("cnn", img_size=28, channels=1, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.2),
        vitakka=VitakkaConfig(num_probes=32, temperature=0.1),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32, latent_dim=64),
    ),
    decoder=ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10),
    use_label_guidance=True,
)
```

### Pattern F: Regression

```python
config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=128,
        adapter=create_adapter_config("mlp", input_dim=100, latent_dim=128),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.1),
        vitakka=VitakkaConfig(num_probes=16),
        vicara=create_vicara_config("standard", latent_dim=128),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-5),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=64, latent_dim=128),
    ),
    decoder=ConditionalDecoderConfig(dim=128, context_dim=64, output_dim=1),
    use_label_guidance=True,
    guidance_loss_type="mse",  # MSE for regression
)
```

### Time Series Anomaly Detection (LSTM)

```python
config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=128,
        adapter=create_adapter_config(
            "lstm",
            input_dim=10,       # features per timestep
            seq_len=50,         # sequence length
            latent_dim=128,
            hidden_dim=256,
        ),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.2),
        vitakka=VitakkaConfig(num_probes=8),
        vicara=create_vicara_config("standard", latent_dim=128),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=64, latent_dim=128),
    ),
    use_label_guidance=False,  # Unsupervised
)
```

---

## 3. Data Preparation

### Standard Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {"x": self.data[idx]}
        if self.labels is not None:
            item["y"] = self.labels[idx]
        return item
```

### VoidDataset for Stage 2

```python
from satipatthana.data import FilteredNoiseVoid, VoidDataset

# Option 1: Synthetic noise (Pattern C)
void_dataset = FilteredNoiseVoid(
    reference_data=train_data_tensor,  # Normal samples
    shape=(input_dim,),
    length=5000,
    min_distance=0.3,                  # Minimum distance from reference
    noise_range=(-1.5, 1.5),
)

# Option 2: Domain OOD data (Pattern D)
# Wrap existing OOD dataset
void_dataset = VoidDataset(ood_dataset)

# Option 3: Target labels as Void (Pattern A, B)
# Filter anomaly labels from training data
anomaly_data = train_data[train_labels == 1]
void_dataset = VoidDataset(TensorDataset(anomaly_data))
```

### Train/Val Split

```python
from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

---

## 4. Training Execution

### Full Curriculum Training

```python
from satipatthana.core.system import SatipatthanaSystem
from satipatthana.train import SatipatthanaTrainer
from transformers import TrainingArguments

# Instantiate model
system = SatipatthanaSystem(config)

# Training arguments
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=32,
    learning_rate=1e-3,
    logging_steps=100,
    save_strategy="epoch",
)

# Initialize trainer
trainer = SatipatthanaTrainer(
    model=system,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    void_dataset=void_dataset,  # For Stage 2
)

# Run full curriculum
results = trainer.run_curriculum(
    stage0_epochs=5,
    stage1_epochs=10,
    stage2_epochs=5,
    stage3_epochs=5,
)
```

### Stage-by-Stage Training (Manual Control)

```python
from satipatthana.core.system import TrainingStage

# Stage 0: Adapter pre-training
system.set_training_stage(TrainingStage.ADAPTER_PRETRAINING)
trainer.train(num_epochs=5)

# Stage 1: Samatha training
system.set_training_stage(TrainingStage.SAMATHA_TRAINING)
trainer.train(num_epochs=10)

# Stage 2: Vipassana training
system.set_training_stage(TrainingStage.VIPASSANA_TRAINING)
trainer.train(num_epochs=5)

# Stage 3: Decoder fine-tuning
system.set_training_stage(TrainingStage.DECODER_FINETUNING)
trainer.train(num_epochs=5)
```

### Learning Rate per Stage

```python
# Different LR for different stages
stage_lr = {
    0: 1e-3,   # Adapter: higher LR
    1: 5e-4,   # Samatha: moderate
    2: 1e-4,   # Vipassana: lower (frozen backbone)
    3: 1e-4,   # Decoder: fine-tuning
}

for stage, epochs in [(0, 5), (1, 10), (2, 5), (3, 5)]:
    args.learning_rate = stage_lr[stage]
    trainer.args = args
    system.set_training_stage(TrainingStage(stage))
    trainer.train(num_epochs=epochs)
```

---

## 5. Checkpoint Management

### Save Checkpoints

```python
import torch

# Save full checkpoint (recommended)
checkpoint = {
    "model_state_dict": system.state_dict(),
    "config": config,
    "stage": current_stage,
    "epoch": epoch,
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint_stage1_epoch10.pt")

# Save model only (for inference)
torch.save(system.state_dict(), "model_final.pt")
```

### Load Checkpoints

```python
# Load full checkpoint (resume training)
checkpoint = torch.load("checkpoint_stage1_epoch10.pt")
system = SatipatthanaSystem(checkpoint["config"])
system.load_state_dict(checkpoint["model_state_dict"])
system.set_training_stage(TrainingStage(checkpoint["stage"]))

# Load model only (inference)
system = SatipatthanaSystem(config)
system.load_state_dict(torch.load("model_final.pt"))
system.eval()
```

### Checkpoint Strategy by Stage

```python
# Save after each stage completes
for stage in [0, 1, 2, 3]:
    system.set_training_stage(TrainingStage(stage))
    trainer.train(num_epochs=epochs_per_stage[stage])

    # Save stage checkpoint
    torch.save({
        "model_state_dict": system.state_dict(),
        "config": config,
        "stage": stage,
    }, f"checkpoint_stage{stage}_complete.pt")
```

### Resume from Specific Stage

```python
# Resume from Stage 1 completion
checkpoint = torch.load("checkpoint_stage1_complete.pt")
system = SatipatthanaSystem(checkpoint["config"])
system.load_state_dict(checkpoint["model_state_dict"])

# Continue from Stage 2
for stage in [2, 3]:
    system.set_training_stage(TrainingStage(stage))
    trainer.train(num_epochs=epochs_per_stage[stage])
```

---

## 6. Inference & Analysis

### Basic Inference

```python
system.eval()
with torch.no_grad():
    result = system(x)

output = result.output           # Task output
s_star = result.s_star           # Converged state
v_ctx = result.v_ctx             # Context vector
trust = result.trust_score       # OOD detection (0-1)
conformity = result.conformity_score  # Trajectory anomaly (0-1)
confidence = result.confidence_score  # Combined score (0-1)
```

### Two-Stage Inference (Anomaly Detection)

```python
def two_stage_inference(system, x, trust_threshold=0.5):
    """Trust filtering before classification."""
    result = system(x)

    # Stage 1: Trust filtering
    is_trustworthy = result.trust_score > trust_threshold

    # Stage 2: Classification (only for trustworthy)
    predictions = torch.zeros(len(x), dtype=torch.long)
    predictions[~is_trustworthy] = -1  # Rejected as anomaly

    if is_trustworthy.any():
        trusted_output = result.output[is_trustworthy]
        predictions[is_trustworthy] = trusted_output.argmax(dim=-1)

    return predictions, result.trust_score
```

### Batch Anomaly Detection

```python
def detect_anomalies(system, dataloader, threshold=0.5):
    """Detect anomalies in batch."""
    system.eval()
    all_scores = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"]
            result = system(x)

            anomaly_score = 1.0 - result.trust_score
            is_anomaly = anomaly_score > threshold

            all_scores.extend(anomaly_score.cpu().numpy())
            all_predictions.extend(is_anomaly.cpu().numpy())

    return np.array(all_scores), np.array(all_predictions)
```

### Trajectory Analysis

```python
def analyze_trajectory(result):
    """Analyze convergence trajectory."""
    santana = result.santana

    print(f"Convergence steps: {santana.num_steps}")
    print(f"Final delta norm: {santana.delta_norms[-1]:.6f}")
    print(f"Energy trajectory: {santana.energies}")

    # Plot convergence
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(santana.delta_norms)
    plt.xlabel("Step")
    plt.ylabel("Delta Norm")
    plt.title("State Change")

    plt.subplot(1, 2, 2)
    plt.plot(santana.energies)
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy Trajectory")

    plt.tight_layout()
    plt.show()
```

### Score Distribution Analysis

```python
def plot_score_distribution(scores_normal, scores_anomaly):
    """Plot trust score distribution."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(scores_normal, bins=50, alpha=0.7, label="Normal", density=True)
    plt.hist(scores_anomaly, bins=50, alpha=0.7, label="Anomaly", density=True)
    plt.xlabel("Trust Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Trust Score Distribution")
    plt.show()
```

---

## 7. Config Quick Reference

| Config Class | Required Fields |
|:---|:---|
| `MlpAdapterConfig` | `input_dim` |
| `LstmAdapterConfig` | `input_dim`, `seq_len` |
| `CnnAdapterConfig` | `img_size`, `channels` |
| `TransformerAdapterConfig` | `input_dim`, `seq_len` |
| `ReconstructionDecoderConfig` | `input_dim` |
| `ConditionalDecoderConfig` | `dim`, `context_dim`, `output_dim` |
| `StandardVipassanaConfig` | `context_dim` |

---

## 8. Troubleshooting

### Dimension Mismatch

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Fix:** Check `input_dim`, `seq_len`, `latent_dim` consistency across Adapter → Decoder.

### Loss is NaN

**Fix:**

- Reduce `max_steps` (default: 10)
- Increase `beta` (inertia) in Vicara
- Extend Stage 0 training epochs
- Check for zero division in loss

### Stage 3 Decoder Error

```text
RuntimeError: size mismatch for decoder.fc.weight
```

**Fix:** `ConditionalDecoderConfig.dim` must equal `latent_dim`, and `context_dim` must match Vipassana's `context_dim`.

### Low Trust Scores Everywhere

**Fix:**

- Extend Stage 2 epochs
- Balance Augmented/Drunk/Mismatch/Void paths
- Check VoidDataset is properly different from training data
- Verify Stage 1 converged (check delta norms)

### Out of Memory

**Fix:**

- Reduce `batch_size`
- Reduce `latent_dim`
- Reduce `max_steps`
- Use gradient checkpointing
