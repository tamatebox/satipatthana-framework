# Training Module Documentation

This directory contains the training logic for Samadhi Framework.

## Directory Structure

* **`trainer.py`**: `SamadhiV4Trainer` - 4-stage curriculum trainer extending Hugging Face's `Trainer`

## 4-Stage Curriculum Training

Samadhi Framework uses a 4-stage curriculum learning approach:

| Stage | Name | Trainable Components | Objective |
|-------|------|---------------------|-----------|
| 0 | Adapter Pre-training | Adapter, AdapterReconHead | Reconstruction |
| 1 | Samatha Training | Adapter, Vitakka, Vicara, SamathaReconHead, (AuxHead) | Stability + Reconstruction + (Label Guidance) |
| 2 | Vipassana Training | Vipassana | Contrastive (BCE on trust scores) |
| 3 | Decoder Fine-tuning | TaskDecoder | Task-specific (CrossEntropy/MSE) |

## Usage

### Basic Usage

```python
from samadhi.train import SamadhiV4Trainer
from samadhi.core.system import SamadhiSystem, TrainingStage
from transformers import TrainingArguments

# Build your SamadhiSystem
system = SamadhiSystem(...)

# Create training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=32,
)

# Initialize trainer
trainer = SamadhiV4Trainer(
    model=system,
    args=args,
    train_dataset=train_dataset,
    stage=TrainingStage.SAMATHA_TRAINING,
)

# Train single stage
trainer.train()
```

### Full Curriculum Training

```python
trainer = SamadhiV4Trainer(
    model=system,
    args=args,
    train_dataset=train_dataset,
)

# Run all 4 stages sequentially
results = trainer.run_curriculum(
    stage0_epochs=5,   # Adapter pre-training (0 to skip)
    stage1_epochs=10,  # Samatha training
    stage2_epochs=5,   # Vipassana training
    stage3_epochs=5,   # Decoder fine-tuning
)
```

### Stage Switching

```python
# Switch to a specific stage
trainer.set_stage(TrainingStage.VIPASSANA_TRAINING)

# Train that stage
trainer.train_stage(TrainingStage.VIPASSANA_TRAINING, num_epochs=5)
```

## Stage 2: Noise Strategies

Stage 2 (Vipassana Training) uses three noise strategies to train the trust scorer:

1. **Augmented** (target: `1.0 - severity * noise_level`): Environmental noise via Augmenter
2. **Drunk** (target: `0.0`): Internal dysfunction via drunk_mode
3. **Mismatch** (target: `0.0`): Logical inconsistency via batch shuffling

```python
from samadhi.train import Stage2NoiseStrategy

trainer = SamadhiV4Trainer(
    model=system,
    args=args,
    train_dataset=train_dataset,
    stage=TrainingStage.VIPASSANA_TRAINING,
    noise_level=0.3,  # Noise intensity for Augmenter
)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage` | `INFERENCE` | Initial training stage |
| `noise_level` | `0.3` | Noise intensity for Stage 2 Augmenter |
| `use_label_guidance` | `False` | Enable label guidance in Stage 1 |
| `task_type` | `"classification"` | Task type for GuidanceLoss and Stage 3 |
| `vipassana_margin` | `0.5` | Margin for contrastive loss (Stage 2) |
| `stability_weight` | `0.1` | Weight for stability loss (Stage 1) |
| `guidance_weight` | `1.0` | Weight for guidance loss (Stage 1) |
| `recon_weight` | `1.0` | Weight for reconstruction loss (Stage 0, 1) |

## Dataset Format

```python
def __getitem__(self, idx):
    return {
        "x": self.data[idx],   # Required: input tensor
        "y": self.labels[idx]  # Optional: labels (required for Stage 1 with guidance, Stage 3)
    }
```

## Objectives

Training objectives are located in `samadhi/components/objectives/`:

**Curriculum Training Objectives:**

* **`VipassanaObjective`**: BCE loss for trust score training (Stage 2)
* **`GuidanceLoss`**: Label guidance loss for Stage 1 (classification or regression)
* **`StabilityLoss`**: Energy-based stability loss for Stage 1

**Legacy Objectives (for simpler use cases):**

* **`AutoencoderObjective`**: Pre-training (skips Vitakka/Vicara)
* **`UnsupervisedObjective`**: Reconstruction + stability + entropy regularization
* **`SupervisedClassificationObjective`**: Classification with CrossEntropy loss
* **`SupervisedRegressionObjective`**: Regression with MSE loss
* **`RobustRegressionObjective`**: Regression with Huber loss
* **`AnomalyObjective`**: Anomaly detection with margin-based loss
