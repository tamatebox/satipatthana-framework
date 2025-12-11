# Satipatthana Training Strategy Guide

This document outlines the **4-Stage Curriculum Training** strategy for the **Satipatthana Framework**. The framework uses a progressive training approach that builds stable convergence, meta-cognition, and task-specific decoding in sequence.

-----

## 🎯 4-Stage Curriculum Overview

Satipatthana's three-engine architecture (SamathaEngine, VipassanaEngine, ConditionalDecoder) is trained progressively to ensure stable learning.

![Training Overview](diagrams/images/v4_sequence_diagram_training_overview.png)

| Stage | Name | Trainable Components | Frozen Components | Objective |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Adapter Pre-training | Adapter, AdapterReconHead | All others | Reconstruction Loss |
| **1** | Samatha Training | Adapter, Vitakka, Vicara, Sati, SamathaReconHead, (AuxHead) | Vipassana, ConditionalDecoder | Stability + Recon + (Guidance) |
| **2** | Vipassana Training | Vipassana | All others | Triple Score BCE |
| **3** | Decoder Fine-tuning | ConditionalDecoder | All others | Task Specific Loss |

-----

## 📋 Stage Details

### Stage 0: Adapter Pre-training

![Stage 0](diagrams/images/v4_sequence_diagram_training_stage0.png)

**Goal:** Learn basic input encoding to latent space.

* **Trainable:** Adapter, AdapterReconHead
* **Objective:** $\mathcal{L}_0 = \mathcal{L}_{recon}(X, \hat{X}_{adapter})$
* **Data:** All available data (labels not required)
* **Notes:**
  * Vitakka/Vicara loop is bypassed
  * Only Adapter → ReconHead path is active
  * Establishes initial latent space structure

### Stage 1: Samatha Training

![Stage 1](diagrams/images/v4_sequence_diagram_training_stage1.png)

**Goal:** Learn convergent dynamics, stable attractors, and task-aligned representation.

* **Trainable:** Adapter, Vitakka, Vicara, Sati, SamathaReconHead, (AuxHead if `use_label_guidance=True`)
* **Objective:**
$$\mathcal{L}_1 = ||S_T - S_{T-1}||^2 + \lambda_r \mathcal{L}_{recon}(X, \hat{X}_{samatha}) + \lambda_g \mathcal{L}_{task}(y, \text{AuxHead}(S^*))$$

* **Loss Components:**
  * **Stability Loss:** Forces convergence ($ ||S_T - S_{T-1}||^2 $)
  * **Reconstruction Loss:** Preserves input information
  * **Guidance Loss (optional):** Aligns $S^*$ with task labels via AuxHead

* **AuxHead vs ConditionalDecoder:**

| Module | Input Dimension | Purpose | After Stage 1 |
|:---|:---|:---|:---|
| `AuxHead` | $d$ (S\* only) | Guidance learning | **Discarded** |
| `ConditionalDecoder` | $d + c$ (S\* ⊕ V_ctx) | Final inference | Trained in Stage 3 |

**Important:** AuxHead weights are NOT transferred to Stage 3. ConditionalDecoder is trained from scratch.

### Stage 2: Vipassana Training

![Stage 2](diagrams/images/v4_sequence_diagram_training_stage2.png)

**Goal:** Train meta-cognition to recognize "good" vs "bad" thinking processes using Triple Score System.

* **Trainable:** Vipassana (GRU Encoder + 8 Grounding Metrics + Triple Score Heads)
* **Objective:** Triple Score BCE
$$\mathcal{L}_2 = \text{BCE}(\text{trust}, \hat{\alpha}) + \text{BCE}(\text{conformity}, \hat{\alpha}) + \text{BCE}(\text{confidence}, \hat{\alpha})$$

**Triple Score System:**

| Score | Input | GRU Gradient | Purpose |
|:---|:---|:---|:---|
| `trust_score` | h_static (metrics) | ✗ | Pure OOD detection |
| `conformity_score` | h_dynamic (GRU) | ✓ | Trajectory process anomaly |
| `confidence_score` | h_static + h_dynamic | ✓ | Comprehensive assessment |

This separation ensures the GRU encoder receives proper gradients during training while maintaining pure OOD detection capability.

#### Noise Generation Strategies

Four data generation strategies teach Vipassana to detect anomalous thinking:

| Strategy | Description | Target α |
|:---|:---|:---|
| **Augmented Path** | Add noise to input data | `1.0 - severity` |
| **Drunk Path** | Perturb SamathaEngine internals | `0.0` |
| **Mismatch Path** | Shuffle S\* and SantanaLog within batch | `0.0` |
| **Void Path** | Genuine OOD samples (VoidDataset) | `0.0` |

**Drunk Path Implementations:**

* Increase Dropout rate in Vicara Refiner
* Add temporary noise to Refiner weights
* Disturb Vitakka's temperature parameter

**Void Path (OOD Detection):**

* Uses `VoidDataset` or `FilteredNoiseVoid` for genuine out-of-distribution samples
* Trains **Grounding Metrics** (`s0_min_dist`, `drift_magnitude`, `recon_error`)
* Critical for detecting inputs that converge to familiar regions but are actually OOD

![Noise Generation](diagrams/images/v4_sequence_diagram_noise_generation.png)

### Stage 3: Decoder Fine-tuning

![Stage 3](diagrams/images/v4_sequence_diagram_training_stage3.png)

**Goal:** Train ConditionalDecoder for final task output.

* **Trainable:** ConditionalDecoder only
* **Objective:** $\mathcal{L}_3 = \mathcal{L}_{task}(y, \text{Decoder}(S^*, V_{ctx}))$
* **Input:** Concatenation of S\* and V_ctx (dimension: $d + c$)

-----

## 🚀 Task-Specific Training Roadmaps

### Case 1: Anomaly Detection (Unsupervised)

**Goal:** Detect outliers using trust score (α) from Vipassana.

| Stage | Configuration | Notes |
|:---|:---|:---|
| 0 | Standard | - |
| 1 | `use_label_guidance=False`, Reconstruction only | Learn normal patterns |
| 2 | All three noise strategies | Train trust score |
| 3 | Skip or Identity Decoder | α is the final output |

**Inference:** `anomaly_score = 1.0 - trust_score`

### Case 2: Supervised Classification

**Goal:** High accuracy classification with explainable confidence.

| Stage | Configuration | Notes |
|:---|:---|:---|
| 0 | Standard | - |
| 1 | `use_label_guidance=True`, CE Loss for AuxHead | Align latent space with classes |
| 2 | Standard | Train confidence estimation |
| 3 | ConditionalDecoder with Softmax output | Task-specific head |

**Inference:** Use trust score to filter low-confidence predictions.

#### Two-Stage Inference Strategy

For imbalanced classification (e.g., fraud detection), a **two-stage inference strategy** is recommended:

```text
All samples
    ↓ Stage 1: Trust Score Filter
    ↓ (Trust Score < threshold → "flagged")
Flagged samples (low trust)
    ↓ Stage 2: Decoder Classification
Final prediction: Class label
```

**Benefits:**

1. **Efficiency:** 99%+ samples auto-processed, only "flagged" samples go through detailed classification
2. **Imbalance mitigation:** Fraud ratio increases after filtering
3. **Explainability:** Trust Score explains why a sample is flagged

**Why Full-Data Training Works:**

A common question: "Should the Decoder be trained only on low-trust samples?"

The answer is **no**. ConditionalDecoder receives `S* + V_ctx` as input. `V_ctx` (Vipassana Context) contains the "embedding of doubt", so the Decoder can learn:

* When `V_ctx` is clear (high trust) → likely Normal
* When `V_ctx` is murky (low trust) → focus on subtle differences in `S*`

This means the two-stage approach is an **inference strategy**, not a training strategy. The Decoder learns conditional behavior through `V_ctx` without requiring filtered training data.

**Threshold Selection (F2 Score):**

For fraud detection, use **F2 score** (Recall-weighted) instead of F1. This prioritizes catching more frauds over precision:

```python
from sklearn.metrics import fbeta_score

# Find threshold that maximizes F2 score on train data
# F2 weighs Recall 2x more than Precision - better for fraud detection
best_f2, best_threshold = 0, 0.5
for th in np.linspace(0.1, 0.95, 50):
    preds = (train_trust_scores < th).astype(int)
    f2 = fbeta_score(train_labels, preds, beta=2)
    if f2 > best_f2:
        best_f2, best_threshold = f2, th
```

#### Advanced: Trust-Weighted Loss (Optional)

For further optimization, apply trust-based weighting during Stage 3:

$$\mathcal{L}_{decoder} = (1 - \alpha)^\gamma \cdot \mathcal{L}_{CE}(y, \hat{y})$$

This makes the Decoder focus on "hard" samples (low trust) while still learning from all data.

### Case 3: Time Series Forecasting

**Goal:** Predict future values with uncertainty estimation.

| Stage | Configuration | Notes |
|:---|:---|:---|
| 0 | LSTM/Transformer Adapter | Encode temporal patterns |
| 1 | MSE reconstruction, optional guidance | Learn sequence dynamics |
| 2 | Standard | Train prediction confidence |
| 3 | ConditionalDecoder with linear output | Regression head |

**Inference:** When trust score < threshold, widen prediction intervals.

### Case 4: LLM Hallucination Detection

**Goal:** Detect "confident lies" in language model outputs.

| Stage | Configuration | Notes |
|:---|:---|:---|
| 0 | Transformer Adapter on LLM hidden states | - |
| 1 | Stability-focused, no guidance | Learn context consistency |
| 2 | **Critical stage** - strong Mismatch training | Detect logical inconsistency |
| 3 | Skip (Vipassana output is the result) | α indicates hallucination risk |

-----

## 📝 Implementation with SatipatthanaTrainer

```python
from satipatthana.train import SatipatthanaTrainer
from satipatthana.core.system import SatipatthanaSystem
from transformers import TrainingArguments

# Build system
system = SatipatthanaSystem(config)

# Training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=32,
)

# Initialize trainer
trainer = SatipatthanaTrainer(
    model=system,
    args=args,
    train_dataset=dataset,
)

# Run full curriculum
results = trainer.run_curriculum(
    stage0_epochs=5,   # Adapter pre-training
    stage1_epochs=10,  # Samatha training
    stage2_epochs=5,   # Vipassana training
    stage3_epochs=5,   # Decoder fine-tuning
)

# Or run individual stages
trainer.run_stage(stage=1, epochs=10)
```

-----

## ⚙️ Hyperparameter Guidelines

### Stage 1 Coefficients

| Parameter | Recommended | Purpose |
|:---|:---|:---|
| `stability_coeff` | 0.1 - 0.5 | Higher = stronger convergence force |
| `recon_coeff` | 0.1 - 0.3 | Higher = better input preservation |
| `guidance_coeff` | 0.1 - 0.5 | Higher = stronger task alignment |

### Stage 2 Data Balance

| Data Type | Recommended Ratio |
|:---|:---|
| Clean (no noise) | 20% |
| Augmented (varying severity) | 20% |
| Drunk Path | 20% |
| Mismatch Path | 20% |
| Void Path (OOD) | 20% |

### General Tips

1. **Stage 0 duration:** Ensure reconstruction loss converges before moving to Stage 1
2. **Stage 1 stability:** If NaN loss occurs, increase `stability_coeff` or reduce `max_steps`
3. **Stage 2 balance:** Ensure sufficient Drunk/Mismatch examples to prevent Vipassana from always outputting high α
4. **Stage 3 learning rate:** Use lower LR (e.g., 1e-4) as Samatha/Vipassana are frozen

-----

## 🔄 Checkpoint Strategy

Save checkpoints after each stage for flexibility:

```python
# After Stage 1
trainer.save_model("./checkpoints/stage1")

# After Stage 2
trainer.save_model("./checkpoints/stage2")

# Final model
trainer.save_model("./checkpoints/final")
```

This allows:

* Retraining Stage 2/3 with different strategies
* A/B testing different Decoder configurations
* Debugging stage-specific issues
