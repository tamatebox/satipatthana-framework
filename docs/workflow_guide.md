# Satipatthana Framework Workflow Guide (Cookbook)

This guide provides a comprehensive step-by-step procedure for building, training, and evaluating models using the Satipatthana Framework. It is designed to help developers and AI assistants navigate the codebase efficiently to implement new tasks.

---

## 1. Overview & Directory Map

The Satipatthana Framework workflow is unique due to its three-engine cognitive architecture (SamathaEngine, VipassanaEngine, ConditionalDecoder) and requires a **4-stage curriculum training** process.

### Key Directories

* **Configuration (`samadhi/configs/`)**: Defines model hyperparameters. Check here for required fields (e.g., `input_dim`, `seq_len`).
* **Core (`samadhi/core/`)**: Main system components (`SamadhiSystem`, `SamathaEngine`, `VipassanaEngine`).
* **Components (`samadhi/components/`)**: Modular components (Adapters, Augmenters, Vitakka, Vicara, Sati, Vipassana, Decoders).
* **Trainer (`samadhi/train/v4_trainer.py`)**: The `SamadhiV4Trainer` implementing 4-stage curriculum.

---

## 2. Architecture Overview

### Three-Engine Structure

```txt
Raw Input (X)
    ↓
[SamathaEngine] — Convergent Thinking
    Augmenter → Adapter → Vitakka → Vicara loop (w/ Sati) → S*, SantanaLog
    ↓
[VipassanaEngine] — Introspective Self-Awareness
    S* + SantanaLog → V_ctx, α (trust_score)
    ↓
[ConditionalDecoder] — Humble Expression
    S* + V_ctx → Output (Y)
```

### 4-Stage Curriculum

| Stage | Name | Trainable | Objective |
|:---|:---|:---|:---|
| **0** | Adapter Pre-training | Adapter, AdapterReconHead | Reconstruction Loss |
| **1** | Samatha Training | Adapter, Vitakka, Vicara, Sati, (SamathaReconHead, AuxHead) | Stability + Recon + (Guidance) |
| **2** | Vipassana Training | Vipassana | BCE (Contrastive) |
| **3** | Decoder Fine-tuning | ConditionalDecoder | Task Specific Loss |

---

## 3. Step-by-Step Workflow

### Step 1: Analyze Requirements & Data

Before writing code, determine the **Data Type** and **Task Goal**.

| Data Type | Task Goal | Recommended Adapter | Stage 1 Strategy | Stage 3 Decoder |
| :--- | :--- | :--- | :--- | :--- |
| **Time Series** | Anomaly Detection | LSTM / Transformer | Reconstruction Only | Identity |
| **Tabular** | Anomaly Detection | MLP | Reconstruction Only | Identity |
| **Tabular** | Classification | MLP | Guidance (CE Loss) | Conditional (Softmax) |
| **Tabular** | Regression | MLP | Guidance (MSE Loss) | Conditional (Linear) |
| **Image** | Classification | CNN | Guidance (CE Loss) | Conditional (Softmax) |

### Step 2: Configuration Strategy

Construct a `SystemConfig` object using factory functions.

**Critical:** Certain parameters are **mandatory** and have no defaults. Verify these in `samadhi/configs/*.py`.

* **Adapters (`samadhi/configs/adapters.py`)**:
  * `MlpAdapterConfig`: Requires `input_dim`.
  * `LstmAdapterConfig`: Requires `input_dim`, `seq_len`.
  * `CnnAdapterConfig`: Requires `img_size`, `channels`.
  * `TransformerAdapterConfig`: Requires `input_dim`, `seq_len`.
* **Decoders (`samadhi/configs/decoders.py`)**:
  * `ReconstructionDecoderConfig`: Requires `input_dim`.
  * `ConditionalDecoderConfig`: Requires `dim`, `context_dim`, `output_dim`.
* **Vipassana (`samadhi/configs/vipassana.py`)**:
  * `StandardVipassanaConfig`: Requires `context_dim`.

**Example Config (MLP for Classification):**

```python
from samadhi.configs import SystemConfig, SamathaConfig, VipassanaEngineConfig
from samadhi.configs import create_adapter_config, create_vicara_config
from samadhi.configs import AugmenterConfig, VitakkaConfig, SatiConfig
from samadhi.configs import StandardVipassanaConfig
from samadhi.configs.enums import AugmenterType, SatiType

config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=64,
        adapter=create_adapter_config("mlp", input_dim=784, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.3),
        vitakka=VitakkaConfig(num_probes=16, temperature=0.2),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
        max_steps=10,
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32, latent_dim=64),
    ),
    use_label_guidance=True,  # Enable Stage 1 Guidance
    seed=42,
)
```

**Example Config (LSTM for Time Series Anomaly Detection):**

```python
config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=128,
        adapter=create_adapter_config(
            "lstm",
            input_dim=10,
            seq_len=50,
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

### Step 3: Data Preparation

Create a custom `torch.utils.data.Dataset`.
The `__getitem__` method **must** return a dictionary compatible with `SamadhiSystem.forward`:

```python
class MyDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "x": self.data[idx],      # Required: input tensor
            "y": self.labels[idx]     # Optional: labels (for Stage 1 guidance, Stage 3)
        }
```

### Step 4: Model Instantiation

Instantiate `SamadhiSystem` with your config:

```python
from samadhi.core.system import SamadhiSystem

system = SamadhiSystem(config)
```

### Step 5: 4-Stage Curriculum Training

Use `SamadhiV4Trainer` to run the full curriculum:

```python
from samadhi.train import SamadhiV4Trainer
from transformers import TrainingArguments

# Training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    learning_rate=1e-3,
)

# Initialize Trainer
trainer = SamadhiV4Trainer(
    model=system,
    args=args,
    train_dataset=dataset,
)

# Run full 4-stage curriculum
results = trainer.run_curriculum(
    stage0_epochs=5,   # Adapter pre-training
    stage1_epochs=10,  # Samatha training (convergence)
    stage2_epochs=5,   # Vipassana training (meta-cognition)
    stage3_epochs=5,   # Decoder fine-tuning
)
```

### Stage-by-Stage Details

#### Stage 0: Adapter Pre-training

**Goal:** Learn basic input encoding.

* **Trainable:** Adapter, AdapterReconHead
* **Objective:** $\mathcal{L}_0 = \mathcal{L}_{recon}(X, \hat{X}_{adapter})$
* **Note:** Vitakka/Vicara are bypassed. Only Adapter + Reconstruction Head are trained.

#### Stage 1: Samatha Training

**Goal:** Learn convergent dynamics and stable attractors.

* **Trainable:** Adapter, Vitakka, Vicara, Sati, SamathaReconHead, (AuxHead if guidance enabled)
* **Objective:** $\mathcal{L}_1 = ||S_T - S_{T-1}||^2 + \lambda_r \mathcal{L}_{recon} + \lambda_g \mathcal{L}_{task}$
* **Key:** AuxHead provides task guidance but is **discarded after Stage 1** (not transferred to Stage 3).

#### Stage 2: Vipassana Training

**Goal:** Train meta-cognition to recognize "good" vs "bad" thinking.

* **Trainable:** Vipassana (LogEncoder + ConfidenceMonitor)
* **Objective:** $\mathcal{L}_2 = \text{BCE}(\alpha, \hat{\alpha})$
* **Data Generation:** Three strategies are used:
    1. **Augmented Path:** Add noise to input → Target: `1.0 - severity`
    2. **Drunk Path:** Perturb SamathaEngine internals → Target: `0.0`
    3. **Mismatch Path:** Shuffle S\* and SantanaLog → Target: `0.0`

#### Stage 3: Decoder Fine-tuning

**Goal:** Train ConditionalDecoder for final task output.

* **Trainable:** ConditionalDecoder only
* **Objective:** $\mathcal{L}_3 = \mathcal{L}_{task}(y, \text{Decoder}(S^*, V_{ctx}))$
* **Note:** ConditionalDecoder input is `S* ⊕ V_ctx` (dim = d + c), different from Stage 1 AuxHead (dim = d).

### Step 6: Evaluation & Inference

**Goal:** Run three-phase inference with trust score.

```python
# Three-phase inference
system.eval()
result = system(x)

# Access outputs
output = result.output           # Decoded result
s_star = result.s_star           # Converged latent state
v_ctx = result.v_ctx             # Vipassana context vector
trust_score = result.trust_score # Confidence (0.0-1.0)
santana = result.santana         # Thinking trajectory (SantanaLog)
```

**Using Trust Scores:**

```python
if result.trust_score > 0.8:
    # High confidence - use output directly
    prediction = result.output
else:
    # Low confidence - take safety measures
    print("Warning: Low confidence prediction")
    # Options: trigger fallback, widen output variance, abstain
```

**Anomaly Detection (Unsupervised):**

For unsupervised anomaly detection, the trust score (`α`) from Vipassana serves as the anomaly score:

```python
# Low trust = anomaly
anomaly_score = 1.0 - result.trust_score
is_anomaly = anomaly_score > threshold
```

---

## 4. Common Pitfalls & Troubleshooting

1. **Dimension Mismatch (`RuntimeError: size a matches size b`):**

    * **Cause:** `input_dim` or `seq_len` mismatch between Data, Adapter, and Decoder.
    * **Fix:** Ensure `config.samatha.adapter.seq_len` matches your data dimensions.

2. **AttributeError: 'MlpAdapterConfig' object has no attribute...**

    * **Cause:** Wrong adapter type specified.
    * **Fix:** Use `create_adapter_config("mlp", ...)` factory function.

3. **Loss is NaN:**

    * **Cause:** Exploding gradients in the recursive Vicara loop.
    * **Fix:** Reduce `max_steps`, increase `beta` (inertia), or extend Stage 0 training.

4. **Stage 3 Decoder Dimension Error:**

    * **Cause:** ConditionalDecoder expects `dim + context_dim` input.
    * **Fix:** Ensure `ConditionalDecoderConfig.dim` matches `latent_dim` and `context_dim` matches Vipassana's `context_dim`.

5. **Low Trust Scores on All Inputs:**

    * **Cause:** Stage 2 training insufficient or data imbalance.
    * **Fix:** Extend Stage 2 epochs, ensure balanced Augmented/Drunk/Mismatch data generation.

---

## 5. Quick Reference: Config Mandatory Fields

| Config Class | Mandatory Fields |
|:---|:---|
| `MlpAdapterConfig` | `input_dim` |
| `LstmAdapterConfig` | `input_dim`, `seq_len` |
| `CnnAdapterConfig` | `img_size`, `channels` |
| `TransformerAdapterConfig` | `input_dim`, `seq_len` |
| `ReconstructionDecoderConfig` | `input_dim` |
| `ConditionalDecoderConfig` | `dim`, `context_dim`, `output_dim` |
| `StandardVipassanaConfig` | `context_dim` |
