# Samadhi Framework Workflow Guide (Cookbook)

This guide provides a comprehensive step-by-step procedure for building, training, and evaluating models using the Samadhi Framework. It is designed to help developers and AI assistants navigate the codebase efficiently to implement new tasks.

---

## 1. Overview & Directory Map

The Samadhi Framework workflow is unique due to its cognitive architecture (Vitakka/Vicara) and typically requires a multi-stage training process.

### Key Directories
* **Configuration (`src/configs/`)**: Defines model hyperparameters. Check here for required fields (e.g., `input_dim`, `seq_len`).
* **Presets (`src/presets/`)**: Factory functions to quickly instantiate standard models (LSTM, Transformer, MLP, CNN). **Start here.**
* **Objectives (`src/train/objectives/`)**: Defines the loss functions for different training phases (Autoencoder, Anomaly, Unsupervised).
* **Trainer (`src/train/hf_trainer.py`)**: The custom `SamadhiTrainer` wrapper.

---

## 2. Step-by-Step Workflow

### Step 1: Analyze Requirements & Data

Before writing code, determine the **Data Type** and **Task Goal**.

| Data Type | Task Goal | Recommended Model | Recommended Objective (Phase 2) |
| :--- | :--- | :--- | :--- |
| **Time Series** | Anomaly Detection | `create_lstm_samadhi` / `create_transformer_samadhi` | `AnomalyObjective` |
| **Tabular** | Anomaly/Classification | `create_mlp_samadhi` | `AnomalyObjective` / `UnsupervisedObjective` |
| **Image** | Reconstruction/Gen | `create_conv_samadhi` | `UnsupervisedObjective` |

### Step 2: Configuration Strategy

You must construct a `SamadhiConfig` object. The recommended approach is to define a dictionary (`config_dict`) and use `SamadhiConfig.from_dict()`.

**Critical:** Certain parameters are **mandatory** and have no defaults. You *must* verify these in `src/configs/*.py`.

* **Adapters (`src/configs/adapters.py`)**:
    * `MlpAdapterConfig`: Requires `input_dim`.
    * `LstmAdapterConfig`: Requires `input_dim`, `seq_len`.
    * `CnnAdapterConfig`: Requires `img_size`, `channels`.
* **Decoders (`src/configs/decoders.py`)**:
    * `ReconstructionDecoderConfig`: Requires `input_dim` (target dim).
    * `LstmDecoderConfig`: Requires `output_dim` (target dim), `seq_len`.

**Example Config Dictionary:**
```python
config_dict = {
    "dim": 32, # Latent dimension
    # Explicitly specify Component Configs
    "adapter": {
        "type": "lstm",     # Must match factory expectation
        "input_dim": 10,    # Mandatory
        "seq_len": 50,      # Mandatory for Time Series
        "hidden_dim": 64
    },
    "decoder": {
        "type": "lstm",
        "output_dim": 10,   # Mandatory (Match input_dim)
        "seq_len": 50,      # Mandatory
    },
    "vitakka": {"n_probes": 10},
    "vicara": {"refine_steps": 5},
    # Note: Objective config will be overwritten per phase
    "objective": {}
}
config = SamadhiConfig.from_dict(config_dict)
````

### Step 3: Data Preparation

Create a custom `torch.utils.data.Dataset`.
The `__getitem__` method **must** return a dictionary compatible with `SamadhiEngine.forward`:

```python
class MyDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "x": self.data[idx],      # Required: Passed to model(x)
            "y": self.labels[idx]     # Optional: Passed to Objective (0=Normal, 1=Anomaly)
        }
```

### Step 4: Model Instantiation

Use a preset factory from `src/presets/` if possible. This handles connecting the Adapter, Vitakka, Vicara, and Decoder.

```python
from src.presets.sequence import create_lstm_samadhi
model = create_lstm_samadhi(config)
```

### Step 5: Phase 1 - Pre-training (Dynamics Learning)

**Goal:** Initialize the latent space structure and learn "Core Concepts" (Attractors).
In this phase, we want to **force strong convergence** to establish the physics of the latent space.

  * **Strategy:** High Stability + High Entropy Penalty.
  * **Trainer:** `SamadhiTrainer` using `AutoencoderObjective` (or `UnsupervisedObjective`).

**Implementation (Independent Config):**
Create a dedicated config for Phase 1. Do not reuse the Phase 2 settings here.

```python
from src.train.objectives.autoencoder import AutoencoderObjective
from src.configs.objectives import ObjectiveConfig
import copy

# 1. Create Independent Config for Phase 1
# We use deepcopy to ensure we don't mess up the base config
phase1_config = copy.deepcopy(config)

# 2. Inject "Force Convergence" Objective Settings
phase1_config.objective = ObjectiveConfig(
    stability_coeff=0.2,      # HIGH: Force strong attraction to S_0
    entropy_coeff=0.5,        # HIGH: Force clear, single-topic selection
    balance_coeff=0.0001,     # LOW: Structure discovery > Load balancing
    anomaly_margin=5.0
)

# 3. Instantiate Objective & Train
# AutoencoderObjective typically skips Vicara loop but respects stability loss if enabled
ae_objective = AutoencoderObjective(phase1_config)

trainer_p1 = SamadhiTrainer(model=model, objective=ae_objective, train_dataset=normal_data, ...)
trainer_p1.train()
```

### Step 6: Probe Initialization (Interim Step)

**Goal:** Initialize Vitakka's probes to meaningful locations in the latent space. This is highly recommended for faster convergence, as it gives the model good starting "concepts."

  * **Methods:**
      * **K-Means (Recommended):** Cluster latent vectors produced by the pre-trained Adapter.
      * **Average:** Use class averages if labels are available.


```python
# 1. Get Latents (use model in eval mode)
model.eval()
z_latents = []
for batch in DataLoader(normal_data, batch_size=32):
    # Just use the adapter to get raw embeddings
    z = model.vitakka.adapter(batch["x"].to(device))
    z_latents.append(z.detach().cpu().numpy())
z_latents = np.concatenate(z_latents, axis=0)

# 2. KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=model.config.vitakka.n_probes)
kmeans.fit(z_latents)

# 3. Update Probes
model.vitakka.probes.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
print("Probes initialized.")
```

### Step 7: Phase 2 - Main Training (Purification & Task)

**Goal:** Train the full cognitive loop (Vitakka search + Vicara refinement) for the specific task (Anomaly Detection, Classification).
In this phase, we **relax the constraints** to allow fine-tuning and flexibility.

  * **Strategy:** Low Stability (allow subtle shifts) + High Balance (prevent collapse).
  * **Objective:** `AnomalyObjective` or `SupervisedObjective`.

**Implementation (Independent Config):**
Create a NEW config for Phase 2.

```python
from src.train.objectives.anomaly import AnomalyObjective

# 1. Create Independent Config for Phase 2
phase2_config = copy.deepcopy(config)

# 2. Inject "Flexible Fine-tuning" Objective Settings
phase2_config.objective = ObjectiveConfig(
    stability_coeff=0.01,     # LOW: Allow gentle updates, don't break the learned physics
    entropy_coeff=0.1,        # LOW: Allow flexible/mixed probe usage if needed
    balance_coeff=0.01,       # HIGH: Ensure all probes are utilized (prevent collapse)
    anomaly_margin=2.0,       # Task specific
    anomaly_weight=1.0
)

# 3. Instantiate Objective & Train
# This objective enables the full Vicara loop
main_objective = AnomalyObjective(phase2_config)

trainer_p2 = SamadhiTrainer(model=model, objective=main_objective, train_dataset=full_data, ...)
trainer_p2.train()
```

### Step 8: Evaluation & Inference

**Goal:** Detect anomalies based on reconstruction error and cognitive stability.

  * **Inference:** You must explicitly enable the cognitive loop.
    ```python
    # Returns: output, s_final, metadata
    # Ensure to handle the case where gate is closed (returns None if handled in wrapper)
    recon, s_final, meta = model(x, run_vitakka=True, run_vicara=True)
    ```
  * **Metrics:**
    1.  **Reconstruction Error:** `MSE(x, recon)`. High = Anomaly.
    2.  **Confidence:** `meta['confidence']`. Low = Anomaly (Gate closed).
    3.  **Stability:** `meta['s_history']` (Movement of $S_t$). Large movement = Anomaly.

-----

## Common Pitfalls & Troubleshooting

1.  **Dimension Mismatch (`RuntimeError: size a matches size b`):**

      * **Cause:** `input_dim` or `seq_len` mismatch between Data, Adapter, and Decoder.
      * **Fix:** Ensure `config.adapter.seq_len` AND `config.decoder.seq_len` match your data.

2.  **AttributeError: 'MlpAdapterConfig' object has no attribute...**

      * **Cause:** You forgot to specify `"type": "..."` in the config dictionary.
      * **Fix:** Explicitly set `"type": "lstm"` (or "cnn", "transformer") in the config dict.

3.  **Loss is NaN:**

      * **Cause:** Exploding gradients in the recursive Vicara loop.
      * **Fix:** Reduce `refine_steps`, or **increase `stability_coeff`** during Phase 1 to force tighter bounds before moving to Phase 2.
