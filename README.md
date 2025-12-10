# Samadhi Framework (Deep Convergence Architecture)

> **"From Chaos to Essence."**

![Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Samadhi Framework** is a modular **recursive attention architecture** designed for **extracting the essential structure** and **stabilizing the internal state** from complex, noisy data.

Instead of expanding information horizontally (generation), it implements a vertical deepening (insight) that reduces information entropy to reach a stable, meaningful state (Attractor).

Now evolved into a **Meta-Framework**, it allows you to compose **Adapters** (Input), **Vitakka** (Search), **Vicāra** (Refinement), and **Decoders** (Output) to apply this convergence philosophy to any domain—Tabular, Vision, Time Series, and NLP.

---

## 🧘 Concept & Philosophy

**Samadhi** is a "convergent" engine that transitions its state towards a dynamical system's attractor (fixed point).

It implements the process of meditative concentration (Samadhi) in Buddhist psychology as the following engineering modules:

| Module | Buddhist Term | Engineering Concept | Function |
| :--- | :--- | :--- | :--- |
| **Vitakka** | 尋 (Initial Application) | **Active Probing** | Searches and captures "intentions (Probes)" from chaotic input. |
| **Sati** | 正知 (Clear Comprehension) | **Gating Mechanism** | Detects noise and hallucinations, blocking further processing. |
| **Vicāra** | 伺 (Sustained Application) | **Recurrent Refinement** | Blocks external input and minimizes state energy (purifies) through a recursive loop. |
| **Santāna** | 相続 (Continuity) | **State Dynamics Log** | Tracks the temporal transitions of intentions (concentration, shift, dispersion). |

> 📖 For detailed architecture specifications, see [Japanese](docs/model.md) / [English](docs/model_en.md).

> 📜 For theoretical foundations, see [Japanese Theory](docs/theory/jp.md) / [English Theory](docs/theory/en.md).

---

## 🚀 Key Features

* **Modular Framework:** Easily swap Adapters (CNN, LSTM, MLP) and Decoders to fit any data modality.
* **Type-Safe Configuration:** Robust configuration management using Dataclasses and Enums for better validation and developer experience.
* **Objective-Driven Training:** Flexible training strategies (Autoencoder, Anomaly Detection, Supervised) by simply switching the `Objective` component.
* **Convergence:** The output is not a text stream, but a single "Purified State" with minimized entropy.
* **O(1) Inference:** Inference cost does not depend on the input length (Context Length), but only on the number of convergence steps (a constant).
* **Explainability (XAI):** "Why a particular subject was focused on" and "how concentration deepened" are fully visualized as logs.

---

## 🌟 Potential Applications

The unique properties of the Samadhi Framework make it suitable for tasks requiring deep insight and state stability rather than simple generation.

1. **Biosignal Analysis (Healthcare):** Extract stable physiological states (e.g., stress levels, cognitive load) from noisy EEG or heart rate data.
2. **Anomaly Detection (Forensics):** Identify "essential anomalies" in financial transactions or machine logs by converging normal patterns and detecting deviations.
3. **Human Intent Analysis (UX/Psychology):** Capture deep user intent or emotional shifts from interactions, beyond surface-level keywords.
4. **Autonomous Agents (Robotics):** Enable stable decision-making in chaotic environments by converging sensory inputs into clear actionable states.
5. **Creative Assistance (Structure Extraction):** Distill core concepts or themes from multiple creative drafts.

---

## 📂 Project Structure

```bash
.
├── data/               # Datasets
├── docs/               # Theoretical specifications and plans
├── notebooks/          # Experiments and Analysis (Jupyter)
├── samadhi/
│   ├── configs/        # Configuration System ([Details](samadhi/configs/README.md))
│   │   ├── main.py     # Root SamadhiConfig
│   │   ├── factory.py  # Config Factories
│   │   ├── adapters.py # Adapter Configs
│   │   └── ...
│   ├── components/     # Modularized Components ([Details](samadhi/components/README.md))
│   │   ├── adapters/   # Input Adapters (MLP, CNN, LSTM, Transformer)
│   │   ├── decoders/   # Output Decoders
│   │   ├── vitakka/    # Search Modules
│   │   ├── vicara/     # Refinement Modules
│   │   └── refiners/   # Core refinement networks (MLP, GRU)
│   ├── core/           # Core Engine and Builder
│   ├── presets/        # Factory functions for standard configurations (Tabular, Vision, Sequence) ([Details](samadhi/presets/README.md))
│   ├── train/          # Training Logic ([Details](samadhi/train/README.md))
│   │   └── trainer.py  # 4-Stage Curriculum Trainer
│   └── utils/          # Utility functions
├── tests/              # Unit Tests
├── main.py             # Entry point
└── pyproject.toml      # Project configuration (uv)
```

### Logging

The Samadhi Framework utilizes a centralized logging system managed by `samadhi/utils/logger.py`. For consistent logging behavior across the project, please refer to the detailed guidelines and setup instructions in [docs/logging.md](docs/logging.md).

-----

## ⚡ Quick Start

> 📖 **For detailed instructions on model construction, training, and evaluation, please refer to the [Workflow Guide](docs/workflow_guide.md).**

### Prerequisites

This project uses `uv` as its package manager.

```bash
# Install dependencies
uv sync
```

### 1. Basic Usage (Using Presets)

Easily create a model for your data type using presets. Configuration can be passed as a dictionary (automatically converted) or a `SamadhiConfig` object.

```python
import torch
from samadhi.presets.tabular import create_mlp_samadhi
from samadhi.configs.main import SamadhiConfig

# Configuration (Dictionary)
config_dict = {
    "dim": 64,
    "adapter": {"type": "mlp", "input_dim": 10},
    "decoder": {"type": "reconstruction", "input_dim": 10},
    "vitakka": {"n_probes": 5, "gate_threshold": 0.5},
    "vicara": {"refine_steps": 5},
    "objective": {"stability_coeff": 0.01, "entropy_coeff": 0.1, "balance_coeff": 0.001}
}

# Create a tabular model
# The dictionary is automatically converted to SamadhiConfig internally
model = create_mlp_samadhi(config_dict)

# Or using SamadhiConfig directly for type safety
# config = SamadhiConfig.from_dict(config_dict)
# model = create_mlp_samadhi(config)

# Inference
input_data = torch.randn(1, 10)
output, s_final, meta = model(input_data)

print(f"Purified state shape: {s_final.shape}")
print(f"Reconstructed output shape: {output.shape}")
```

### 2. Custom Construction (Using Builder)

Build a custom model by mixing and matching components.

```python
from samadhi.core.builder import SamadhiBuilder
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.enums import AdapterType, DecoderType
from samadhi.configs.objectives import ObjectiveConfig

# Load config
config_data = {
    "dim": 32,
    "adapter": {"type": AdapterType.CNN.value, "channels": 3, "img_size": 32},
    "decoder": {"type": DecoderType.CNN.value, "channels": 3, "img_size": 32, "input_dim": 32*32*3}, # input_dim needed for linear layer before deconv
    "vitakka": {"n_probes": 5},
    "objective": {"stability_coeff": 0.05, "anomaly_margin": 5.0}
}
config = SamadhiConfig.from_dict(config_data)

# Builder uses the config to instantiate appropriate components
model = SamadhiBuilder(config) \
    .set_adapter(type=AdapterType.CNN.value) \
    .set_vitakka() \
    .set_vicara(refiner_type="mlp") \
    .set_decoder(type=DecoderType.CNN.value) \
    .build()
```

### 3. Training (4-Stage Curriculum)

Train using the 4-stage curriculum trainer with Hugging Face integration.

```python
from samadhi.train import SamadhiV4Trainer
from samadhi.core.system import SamadhiSystem, TrainingStage
from transformers import TrainingArguments

# Build SamadhiSystem (see samadhi/core/system.py for details)
system = SamadhiSystem(...)

# Training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=32,
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
    stage1_epochs=10,  # Samatha training
    stage2_epochs=5,   # Vipassana training
    stage3_epochs=5,   # Decoder fine-tuning
)
```

---

## 📚 Notebook Demos

The `notebooks/` directory contains various Jupyter Notebooks that demonstrate the Samadhi Framework's capabilities across different domains. These notebooks provide hands-on examples for understanding model behavior, exploring applications, and experimenting with various configurations.

To run these demos, ensure you have `jupyter lab` installed (`uv pip install "jupyterlab>=3"` if not already installed) and then navigate to the `notebooks/` directory.

### Available Demos

* **MNIST Demo (`mnist_demo.ipynb`):** Visualizes the "purification" process of noisy MNIST digits, showcasing the convergence property.
* **Fraud Detection Demo (`fraud_unsupervised_detection_explained.ipynb`):** An example of applying the Samadhi Model for fraud detection using unsupervised learning.
* **Time Series Anomaly Detection Demo (`time_series_anomaly_detection.ipynb`):** Demonstrates anomaly detection on time series data.

### How to Run

1. **Install Jupyter Lab (if not already installed):**

    ```bash
    uv pip install "jupyterlab>=3"
    ```

2. **Start Jupyter Lab from the project root:**

    ```bash
    jupyter lab
    ```

3. **Navigate to the `notebooks/` directory and open any of the `.ipynb` files.**

---

## 🛠 Roadmap

* [x] **v1.0:** Theoretical Definition (Concept Proof)
* [x] **v2.2:** Waveform Simulation (Vitakka/Vicāra Implemented)
* [x] **v2.3:** Gating & Meta-Cognition (Sati Implemented)
* [x] **v2.4:** Anomaly Detection & Time Series Support
* [x] **v3.0:** **Framework Refactoring** (Modularization, Builder, HF Trainer)
* [x] **v3.1:** **Configuration Refactoring** (Type-Safe Configs, Factory Pattern)
* [x] **v4.0:** **Meta-Cognition Architecture** (Vipassana, SamadhiSystem, 4-Stage Curriculum Training)
* [ ] **Future:** NLP Implementation (Text Summarization/Concept Extraction)
* [ ] **Future:** Multi-Agent Samadhi (Dialogue of Insight)

-----

## 📜 License

MIT License
