# Samadhi Framework (Deep Convergence Architecture)

> **"From Generation to Convergence."**

![Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Samadhi Framework** is a modular **recursive attention architecture** designed not for traditional "sequence prediction (Next Token Prediction)" seen in generative AI, but for "extracting the essential structure" and "stabilizing the internal state" of the subject.

It engineeringly implements a vertical deepening of information (quiet insight) rather than horizontal expansion (talkative generation).

Now evolved into a **Meta-Framework**, it allows you to compose **Adapters** (Input), **Vitakka** (Search), **Vicāra** (Refinement), and **Decoders** (Output) to apply this convergence philosophy to any domain—Tabular, Vision, Time Series, and NLP.

---

## 🧘 Concept & Philosophy

Modern LLMs (Transformers) have a "divergent" nature, generating tokens one after another by riding the waves of probability distributions. In contrast, **Samadhi** is a "convergent" engine that transitions its state towards a dynamical system's attractor (fixed point).

It implements the process of meditative concentration (Samadhi) in Buddhist psychology as the following engineering modules:

| Module | Buddhist Term | Engineering Concept | Function |
| :--- | :--- | :--- | :--- |
| **Vitakka** | 尋 (Initial Application) | **Active Probing** | Searches and captures "intentions (Probes)" from chaotic input. |
| **Sati** | 正知 (Clear Comprehension) | **Gating Mechanism** | Detects noise and hallucinations, blocking further processing. |
| **Vicāra** | 伺 (Sustained Application) | **Recurrent Refinement** | Blocks external input and minimizes state energy (purifies) through a recursive loop. |
| **Santāna** | 相続 (Continuity) | **State Dynamics Log** | Tracks the temporal transitions of intentions (concentration, shift, dispersion). |

> 📖 For detailed architecture specifications, see [docs/model.md](docs/model.md).

---

## 🚀 Key Features

*   **Modular Framework:** Easily swap Adapters (CNN, LSTM, MLP) and Decoders to fit any data modality.
*   **Type-Safe Configuration:** Robust configuration management using Dataclasses and Enums for better validation and developer experience.
*   **Objective-Driven Training:** Flexible training strategies (Autoencoder, Anomaly Detection, Supervised) by simply switching the `Objective` component.
*   **Convergence:** The output is not a text stream, but a single "Purified State" with minimized entropy.
*   **O(1) Inference:** Inference cost does not depend on the input length (Context Length), but only on the number of convergence steps (a constant).
*   **Explainability (XAI):** "Why a particular subject was focused on" and "how concentration deepened" are fully visualized as logs.

---

## 🌟 Potential Applications

The unique properties of the Samadhi Framework make it suitable for tasks requiring deep insight and state stability rather than simple generation.

1.  **Biosignal Analysis (Healthcare):** Extract stable physiological states (e.g., stress levels, cognitive load) from noisy EEG or heart rate data.
2.  **Anomaly Detection (Forensics):** Identify "essential anomalies" in financial transactions or machine logs by converging normal patterns and detecting deviations.
3.  **Human Intent Analysis (UX/Psychology):** Capture deep user intent or emotional shifts from interactions, beyond surface-level keywords.
4.  **Autonomous Agents (Robotics):** Enable stable decision-making in chaotic environments by converging sensory inputs into clear actionable states.
5.  **Creative Assistance (Structure Extraction):** Distill core concepts or themes from multiple creative drafts.

---

## 📂 Project Structure

```bash
.
├── data/               # Datasets
├── docs/               # Theoretical specifications and plans
├── notebooks/          # Experiments and Analysis (Jupyter)
├── src/
│   ├── configs/        # Configuration System ([Details](src/configs/README.md))
│   │   ├── main.py     # Root SamadhiConfig
│   │   ├── factory.py  # Config Factories
│   │   ├── adapters.py # Adapter Configs
│   │   └── ...
│   ├── components/     # Modularized Components ([Details](src/components/README.md))
│   │   ├── adapters/   # Input Adapters (MLP, CNN, LSTM, Transformer)
│   │   ├── decoders/   # Output Decoders
│   │   ├── vitakka/    # Search Modules
│   │   ├── vicara/     # Refinement Modules
│   │   └── refiners/   # Core refinement networks (MLP, GRU)
│   ├── core/           # Core Engine and Builder
│   ├── presets/        # Factory functions for standard configurations (Tabular, Vision, Sequence) ([Details](src/presets/README.md))
│   ├── train/          # Training Logic ([Details](src/train/README.md))
│   │   ├── hf_trainer.py # Hugging Face Trainer Wrapper
│   │   └── objectives/   # Pluggable Training Objectives
│   └── utils/          # Utility functions
├── tests/              # Unit Tests
├── main.py             # Entry point
└── pyproject.toml      # Project configuration (uv)
```

### Logging
The Samadhi Framework utilizes a centralized logging system managed by `src/utils/logger.py`. For consistent logging behavior across the project, please refer to the detailed guidelines and setup instructions in [docs/logging.md](docs/logging.md).

-----

## ⚡ Quick Start

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
from src.presets.tabular import create_mlp_samadhi
from src.configs.main import SamadhiConfig

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
from src.core.builder import SamadhiBuilder
from src.configs.main import SamadhiConfig
from src.configs.enums import AdapterType, DecoderType
from src.configs.objectives import ObjectiveConfig

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

### 3. Training (Objective-Driven)

Train the model using the Hugging Face compatible Trainer and pluggable Objectives.

```python
from src.train import SamadhiTrainer
from src.train.objectives.unsupervised import UnsupervisedObjective
from src.configs.main import SamadhiConfig

# Assuming 'config' is a SamadhiConfig object created previously
# Define Objective (e.g., Unsupervised Learning: Reconstruction + Stability)
objective = UnsupervisedObjective(config)

# Initialize Trainer
trainer = SamadhiTrainer(
    model=model,
    args=training_args, # Hugging Face TrainingArguments
    objective=objective,
    train_dataset=dataset
)

# Train
trainer.train()
```

-----

## 📊 Architecture Comparison

| Feature | Transformer (GPT) | Samadhi Framework (Ours) |
| :--- | :--- | :--- |
| **Vector Flow** | Divergence | Convergence |
| **Time Complexity** | $O(N^2)$ (Quadratic) | $O(1)$ (Constant/Iterative) |
| **Dependency** | Context History | Current State Only (Markov) |
| **Objective** | Likelihood Maximization | Stability Energy Minimization |
| **Output** | Probability Distribution | Fixed Point Attractor |

-----

## 🛠 Roadmap

*   [x] **v1.0:** Theoretical Definition (Concept Proof)
*   [x] **v2.2:** Waveform Simulation (Vitakka/Vicāra Implemented)
*   [x] **v2.3:** Gating & Meta-Cognition (Sati Implemented)
*   [x] **v2.4:** Anomaly Detection & Time Series Support
*   [x] **v3.0:** **Framework Refactoring** (Modularization, Builder, HF Trainer)
*   [x] **v3.1:** **Configuration Refactoring** (Type-Safe Configs, Factory Pattern)
*   [ ] **v3.2:** NLP Implementation (Text Summarization/Concept Extraction)
*   [ ] **Future:** Multi-Agent Samadhi (Dialogue of Insight)

-----

## 📜 License

MIT License
