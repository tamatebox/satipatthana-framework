# Samadhi Model (Deep Convergence Architecture)

> **"From Generation to Convergence."**

![Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Samadhi Model** is a novel **recursive attention architecture** designed not for traditional "sequence prediction (Next Token Prediction)" seen in generative AI, but for "extracting the essential structure" and "stabilizing the internal state" of the subject.

It engineeringly implements a vertical deepening of information (quiet insight) rather than horizontal expansion (talkative generation).

---

## 🧘 Concept & Philosophy

Modern LLMs (Transformers) have a "divergent" nature, generating tokens one after another by riding the waves of probability distributions. In contrast, the **Samadhi Model** is a "convergent" model that transitions its state towards a dynamical system\'s attractor (fixed point).

It implements the process of meditative concentration (Samadhi) in Buddhist psychology as the following engineering modules:

| Module | Buddhist Term | Engineering Concept | Function |
| :--- | :--- | :--- | :--- |
| **Vitakka** | 尋 (Initial Application) | **Active Probing** | Searches and captures "intentions (Probes)" from chaotic input. |
| **Sati** | 正知 (Clear Comprehension) | **Gating Mechanism** | Detects noise and hallucinations, blocking further processing. |
| **Vicāra** | 伺 (Sustained Application) | **Recurrent Refinement** | Blocks external input and minimizes state energy (purifies) through a recursive loop. |
| **Santāna** | 相続 (Continuity) | **State Dynamics Log** | Tracks the temporal transitions of intentions (concentration, shift, dispersion). |

---

## 🚀 Key Features

*   **Convergence:** The output is not a text stream, but a single "Purified State" with minimized entropy.
*   **O(1) Inference:** Inference cost does not depend on the input length (Context Length), but only on the number of convergence steps (a constant).
*   **Noise Robustness:** The powerful Gating mechanism returns "silence" for meaningless inputs (distractions) without allocating computational resources.
*   **Explainability (XAI):** "Why a particular subject was focused on" and "how concentration deepened" are fully visualized as logs.

---

## 📂 Project Structure

```bash
.
├── data/               # MNIST, Sensor, Credit Card datasets
├── docs/               # Theoretical specifications
├── notebooks/          # Experiments and Analysis
│   ├── fraud_detection.ipynb             # Tabular Anomaly Detection (Credit Card)
│   ├── time_series_anomaly_detection.ipynb # Time Series Anomaly Detection (Sensor)
│   └── mnist.ipynb                         # Visual Samadhi Demo
├── src/
│   ├── components/     # Vitakka (Search) and Vicara (Refinement) modules
│   ├── model/          # Core Architectures
│   │   ├── samadhi.py              # Base Class
│   │   ├── conv_samadhi.py         # CNN-based (Image)
│   │   ├── mlp_samadhi.py          # MLP-based (Tabular)
│   │   ├── lstm_samadhi.py         # LSTM-based (Time Series)
│   │   └── transformer_samadhi.py  # Transformer-based (Time Series)
│   └── train/          # Trainer Implementations
│       ├── base_trainer.py
│       ├── supervised_trainer.py
│       ├── unsupervised_trainer.py
│       └── anomaly_trainer.py      # Contrastive Margin Loss Trainer
├── tests/              # Unit Tests (Contains unit and integration test scripts)
├── main.py             # Entry point
└── pyproject.toml      # Project configuration (uv)
```

-----

## ⚡ Quick Start

### Prerequisites

This project uses `uv` as its package manager.

```bash
# Install dependencies
uv sync
```

### 1. Basic Usage (Signal Purification)

This is a minimal demo for extracting specific signals (intentions) from noisy waveforms.

```python
from src.model.samadhi import SamadhiModel
from src.model.mlp_samadhi import MlpSamadhiModel # Example of a specific implementation
import torch

# Configuration would be specific to the model type
config = {"input_dim": 10, "dim": 64, "n_probes": 5, "refine_steps": 5, "gate_threshold": 0.5}
model = MlpSamadhiModel(config)

# Example usage:
input_data = torch.randn(1, config["input_dim"]) # Batch size 1
purified_state, _ = model.forward_step(input_data, step_idx=0)
print(f"Purified state shape: {purified_state.shape}")
```

### 2. Run Demos (Jupyter Notebooks)

For various experiments including visual demos, supervised/unsupervised training, and anomaly detection tasks (Credit Card Fraud, Sensor Failures), use the Jupyter Notebooks provided in `notebooks/`.

```bash
# Start Jupyter
uv run jupyter notebook
# Open notebooks such as:
# - 'notebooks/mnist.ipynb' (Visual Samadhi Demo)
# - 'notebooks/fraud_detection.ipynb' (Tabular Anomaly Detection)
# - 'notebooks/time_series_anomaly_detection.ipynb' (Time Series Anomaly Detection)
```

-----

## 📊 Architecture Comparison

| Feature | Transformer (GPT) | Samadhi Model (Ours) |
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
*   [x] **v2.4:** Anomaly Detection & Time Series Support (LSTM/Transformer)
*   [ ] **v3.0:** NLP Implementation (Text Summarization/Concept Extraction)
*   [ ] **Future:** Multi-Agent Samadhi (Dialogue of Insight)

-----

## 📜 License

MIT License