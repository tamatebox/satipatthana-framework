# Samadhi Model (Deep Convergence Architecture)

> **"From Generation to Convergence."**

![Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Samadhi Model** is a novel **recursive attention architecture** designed not for traditional "sequence prediction (Next Token Prediction)" seen in generative AI, but for "extracting the essential structure" and "stabilizing the internal state" of the subject.

It engineeringly implements a vertical deepening of information (quiet insight) rather than horizontal expansion (talkative generation).

---

## 🧘 Concept & Philosophy

Modern LLMs (Transformers) have a "divergent" nature, generating tokens one after another by riding the waves of probability distributions. In contrast, the **Samadhi Model** is a "convergent" model that transitions its state towards a dynamical system's attractor (fixed point).

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
├── data/               # MNIST, Waveform datasets
├── docs/               # Theoretical specifications
├── src/
│   ├── components/     # Vitakka (Search) and Vicara (Refinement) modules
│   ├── model/          # Core Architectures (SamadhiCore, ConvSamadhi)
│   └── train/          # Trainer Implementations (Base, Supervised, Unsupervised)
├── test/               # Demos and Training Examples
│   ├── test_trainer_cifar10.py
│   └── test_trainer_mnist.py
├── main.py             # Entry point
└── pyproject.toml      # Project configuration (uv)
````

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
from src.model import SamadhiCore, CONFIG
import torch

# Initialize Model
CONFIG["dim"] = 64
model = SamadhiCore(CONFIG)

# Input: Noise mixed with a target signal
noisy_input = torch.randn(1, 64)

# Execute One Step of Meditation
s_final, log = model.forward_step(noisy_input, step_idx=0)

if log["probe_log"]["gate_open"]:
    print(f"Focused on: {log['probe_log']['winner_label']}")
    print(f"Converged Energy: {log['energies'][-1]}")
else:
    print("[--- SILENCE ---] Distraction detected.")
```

### 2. Run Demos

**Supervised Training Loop (MNIST Denoising)**
This demo shows supervised learning using the MNIST dataset. It learns the process of purifying noisy images.

```bash
uv run test/test_trainer_mnist.py
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
*   [ ] **v3.0:** NLP Implementation (Text Summarization/Concept Extraction)
*   [ ] **Future:** Multi-Agent Samadhi (Dialogue of Insight)

-----

## 📜 License

MIT License