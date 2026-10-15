# Satipatthana Framework (GEMINI Context)

## Project Overview

**Satipatthana** is an introspective recursive attention architecture (Deep Learning Framework) designed for converging to essential structures ("Samatha") from noisy data and introspecting its own reasoning process ("Vipassana").

It differentiates itself from traditional generative models by focusing on **convergence** (reducing entropy to reach a fixed point) rather than expansion (autoregressive generation). It provides built-in **uncertainty quantification** (Trust Score) and **explainability** (Thinking Trajectory).

**Core Philosophy:** "From Chaos to Essence, with Awareness."

### Key Technologies

* **Language:** Python 3.12+
* **Framework:** PyTorch 2.9+
* **Package Manager:** `uv`
* **Dependencies:** Transformers, Scikit-learn, NumPy, Pandas, Matplotlib.
* **Architecture Terminology:** Buddhist psychology concepts (Vitakka, Vicara, Sati, Vipassana) mapped to engineering modules.

## Architecture & Components

The system is composed of three main engines and modular components.

### 1. SamathaEngine (Convergence)

The "World Model" that converges input to a stable fixed point ($S^*$).

* **Adapters:** Projects input to latent space (`MlpAdapter`, `CnnAdapter`, etc.).
* **Augmenters:** Injects noise for robustness (`GaussianAugmenter`).
* **Vitakka (Initial Application):** Probes for semantic initialization ($S_0$).
* **Vicara (Sustained Application):** Refinement loop ($S_t \rightarrow S_{t+1}$).
* **Sati (Mindfulness):** Monitors convergence and determines when to stop.

### 2. VipassanaEngine (Introspection)

The "Meta-Cognition" module that monitors the thinking process.

* **Vipassana:** Encodes the thinking trajectory (`SantanaLog`) to produce a Context Vector ($V_{ctx}$) and a Trust Score ($\alpha$).

### 3. ConditionalDecoder (Expression)

The output module.

* **Decoder:** Generates final output ($Y$) from the converged state ($S^*$) and context ($V_{ctx}$).

## Directory Structure

* **`satipatthana/`**: Main source code.
  * `configs/`: Configuration system (DataClasses).
  * `components/`: Implementation of core modules (adapters, decoders, vicara, etc.).
  * `core/`: Engine implementations (`System`, `Engines`).
  * `train/`: Training logic (`SatipatthanaTrainer`).
  * `utils/`: Logging and helper functions.
* **`docs/`**: Detailed specifications and theory (essential reading for logic changes).
* **`notebooks/`**: Usage examples and demos.
* **`tests/`**: Unit tests (`pytest`).

## Development Conventions

* **Package Management:** Use `uv` for all dependency management.
  * Install/Sync: `uv sync`
  * Add dependency: `uv add <package>`
* **Building/Running:**
  * Run scripts: `uv run python main.py` (Currently a placeholder, see `README.md` for example code) or `uv run jupyter lab`.
  * Training: Via `SatipatthanaTrainer` (see `README.md` examples).
* **Testing:**
  * Run tests: `uv run pytest`
* **Formatting:** `black`
  * Line length: 119
* **Configuration:**
  * Uses Python `dataclasses`.
  * Located in `satipatthana/configs/`.
  * Use factory methods for creating configs (e.g., `create_adapter_config`).

## Training Curriculum (4-Stage)

The model requires a specific training order:

1. **Stage 0:** Adapter Pre-training (Reconstruction).
2. **Stage 1:** Samatha Training (Stability + Convergence).
3. **Stage 2:** Vipassana Training (Introspection/Trust Score).
4. **Stage 3:** Decoder Fine-tuning (Task performance).

## Quick Reference

* **Input:** $X$ (Batch, *)
* **Latent State:** $S_t$ (Batch, $d$)
* **Trust Score:** $\alpha \in [0, 1]$
* **Run Minimal Example:** `uv run python main.py` (if `main.py` contains the example).
