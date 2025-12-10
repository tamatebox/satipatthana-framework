# Satipatthana Framework Specification

**Version:** 4.0 (The Three Engines & Guided Convergence)
**Status:** Active Specification

-----

## 1. Overview

### 1.1. Purpose

This document is the **implementation specification** for the Satipatthana Framework. It defines:

* System architecture and data flow
* Component interfaces and responsibilities
* Mathematical formulations (update rules, loss functions)
* Training curriculum
* Hyperparameters

For theoretical background and design rationale, see [theory_en.md](theory_en.md).

### 1.2. Target Audience

* **ML Engineers** implementing or extending the framework
* **Researchers** reproducing experiments
* **Code Reviewers** understanding system behavior

### 1.3. Prerequisites

* Basic understanding of deep learning (PyTorch)
* Familiarity with attention mechanisms
* Understanding of fixed-point iteration (helpful but not required)

-----

## 2. Terminology & Symbols

### 2.1. Key Symbols

| Symbol | Type | Description |
|:---|:---|:---|
| $X$ | Tensor (Batch, *) | Raw input data |
| $z$ | Tensor (Batch, $d$) | Adapted latent vector |
| $S_t$ | Tensor (Batch, $d$) | Latent state at iteration $t$ |
| $S^*$ | Tensor (Batch, $d$) | Converged fixed-point state |
| $S_0$ | Tensor (Batch, $d$) | Initial state from Vitakka |
| $P_k$ | Tensor ($d$,) | $k$-th concept probe vector |
| $V_{ctx}$ | Tensor (Batch, $c$) | Vipassana context vector |
| $\alpha$ | Tensor (Batch, 1) | Trust score (0.0–1.0) |
| $Y$ | Tensor (Batch, output_dim) | Final output |
| $\mathcal{T}$ | SantanaLog | Thinking trajectory $[S_0, S_1, \dots, S^*]$ |

### 2.2. Hyperparameter Symbols

| Symbol | Description | Typical Range |
|:---|:---|:---|
| $d$ | Latent space dimension | 64–256 |
| $c$ | Vipassana context dimension | 32–128 |
| $K$ | Number of concept probes | 8–32 |
| $T$ | Maximum Vicara steps | 6–20 |
| $\beta$ | State update inertia | 0.3–0.7 |
| $\epsilon$ | Convergence threshold (Sati) | 1e-4 |
| $\lambda_r$ | Reconstruction loss weight | 0.1–0.3 |
| $\lambda_g$ | Guidance loss weight | 0.1–0.5 |

-----

## 3. System Architecture

This framework consists of three main engines (Samatha, Vipassana, Decoder) and modular components that compose them.

### 3.1. Data Flow Overview

```txt
Raw Input (X)
    ↓
[SamathaEngine]
    Augmenter → Adapter → Vitakka → Vicara loop (w/ Sati) → S*, SantanaLog
    ↓
[VipassanaEngine]
    S* + SantanaLog → V_ctx, α (trust_score)
    ↓
[ConditionalDecoder]
    S* + V_ctx → Output (Y)
```

### 3.2. Class Diagram

![Class Diagram](diagrams/images/v4_class_diagram.png)

### 3.3. Engine 1: SamathaEngine

**Role:** World model. Converges any input to a "meaningful point".

**Input:** Raw Data `X` (Batch, *)
**Output:**

* `S*` (Batch, Dim): Converged latent state
* `SantanaLog`: Object recording the thinking trajectory
* `severity` (Batch,): Noise intensity (for Vipassana target)

**Component Structure:**

| Component | Role |
|:---|:---|
| **Adapter** | Projects and normalizes raw input to latent space |
| **Augmenter** | Applies noise/perturbation to input (during training) |
| **Vitakka** | Probe-based initial state $S_0$ generation |
| **Vicara** | Single-step state update ($S_t \rightarrow S_{t+1}$) |
| **Sati** | Convergence check and stopping control |

**Features:** Independent of tasks or labels, performs only "structure extraction". Internal perturbation control is possible via `drunk_mode` flag.

### 3.4. Engine 2: VipassanaEngine

**Role:** Meta-cognition. Monitors whether Samatha's thinking process (log) was sound.

**Input:** `S*` (Batch, Dim) + `SantanaLog`
**Output:**

* `V_ctx` (Batch, context_dim): Hint information for decoder (embedding of "doubt")
* `α` (Batch, 1): Trust score (0.0–1.0)

**Structure:** `StandardVipassana` (LogEncoder + ConfidenceMonitor)

### 3.5. Engine 3: ConditionalDecoder

**Role:** Expression. Integrates state and context into human-understandable form.

**Input:** `S*` (Batch, Dim) + `V_ctx` (Batch, context_dim) → Concatenate → (Batch, Dim + context_dim)
**Output:** `Y` (Batch, output_dim)

**Features:** Enables "humble expression"—when uncertain, output reflects that uncertainty (e.g., wider variance). **The only Decoder used during inference.**

### 3.6. Reconstruction Heads & AuxHead (Training Auxiliary)

Auxiliary modules for training stabilization. **Not used during inference.**

* **`adapter_recon_head`** (Stage 0): Reconstructs original input from Adapter output `z`
* **`samatha_recon_head`** (Stage 1): Reconstructs original input from converged point `S*`
* **`AuxHead`** (Stage 1): Auxiliary head for task prediction from `S*` (dimension: $d$)

#### Important: Relationship between AuxHead and ConditionalDecoder

| Module | Input Dimension | Purpose | Handling in Stage 3 |
|:---|:---|:---|:---|
| `AuxHead` | $d$ (`S*` only) | Stage 1 Guidance learning | **Discarded** |
| `ConditionalDecoder` | $d + c$ (`S*` ⊕ `V_ctx`) | Inference from Stage 3 onwards | Trained from scratch |

Stage 1's `AuxHead` and Stage 3's `ConditionalDecoder` are **physically separate modules due to different input dimensions**. `AuxHead` weights are not transferred to Stage 3; `ConditionalDecoder` is trained from scratch.

-----

## 4. Component Details

### 4.0. Component I/O Summary

| Component | Input | Output | Interface |
|:---|:---|:---|:---|
| **Adapter** | $X$ (Batch, *) | $z$ (Batch, $d$) | `BaseAdapter` |
| **Augmenter** | $X$ (Batch, *) | $(X_{aug}, severity)$ | `BaseAugmenter` |
| **Vitakka** | $z$ (Batch, $d$) | $(S_0, metadata)$ | `BaseVitakka` |
| **Vicara** | $S_t$ (Batch, $d$), context | $S_{t+1}$ (Batch, $d$) | `BaseVicara` |
| **Sati** | $S_t$, $\mathcal{T}$ | $(should\_stop, info)$ | `BaseSati` |
| **Vipassana** | $S^*$, $\mathcal{T}$ | $(V_{ctx}, \alpha)$ | `BaseVipassana` |
| **ConditionalDecoder** | $S^* \oplus V_{ctx}$ (Batch, $d+c$) | $Y$ (Batch, output\_dim) | `BaseDecoder` |

### 4.1. Adapter

**Function:** Projects and normalizes raw external input $X_{raw}$ to latent space.

* **Interface:** `BaseAdapter`
* **Implementations:** `MlpAdapter`, `CnnAdapter`, `LstmAdapter`, `TransformerAdapter`
* **Input:** Raw data $X$ (Batch, *)
* **Output:** Latent vector $z \in \mathbb{R}^d$

### 4.2. Augmenter

**Function:** Applies environmental noise or perturbation to input.

* **Interface:** `BaseAugmenter`
* **Implementations:** `IdentityAugmenter`, `GaussianNoiseAugmenter`
* **Input:** Raw data $X$ (Batch, *)
* **Output:** `(x_augmented, severity)` — severity is per-sample noise intensity $\in [0, 1]$

### 4.3. Vitakka

**Function:** Initial attractor search in latent space.

1. **Active Resonance:** Calculates resonance between concept probes $\mathbf{P}$ and input
2. **$S_0$ Generation:** Uses winner probe as Query to generate initial state

* **Interface:** `BaseVitakka`
* **Input:** Latent vector $z$ (Batch, $d$)
* **Output:** `(s0, metadata)` — metadata includes `winner_id`, `probs`, etc.

### 4.4. Vicara

**Function:** Single-step state update.

$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

* **Interface:** `BaseVicara`
* **Implementations:** `StandardVicara`, `WeightedVicara`, `ProbeSpecificVicara`
* **Input:** Current state $S_t$ (Batch, $d$), optional context from Vitakka
* **Output:** Next state $S_{t+1}$ (Batch, $d$)
* **Responsibility:** Single-step update only. Loop control is delegated to SamathaEngine.

**Variants:**

| Class | Description |
|:---|:---|
| `StandardVicara` | State update with single Refiner. Simplest |
| `WeightedVicara` | Weighted combination of multiple Refiners |
| `ProbeSpecificVicara` | Selects Refiner based on Vitakka's winner probe/probability |

### 4.5. Sati

**Function:** Convergence check and stopping control.

* **Interface:** `BaseSati`
* **Implementations:** `FixedStepSati`, `ThresholdSati`
* **Input:** Current state $S_t$ (Batch, $d$), trajectory $\mathcal{T}$
* **Output:** `(should_stop: bool, info: dict)`
* **Stop Condition:** Stops when state change energy $||S_{t+1} - S_t||$ falls below threshold $\epsilon$

### 4.6. Vipassana

**Function:** Meta-cognition module that monitors Samatha's thinking log and evaluates logical consistency and confidence.

* **Interface:** `BaseVipassana`
* **Implementation:** `StandardVipassana`
* **Input:** Converged state $S^*$ (Batch, $d$), trajectory $\mathcal{T}$
* **Output:** Context vector $V_{ctx}$ (Batch, $c$), trust score $\alpha$ (Batch, 1)
* **LogEncoder:** Compresses time-series log $\mathcal{T}$ into fixed-length vector
  * **Recommended Implementation:** Bi-LSTM or Transformer Encoder (1-2 layers). A time-series model is essential to capture "order" of thinking and "acceleration of convergence".
* **ConfidenceMonitor:** Detects "hesitation" or "contradiction", outputs trust score $\alpha$ and context vector $V_{ctx}$

**Fallback Strategy:** When $\alpha < \text{threshold}$ during inference:

* Output default answer ("I don't know")
* Or maximize output distribution variance
* Or trigger search/answer refusal

-----

## 5. Mathematical Formulation

### 5.1. Samatha Phase (Convergence)

**State update rule:**
$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

**Convergence guarantee:** The inertial update with $\beta \in (0, 1)$ reduces the effective Lipschitz constant of the mapping. If $\Phi$ has Lipschitz constant $L$, the combined mapping has effective constant $L_{eff} = (1 - \beta) + \beta L$. When $L < 1$ or when stability loss encourages contraction, convergence to a fixed point is promoted.

**Stop condition (Sati):**
$$\text{Stop if } ||S_{t+1} - S_t|| < \epsilon_{sati}$$

### 5.2. Vipassana Phase (Introspection)

Calculates trust from thinking log $\mathcal{T} = [S_0, \dots, S^*]$.

$$V_{ctx} = \text{Encoder}(\mathcal{T})$$
$$\alpha = \sigma(\text{Linear}(V_{ctx})) \in [0, 1]$$

* Target ($\hat{\alpha}$): Clean=1.0, Mismatch/Drunk=0.0

### 5.3. Loss Function (Stage-wise)

Objective function switches per training stage.

* **Stage 0 (Adapter Pre-training):** Reconstruction Only
    $$\mathcal{L}_0 = \mathcal{L}_{recon}(X, \hat{X}_{adapter})$$

* **Stage 1 (Samatha Training):** Stability + Reconstruction + (Optional) Label Guidance
    $$\mathcal{L}_1 = ||S_T - S_{T-1}||^2 + \lambda_r \mathcal{L}_{recon} + \lambda_g \mathcal{L}_{task}(y, \text{AuxHead}(S^*))$$

* **Stage 2 (Vipassana Training):** Binary Cross Entropy (Contrastive)
    $$\mathcal{L}_2 = \text{BCE}(\alpha, \hat{\alpha})$$

* **Stage 3 (Decoder Fine-tuning):** Task Specific Loss
    $$\mathcal{L}_3 = \mathcal{L}_{task}(y, \text{Decoder}(S^*, V_{ctx}))$$

-----

## 6. Data Structures

### 6.1. SantanaLog

Object that records state history during the convergence process.

```python
class SantanaLog:
    def add(self, state: Tensor) -> None:
        """Add state to trajectory"""

    def to_tensor(self) -> Tensor:
        """Convert trajectory to tensor (Steps, Batch, Dim)"""

    def __len__(self) -> int:
        """Number of recorded steps"""
```

### 6.2. SystemOutput

```python
@dataclass
class SystemOutput:
    output: Tensor        # Decoded result
    s_star: Tensor        # Converged latent state
    v_ctx: Tensor         # Vipassana context vector
    trust_score: Tensor   # Trust score (0.0–1.0)
    santana: SantanaLog   # Thinking trajectory
    severity: Tensor      # Noise intensity
```

-----

## 7. Algorithm Flow

### 7.1. Inference Sequence Diagram

![Inference Sequence Diagram](diagrams/images/v4_sequence_diagram_inference.png)

### 7.2. Inference Flow

```python
def inference(x: Tensor) -> SystemOutput:
    # Phase 1: Samatha (Convergence)
    s_star, santana, severity = samatha_engine(x, run_augmenter=False)

    # Phase 2: Vipassana (Introspection)
    v_ctx, trust_score = vipassana_engine(s_star, santana)

    # Phase 3: Decode (Expression)
    output = conditional_decoder(concat(s_star, v_ctx))

    return SystemOutput(output, s_star, v_ctx, trust_score, santana, severity)
```

### 7.3. SamathaEngine Internal Flow

```python
def samatha_forward(x, noise_level=0.0, run_augmenter=True):
    # Augment (training only)
    if run_augmenter:
        x_aug, severity = augmenter(x, noise_level)
    else:
        x_aug, severity = x, zeros(batch_size)

    # Adapt
    z = adapter(x_aug)

    # Vitakka: Initial state generation
    s0, metadata = vitakka(z)

    # Vicara loop with Sati
    santana = SantanaLog()
    s_t = s0
    santana.add(s_t)

    for step in range(max_steps):
        s_t = vicara(s_t, context=metadata)
        santana.add(s_t)

        should_stop, _ = sati(s_t, santana)
        if should_stop:
            break

    return s_t, santana, severity
```

-----

## 8. Training Curriculum (4-Stage)

### 8.1. Training Policy

| Stage | Name | Trainable | Frozen | Objective |
|:---|:---|:---|:---|:---|
| **0** | Adapter Pre-training | Adapter, adapter_recon_head | All others | Reconstruction Loss |
| **1** | Samatha Training | Adapter, Vitakka, Vicara, Sati, (samatha_recon_head, AuxHead) | Vipassana, TaskDecoder | Stability + Recon + (Guidance) |
| **2** | Vipassana Training | Vipassana | All others | BCE (Contrastive) |
| **3** | Decoder Fine-tuning | TaskDecoder | All others | Task Specific Loss |

### 8.2. Iteration Strategy

| Mode | Description | Use Case |
|:---|:---|:---|
| **Fixed Steps** | Always run exactly $T$ iterations | Training (gradient stability) |
| **Early Stopping** | Stop when $\|S_{t+1} - S_t\| < \epsilon$ | Inference (efficiency) |
| **Hybrid** | Run minimum steps, then allow early stop | Balance stability and efficiency |

**Recommended:**

* **Training:** Fixed steps ($T = 10$) for stable gradient flow
* **Inference:** Early stopping with $\epsilon = 10^{-4}$ for efficiency
* **Transition:** Use `SatiConfig.mode` to switch between strategies

### 8.3. Stage Transition Criteria

| Transition | Criterion | Fallback |
|:---|:---|:---|
| 0 → 1 | Reconstruction loss plateaus | Fixed epochs (e.g., 5) |
| 1 → 2 | Stability loss $< 10^{-3}$ | Fixed epochs (e.g., 10) |
| 2 → 3 | Vipassana BCE $< 0.3$ | Fixed epochs (e.g., 5) |

**Early Stopping:** Monitor validation loss per stage. If no improvement for `patience` epochs, transition to next stage.

### 8.4. Stage 2 Noise Generation Strategy

Three data generation strategies to teach Vipassana meta-cognition:

1. **Environmental Ambiguity (Augmented Path)**
   * Add noise to input data
   * Target: `1.0 - severity`

2. **Internal Dysfunction (Drunk Path)**
   * Perturb SamathaEngine internals (`drunk_mode=True`)
   * Specific implementations: Increase Dropout rate in Vicara, add temporary noise to Refiner weights, disturb Vitakka's temperature parameter, etc.
   * Target: `0.0`

3. **Logical Inconsistency (Mismatch Path)**
   * Shuffle S* and SantanaLog within batch
   * Target: `0.0`

**Batch Composition (recommended):**

| Path | Proportion | Purpose |
|:---|:---|:---|
| Clean | 25% | Baseline trust |
| Augmented | 25% | Environmental uncertainty |
| Drunk | 25% | Internal dysfunction detection |
| Mismatch | 25% | Logical inconsistency detection |

-----

## 9. Hyperparameters

### 9.1. Model Architecture

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `latent_dim` | $d$ | 64-256 | Latent space dimension |
| `context_dim` | $c$ | 32-128 | Vipassana output dimension |
| `num_probes` | $K$ | 8-32 | Number of Vitakka probes |
| `max_steps` | $T$ | 6-20 | Maximum Vicara steps |

### 9.2. Training Strategy

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `sati_threshold` | $\epsilon$ | 1e-4 | Convergence threshold |
| `beta` | $\beta$ | 0.3-0.7 | State update inertia parameter |
| `guidance_weight` | $\lambda_g$ | 0.1-0.5 | (Stage 1) Guidance loss strength |
| `recon_weight` | $\lambda_r$ | 0.1-0.3 | Reconstruction loss weight |

-----

## 10. Applications & Training Strategies

For supervised tasks, actively use **Stage 1 Guidance (AuxHead)** to optimize Samatha's convergence space for the task.

| Application Task | Stage 1 Strategy | Stage 2 Role | Stage 3 Decoder |
|:---|:---|:---|:---|
| **Supervised Classification** | Guidance (CE Loss) | Hallucination Check | Classifier (Softmax) |
| **Supervised Regression** | Guidance (MSE Loss) | Uncertainty Est. | Regressor (Linear) |
| **Anomaly Detection** | Reconstruction Only | Anomaly Score (final output) | Identity |
| **Structure Discovery** | Stability Only | Boundary Detection | None |

-----

## 11. Architectural Extensibility

Components can be freely combined using `SystemConfig` and various `ComponentConfig`.

### 11.1. Task-Specific Customization Example

| Task | Adapter | Augmenter | Vicara | Decoder |
|:---|:---|:---|:---|:---|
| **Time Series Anomaly Detection** | LSTM | Gaussian | Standard | Reconstruction |
| **Image Classification** | CNN | Identity | Standard | Conditional |
| **Dialogue Intent Estimation** | Transformer | Identity | ProbeSpecific | Conditional |
| **Robot Control** | MLP | Gaussian | Weighted | Conditional |

### 11.2. Config Example

```python
from satipatthana.configs import SystemConfig, SamathaConfig, VipassanaEngineConfig
from satipatthana.configs import create_adapter_config, create_vicara_config

config = SystemConfig(
    samatha=SamathaConfig(
        adapter=create_adapter_config("mlp", input_dim=784, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.3),
        vitakka=VitakkaConfig(num_probes=16),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32),
    ),
    use_label_guidance=True,
)
```
