# Samadhi Framework (Deep Convergence Architecture) Specification

**Version:** 3.0 (Framework Modularization)
**Status:** Active Specification

-----

## 1. Concept Definition

The **Samadhi Framework** is a **recursive attention architecture** designed to **extract essential structures (State Refinement)** and **stabilize internal states (Convergence)** from chaotic information streams. It adopts an approach of convergent order formation rather than divergent information expansion.

  * **Core Philosophy:** Focuses on vertical deepening (Convergence/Insight) rather than horizontal expansion (Divergence/Generation).
  * **Output:** A single invariant state vector with minimized entropy (Latent Point Attractor).
  * **Operational Mode:** Dynamic transition from an Open System to a Closed System.

-----

## 2. System Architecture

This framework is built by composing modular components: Adapter, Vitakka, Vicara, and Decoder.

### Adapter (Manasikāra - Input Adaptation)

**Function:** Projects and normalizes raw external inputs $X_{raw}$ from various modalities (image, time series, text, etc.) into the model-specific latent space (Samadhi Space).

*   **Role:** Transforms external signals into a "semantic format" that the model can process (Attention/Manasikāra).
*   **Interface:** `BaseAdapter` (`samadhi/components/adapters/base.py`)
*   **Implementations:**
    *   *MlpAdapter:* For tabular data or flat vectors.
    *   *CnnAdapter:* For image data (Conv2d).
    *   *LstmAdapter:* For time series data (LSTM).
    *   *TransformerAdapter:* For sequence data (Transformer Encoder).
*   **Output:** Latent vector $X_{adapted} \in \mathbb{R}^d$.

### Vitakka (Search & Orientation)

**Function:** Discovers and orients towards an initial attractor (seed) worth converging to from the chaotic input stream ($X_{adapted}$).

*   **Interface:** `BaseVitakka` (`samadhi/components/vitakka/base.py`)
1.  **Concept Probes ($\mathbf{P}$):**
      * The system holds $K$ "Concept Probes (Basis Vectors)".
2.  **Active Resonance:**
      * Calculates resonance (dot product) between input $X$ and probes $\mathbf{P}$.
      * **Lateral Inhibition:** Sets Softmax temperature $\tau$ low to highlight the strongest probe (Winner).
3.  **Confidence Gating (Anti-Hallucination):**
      * If maximum resonance is below threshold $\theta_{gate}$, input is considered "noise (distraction)" and processing is blocked (Gate Closed).
4.  **$S_0$ Slice Generation:**
      * Uses the winner probe $p_{win}$ as a Query to slice the input $X$ via Attention, generating initial state $S_0$.

### Vicāra (Recurrent Refinement)

**Function:** Blocks external input and recursively purifies the internal state.

*   **Interface:** `BaseVicara` (`samadhi/components/vicara/base.py`)
*   **Implementations:**
    *   *StandardVicara:* Shares a single general Refiner ($\Phi$).
    *   *WeightedVicara:* Uses a weighted sum of multiple Refiners.
    *   *ProbeSpecificVicara:* Has a dedicated Refiner ($\Phi_k$) for each concept probe $p_k$.

1.  **Isolation:** At $t > 0$, the gate to external input $X$ is closed, allowing only self-loops.
2.  **Refinement Loop:**
      * **Hard Attention Mode (Inference):** Applies only the Refiner $\Phi_{win}$ corresponding to the winner probe.
          * $S_{t+1} = \Phi_{win}(S_t)$
      * **Soft Attention Mode (Training):** Updates using a weighted sum based on probability distribution of all probes (for gradient propagation).
          * $S_{t+1} = \sum_k w_k \Phi_k(S_t)$
3.  **Convergence Check:**
      * When state change $||S_{t+1} - S_t||$ falls below $\epsilon$, it is considered "Appanā (Absorption)" and inference ends.

### Refiner (Internal Dynamics)

**Function:** An execution unit inside Vicāra that defines state transitions (dynamics) in the latent space.

*   **Interface:** `BaseRefiner` (`samadhi/components/refiners/base.py`)
*   **Implementations:**
    *   *MlpRefiner:* Simple state update using Fully Connected layers and activation functions.
    *   *GruRefiner:* (Future) State update with memory using GRU cells.
    *   *AttentionRefiner:* (Future) Organizing relationships between states using Self-Attention.

### Sati-Sampajañña (Meta-Cognition & Logging)

**Function:** Records system behavior as a "causal narrative" to ensure Explainability (XAI).

1.  **Probe Log (Momentary Awareness):** What was selected in that step.
2.  **Cetanā Dynamics (Flow of Time):** Tracking transition of intent (Sustain, Shift, Scatter) by comparing with the previous step.

### Decoder (Expression - Output Reconstruction)

**Function:** Restores or converts the converged and purified latent state $S_{final}$ back to the original input format or target format.

*   **Role:** Returning internal insight to external expression.
*   **Interface:** `BaseDecoder` (`samadhi/components/decoders/base.py`)
*   **Implementations:**
    *   *ReconstructionDecoder:* Returns $S_{final}$ to original input dimensions (for Autoencoder).
    *   *CnnDecoder:* For image reconstruction.
    *   *LstmDecoder / SimpleSequenceDecoder:* For sequence reconstruction.
*   **Output:** Reconstructed data $\hat{X}$ or predicted value $Y$.

-----

## 3. Mathematical Formulation

### 3.1. Vitakka Phase (Resonance)

Resonance score $R_k$ of probe $p_k$ for input $X \in \mathbb{R}^{L \times d}$:

$Score_k = || \frac{1}{\sqrt{d}} \sum_{i=1}^{L} \text{Softmax}(p_k^T x_i) \cdot x_i ||$

Winner determination and probability distribution (with lateral inhibition):
$\hat{w} = \text{Softmax}\left( \frac{[Score_1, \dots, Score_K]}{\tau} \right)$

### 3.2. Initialization ($S_0$)

Initial state determination by gate $G \in \{0, 1\ gimana
$S_0 = G \cdot \text{Attention}(Q=p_{win}, K=X, V=X)$
Where $G = 1 \text{ if } \max(Score) > \theta_{gate} \text{ else } 0$.

### 3.3. Vicāra Phase (State Transition)

Update rule as a first-order Markov process:
$S_{t+1} = (1 - \beta) S_t + \beta \Phi_k(S_t)$

  * $\beta$: Update rate (inertia term). Prevents sudden changes and ensures stable trajectory.
  * $\Phi_k$: Mapping function specific to selected concept $k$ (in Probe-Specific case).
  * $\lim_{t \to \infty} || S_{t+1} - S_t || = 0$ (Convergence to fixed point)

### 3.4. Loss Function (Stability Loss)

The objective function during training depends on the selected `Objective`. The basic form is:
$\mathcal{L} = \underbrace{|| S_{T} - S_{T-1} ||^2}_{Stability} + \lambda_1 \underbrace{\sum |S_T|}_{Sparsity} - \lambda_2 \underbrace{I(S_T; S_0)}_{Info Retention}$

-----

## 4. Data Structures

### 4.1. Probe Log (Snapshot)

Metadata for each inference step.

```json
{
  "timestamp": 12345678,
  "intention": {
    "winner_id": 3,
    "winner_label": "Logical_Causality",
    "confidence": 0.94,
    "gate_status": "OPEN",
    "entropy": 0.05
  },
  "raw_scores": [0.01, 0.02, 0.01, 0.94, 0.02]
}
```

### 4.2. Cetanā Dynamics Log (Transition)

Log describing the temporal flow of intention.

```json
{
  "step": 5,
  "transition": {
    "from": "Breath_Rhythm",
    "to": "Body_Sensation",
    "type": "Shift",
    // Types: "Sustain", "Shift", "Distracted", "Deepening"
    "attention_shift_magnitude": 0.45,
    "smoothness": 0.8
  }
}
```

-----

## 5. Algorithm Flow

1.  **Input:** Acquire data $X$.
2.  **SamadhiEngine.forward(x, run_vitakka=True, run_vicara=True):
    *   **Adapter:** $z = \text{Adapter}(x)$
    *   **Vitakka (Optional):** $s_0, \text{meta} = \text{Vitakka}(z)$
        *   Gate Decision (Threshold Check)
    *   **Vicāra (Optional):**
        *   Loop $t=1 \dots N$:
            *   $S_{next} = \Phi(S_{curr})$
            *   Update State with Inertia
    *   **Decoder:** $\text{Output} = \text{Decoder}(S_{final})$
3.  **Output:** Converged $S_{final}$, decoder output, and `Logs`.

-----

## 6. Recommended Hyperparameters

Classification of keys used in `config` dictionary and recommended values.

### Model Architecture
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`dim`** | $d$ | 64 - 512 | Dimension of latent state vector. |
| **`input_dim`** | $D_{input}$ | - | Dimension of input data. |
| **`seq_len`** | $L$ | 10 - 60 | *(Time Series Model Only)* Sequence length. |
| **`n_probes`** | $K$ | 16 - 64 | Number of concept probes. |
| **`vicara_type`** | - | `"probe_specific"` | `"standard"` (Shared) or `"probe_specific"` (Individual). |
| **`probe_trainable`** | - | `True` | Whether to train probes themselves. |
| **`adapter_hidden_dim`** | $D_{hidden}$ | 256 | Hidden layer dimension in Adapter. |

### Vitakka (Search)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`gate_threshold`** | $\theta$ | 0.3 - 0.5 | Strength to reject delusion (noise). |
| **`softmax_temp`** | $\tau$ | 0.1 - 0.2 | Lower values select "One-pointedness (Single Theme)". |

### Vicara (Refinement)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`refine_steps`** | $T_{max}$ | 5 - 10 | Number of recursive refinement steps. |
| **`inertia`** | $\beta$ | 0.7 | Inertia of state update. |

### Training (Objective Params)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`stability_coeff`** | $\lambda_{stab}$ | 0.01 | Strength promoting state convergence. |
| **`entropy_coeff`** | $\lambda_{ent}$ | 0.1 | Strength penalizing ambiguous search results. |
| **`balance_coeff`** | $\lambda_{bal}$ | 0.001 | Equalizes frequency of probe usage. |
| **`anomaly_margin`** | `5.0` | - | *(AnomalyObjective)* Margin for anomalous data. |
| **`anomaly_weight`** | `1.0` | - | *(AnomalyObjective)* Penalty weight for anomalous data. |

-----

## 7. Core Dynamics: Divergence vs. Convergence

Contrasting the core dynamics of the Samadhi Framework, which is based on convergence, with traditional generative models that take a divergent approach.

| Feature | Divergent Models | **Convergent Models** |
| :--- | :--- | :--- |
| **Basic Operation** | Sequence Prediction, Generation, Divergence | State Purification, Stabilization, Convergence |
| **Time Dependency** | Dependent on Context History | Dependent only on Current State (Markovian) |
| **Attention** | Self-Attention (Between Elements) | Recursive Attention (Between State-Probe) |
| **Nature of Inference** | **Open/Infinite**<br>Can continue indefinitely | **Closed/Finite**<br>Settles to a point |
| **Explainability** | Limited (Attention Map, etc.) | **Extremely High (Probe/Cetanā Log)** |
| **Philosophical Basis** | Association, Generation, Expansion | **Meditation (Samadhi), Insight, Essence Extraction** |

-----

## 8. Applications & Training Strategies

The Samadhi Framework can be applied to different tasks by combining **Training Strategies (Trainer + Objective)** and **Decoders**.

| Application Task | Objective | Decoder Role | Loss Function |
| :--- | :--- | :--- | :--- |
| **Structure Discovery / Clustering**<br>(Unsupervised) | `UnsupervisedObjective` | **Identity** | Stability + Entropy + Sparsity<br>(Pursuing only internal state stabilization) |
| **Autoencoder Pre-training**<br>(Pre-training) | `AutoencoderObjective` | **Reconstruction** | Reconstruction Loss Only<br>(Minimize reconstruction error, skip Vicara) |
| **Anomaly Detection** | `AnomalyObjective` | **Reconstruction** | Recon + Stability + Margin<br>(Reconstruct normal data, reject anomalous data) |
| **Supervised Task**<br>(Classification) | `SupervisedClassificationObjective` | **Classifier** | CrossEntropy + Stability<br>(Target prediction) |
| **Supervised Task**<br>(Regression) | `SupervisedRegressionObjective` | **Regressor** | MSE + Stability<br>(Target prediction) |
| **Supervised Task**<br>(Robust Regression) | `RobustRegressionObjective` | **Regressor** | Huber / L1 + Stability<br>(Target prediction robust to outliers) |
| **Semantic Similarity Learning**<br>(Unsupervised) | `CosineSimilarityObjective` | **Identity** / **Reconstruction** | Cosine Embedding Loss + Stability<br>(Align directionality of input and reconstruction) |

*   **Meditation Mode (Unsupervised):** Discover inherent structures (Dharma) in data without relying on external labels.
*   **Expression Mode (Supervised/Anomaly):** Solve external tasks (classification, detection) using the discovered structures.

-----

## 9. Integration with Large Language Models (LLMs)

Samadhi's "Convergence/Stabilization" and LLM's "Generation/Divergence" are complementary.

*   **LLM (Generator):** Responsible for divergent thinking, token prediction, and context generation.
*   **Samadhi (Stabilizer):** Responsible for convergent thinking, state purification, and fixing intention.

Key Integrations:
1.  **Intent Stabilization:** Purify LLM output with Samadhi to achieve consistent dialogue without drift.
2.  **Prompt Refinement:** Purify user input into a clear instruction vector for the LLM.
3.  **Output Verification:** Detect LLM hallucinations using the Convergence Score (Stability Score).

-----

## 10. Architectural Extensibility

Components can be freely combined using `SamadhiBuilder` or `presets`.

### 10.1. Task-Specific Customization Example

| Task | Adapter | Refiner | Decoder | Objective |
| :--- | :--- | :--- | :--- | :--- |
| **Time Series Anomaly Detection** | LSTM | MLP | Reconstruction | AnomalyObjective |
| **Image Classification** | CNN | MLP | Classification | SupervisedObjective |
| **Dialogue Intent Estimation** | Transformer | Attention | Classification | SupervisedObjective |
| **Robot Control** | Sensor Fusion | MLP | Action | RL (PPO) |

```
```