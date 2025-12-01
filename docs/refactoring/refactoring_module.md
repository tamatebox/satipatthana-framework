# Samadhi Framework Refactoring Plan (v3.0)

**Status:** Completed
**Goal:** Transition from a monolithic "Model" to a flexible "Framework".

-----

## 1. Vision: From Model to Framework

Current implementation treats **Samadhi** as a specific model architecture with subclasses (`MlpSamadhiModel`, `LstmSamadhiModel`). However, Samadhi's core value lies in its **"Convergence Process" (Vitakka -> Sati -> Vicāra)**.

By redefining Samadhi as a **Meta-Framework** that orchestrates pluggable components, we can apply this convergence philosophy to any domain (Time Series, Vision, NLP, RL) without creating an explosion of subclasses.

**Target Concept:**
> **Samadhi is not a model. It is a dynamical system engine that purifies information.**

-----

## 2. Current Architecture & Issues

### Current Structure (Inheritance-based)
```python
class SamadhiModel(nn.Module):
    def __init__(self, config):
        # Hard-coded initialization of components
        self.vitakka = Vitakka(config)
        self.vicara = self._build_vicara(config)
        self.decoder = self._build_decoder() # Usually Identity

class MlpSamadhiModel(SamadhiModel):
    def _build_adapter(self): ... # Override
    def _build_decoder(self): ... # Override
```

### Issues
1.  **Combinatorial Explosion:** To create a "Time Series Classification Model", we need a class combining `LSTM Adapter` + `Classification Decoder`. If we want `GRU Refiner`, we need another subclass.
2.  **Tight Coupling:** `SamadhiModel` knows too much about how to build its components (`_build_vicara`, etc.).
3.  **Limited Extensibility:** Users cannot easily inject custom Adapters or Decoders without modifying the core code or creating deep inheritance hierarchies.

-----

## 3. Proposed Architecture (Composition-based)

Adopting **Dependency Injection** and **Builder Pattern**.

### 3.1. Core Container (`SamadhiEngine`)

A lightweight orchestrator that holds components and defines the data flow. It knows *how to run* the convergence loop, but not *what* the components are.

```python
class SamadhiEngine(nn.Module):
    def __init__(
        self,
        adapter: BaseAdapter,
        vitakka: BaseVitakka, # Expected to be an instance of a BaseVitakka subclass (e.g., StandardVitakka)
        vicara: BaseVicara,   # Expected to be an instance of a BaseVicara subclass (e.g., StandardVicara)
        decoder: BaseDecoder
    ):
        super().__init__()
        self.adapter = adapter
        self.vitakka = vitakka
        self.vicara = vicara
        self.decoder = decoder

    def forward(self, x):
        z = self.adapter(x)           # Raw -> Samadhi Space
        s0, meta = self.vitakka(z)    # Search & Gate
        s_final, _ = self.vicara(s0)  # Purify
        output = self.decoder(s_final) # Samadhi Space -> Target Space
        return output, s_final, meta
```

### 3.2. Component Interfaces

Standardize inputs/outputs for pluggability.

*   **`BaseAdapter`**: `Raw Input -> Tensor(Batch, Dim)`
    *   Path: `samadhi/components/adapters/base.py`
    *   Implementations: `MlpAdapter` (Implemented), `CnnAdapter` (**Implemented**), `LstmAdapter` (**Implemented**), `TransformerAdapter` (**Implemented**)
*   **`BaseVitakka`** (Prober): `Tensor(Batch, Dim) -> (s0, Metadata)`
    *   Path: `samadhi/components/vitakka/base.py`
    *   Implementations: `StandardVitakka` (Implemented), `HierarchicalVitakka`
*   **`BaseVicara`** (Refiner Orchestrator): `s0 -> s_final`
    *   Path: `samadhi/components/vicara/base.py`
    *   Implementations: `StandardVicara` (Implemented), `WeightedVicara` (Implemented), `ProbeVicara` (Implemented)
*   **`BaseRefiner`**: `Latent State (s) -> Residual (phi(s))`
    *   Path: `samadhi/components/refiners/base.py`
    *   Implementations: `MlpRefiner` (Implemented), `GruRefiner`, `AttentionRefiner`
*   **`BaseDecoder`**: `Tensor(Batch, Dim) -> Output`
    *   Path: `samadhi/components/decoders/base.py`
    *   Implementations: `ReconstructionDecoder` (Implemented), `ClassificationDecoder`, `CnnDecoder` (**Implemented**), `LstmDecoder` (**Implemented**), `SimpleSequenceDecoder` (**Implemented**)

### 3.3. Construction (Builder / Factory)

Provide helpers to assemble common configurations easily.

```python
# Usage Example
model = SamadhiBuilder(dim=64) \
    .set_adapter("cnn", image_size=(28, 28)) \
    .set_refiner("gru", steps=10) \
    .set_decoder("classification", num_classes=10) \
    .build()
```

-----

## 4. Migration Strategy

We will proceed in phases to avoid breaking existing functionality.

### Phase 1: Component Decoupling
*   Extract `Adapter` and `Decoder` logic from `SamadhiModel` subclasses into independent classes in `samadhi/components/adapters/` and `samadhi/components/decoders/`.
    *   **Status:** `MlpAdapter`, `CnnAdapter`, `LstmAdapter`, `TransformerAdapter` and `ReconstructionDecoder`, `CnnDecoder`, `LstmDecoder`, `SimpleSequenceDecoder` have been successfully extracted and implemented.
*   Extract `Refiner` logic from `Vicara` into independent classes in `samadhi/components/refiners/`.
    *   **Status:** `MlpRefiner` has been successfully extracted and implemented.
*   Refactor `Vitakka` and `Vicara` into their respective packages (`samadhi/components/vitakka/` and `samadhi/components/vicara/`) with base classes and implementations.
    *   **Status:** `StandardVitakka`, `StandardVicara`, `WeightedVicara`, `ProbeVicara` have been successfully implemented and restructured.
*   Ensure all components rely only on standard Tensor interfaces.
    *   **Status:** Completed.

### Phase 2: Core Refactoring (`SamadhiEngine`)
*   Create the new `SamadhiEngine` class (now `samadhi/core/engine.py`).
*   **Status:** Completed. The `SamadhiEngine` is now the core container, directly used for model construction.

### Phase 3: Preset Definitions
*   Redefine `MlpSamadhiModel`, `LstmSamadhiModel`, `ConvSamadhiModel`, `TransformerSamadhiModel` as factory functions that return a configured `SamadhiEngine`.
    *   *Old:* `model = MlpSamadhiModel(config)`
    *   *New:* `model = create_mlp_samadhi(config)` (from `samadhi/presets/tabular.py`), `create_conv_samadhi(config)` (from `samadhi/presets/vision.py`), `create_lstm_samadhi(config)` (from `samadhi/presets/sequence.py`), `create_transformer_samadhi(config)` (from `samadhi/presets/sequence.py`).
*   **Status:** Completed. All legacy model classes have been replaced by factory functions in `samadhi/presets/`.

### Phase 4: Trainer Generalization (Objective-Driven)
*   Refactor `BaseSamadhiTrainer` (and move to a Hugging Face Trainer wrapper) to delegate loss calculation to an injectable **`Objective`** component.
*   **`SamadhiObjective` Interface (samadhi/train/objectives/base_objective.py):**
    *   Defines how to compute `Total Loss` from `(Model Output, Target, Samadhi Metadata)`.
    *   **New properties:**
        *   `needs_vitakka: bool = True`: If `False`, the `SamadhiEngine` will skip the Vitakka (Search) component and treat the Adapter's output directly as the initial latent state.
        *   `needs_vicara: bool = True`: If `False`, the `SamadhiEngine` will skip the Vicara (Refinement) component.
    *   Implementations:
        *   `AutoencoderObjective` (Reconstruction Loss only, with `needs_vitakka=False`, `needs_vicara=False`)
        *   `SupervisedRegressionObjective` (MSE + Stability + Entropy + Balance)
        *   `SupervisedClassificationObjective` (CrossEntropy + Stability + Entropy + Balance)
        *   `AnomalyObjective` (Reconstruction + Margin + Stability + Entropy + Balance)
        *   `UnsupervisedObjective` (Reconstruction/Stability + Entropy + Balance)
*   **Model-side "Optimal Path Selection API" (`SamadhiEngine.forward`):**
    *   The `SamadhiEngine`'s `forward` method will be updated to accept `run_vitakka: bool` and `run_vicara: bool` arguments.
    *   It will dynamically execute the forward pass based on these flags, ensuring that only necessary components are run.
    *   Example: `output, s_final, meta = self.model(x, run_vitakka=self.objective.needs_vitakka, run_vicara=self.objective.needs_vicara)`
*   **Hugging Face Trainer Integration (`samadhi/train/hf_trainer.py`):**
    *   A custom `SamadhiTrainer` class (inheriting from `transformers.Trainer`) will be created.
    *   Its `compute_loss` method will call `self.model.forward` with the `needs_vitakka` and `needs_vicara` flags from the injected `objective`.
    *   This leverages HF Trainer's robust features (distributed training, mixed precision, logging) while using Samadhi's custom loss and dynamic execution paths.
*   **Parameter Freezing (samadhi/utils/training.py):**
    *   The responsibility for freezing specific components (e.g., `adapter`, `vitakka`, `vicara`, `decoder`) lies **outside** the `Trainer` itself.
    *   A utility function (e.g., `freeze_components(model, components_to_freeze: List[str])`) will be provided.
    *   Users will call this utility **before** initializing the `SamadhiTrainer` to prepare the model for specific training phases (e.g., pre-training only the Adapter and Decoder).
*   **Status:** Completed. The legacy trainers have been removed, and the new Objective-driven SamadhiTrainer integrated.

-----

## 5. Directory Structure After Refactoring

```
samadhi/
├── core/
│   ├── engine.py           # SamadhiEngine (The Container)
│   └── builder.py          # SamadhiBuilder
├── components/
│   ├── adapters/              # Modularized Adapters (入力変換)
│   │   ├── __init__.py
│   │   ├── base.py            # BaseAdapter (抽象基底クラス)
│   │   ├── mlp.py             # MlpAdapter (Implemented)
│   │   ├── vision.py          # CnnAdapter (Implemented)
│   │   ├── sequence.py        # LstmAdapter, TransformerAdapter (Implemented)
│   │   └── language.py        # BertAdapter
│   ├── decoders/              # Modularized Decoders (出力変換)
│   │   ├── __init__.py
│   │   ├── base.py            # BaseDecoder (抽象基底クラス)
│   │   ├── classification.py  # ClassificationDecoder
│   │   ├── reconstruction.py  # ReconstructionDecoder (Implemented)
│   │   ├── vision.py          # CnnDecoder (Implemented)
│   │   └── sequence.py        # LstmDecoder, SimpleSequenceDecoder (Implemented)
│   ├── refiners/              # Modularized Refiners (状態変換ネットワーク)
│   │   ├── __init__.py
│   │   ├── base.py            # BaseRefiner (抽象基底クラス)
│   │   ├── mlp.py             # MlpRefiner (Implemented)
│   │   └── gru.py             # GruRefiner
│   ├── vicara/                # Vicara Modules (浄化プロセスの制御)
│   │   ├── __init__.py
│   │   ├── base.py            # VicaraBase
│   │   ├── standard.py        # StandardVicara (Implemented)
│   │   ├── weighted.py        # WeightedVicara (Implemented)
│   │   └── probe_specific.py  # ProbeVicara (Implemented)
│   └── vitakka/               # Vitakka Modules (探索プロセスの制御)
│       ├── __init__.py
│       ├── base.py            # BaseVitakka
│       └── standard.py        # StandardVitakka (Implemented)
├── presets/                # Pre-configured models (New)
│   ├── __init__.py            # For package recognition
│   ├── tabular.py          # create_mlp_samadhi (Implemented)
│   ├── vision.py           # create_conv_samadhi (Implemented)
│   └── sequence.py         # create_lstm_samadhi, create_transformer_samadhi (Implemented)
├── train/                  # Trainers
│   ├── __init__.py            # For package recognition
│   ├── hf_trainer.py          # SamadhiTrainer (Implemented)
│   └── objectives/            # Objective Definitions
│       ├── __init__.py
│       ├── base_objective.py  # BaseObjective (Implemented, with needs_vitakka/vicara)
│       ├── autoencoder.py     # AutoencoderObjective (Implemented)
│       ├── supervised_regression.py # SupervisedRegressionObjective (Implemented)
│       └── unsupervised.py    # UnsupervisedObjective (Implemented)
└── utils/                  # Utility Functions
    ├── __init__.py
    └── training.py            # freeze_components, etc. (Implemented)
```
