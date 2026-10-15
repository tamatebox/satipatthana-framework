# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satipatthana Framework is an introspective recursive attention architecture designed for converging to essential structures from noisy data and introspecting the process to explain its own confidence.

The name "Satipatthana" (念処) means "establishment of mindfulness", symbolizing the architecture's essence of discerning truth through self-observation and introspection.

The architecture maps Buddhist psychology concepts to engineering:

- **Vitakka (Search)**: Semantic initialization - generates meaningful initial state S₀ from concept probes
- **Vicara (Refinement)**: Contractive refinement - single-step state update towards fixed point
- **Sati (Convergence Check)**: Monitors state change and determines stopping
- **Santāna (Continuity)**: Tracks state dynamics over time (SantanaLog)
- **Vipassana (Insight)**: Meta-cognition analyzing the thinking process → Triple Score (trust, conformity, confidence) + context vector

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/components/vicara/test_standard_vicara.py

# Run tests matching a pattern
uv run pytest -k "test_vitakka"

# Format code
uv run black .

# Type check (if added)
uv run mypy satipatthana/
```

## Code Style

- **Formatter**: Black (line-length: 119)
- **Python**: 3.12+
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google style for classes and public methods

### Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `SatipatthanaSystem`, `MlpAdapter` |
| Functions/Methods | snake_case | `forward()`, `create_adapter_config()` |
| Config classes | PascalCase + `Config` suffix | `SystemConfig`, `MlpAdapterConfig` |
| Constants | UPPER_SNAKE_CASE | `MAX_STEPS`, `DEFAULT_DIM` |
| Private methods | Leading underscore | `_build_layers()`, `_compute_energy()` |

### Component Pattern

All components follow the same structure:

```python
class MyComponent(nn.Module):
    """One-line description.

    Longer description if needed.

    Args:
        config: Configuration for this component.
    """

    def __init__(self, config: MyComponentConfig):
        super().__init__()
        self.config = config
        self._build_layers()

    def _build_layers(self):
        """Build internal layers."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, dim).

        Returns:
            Output tensor of shape (batch, out_dim).
        """
        pass
```

## Architecture

### Three-Engine Structure

```txt
Raw Input (X)
    ↓
[SamathaEngine] - The Meditator
    Augmenter → Adapter → Vitakka → Vicara loop (w/ Sati) → S*, SantanaLog
    ↓
[VipassanaEngine] - The Observer
    S* + SantanaLog → V_ctx, Triple Score (trust, conformity, confidence)
    ↓
[ConditionalDecoder] - The Speaker
    S* ⊕ V_ctx → Output (Y)
```

### Directory Structure

```txt
satipatthana/
├── core/           # Main system and engines
│   ├── system.py   # SatipatthanaSystem (orchestrator)
│   ├── engines.py  # SamathaEngine, VipassanaEngine
│   └── santana.py  # SantanaLog (trajectory)
├── components/     # Modular components
│   ├── adapters/   # Input transformers (MLP, CNN, LSTM, Transformer)
│   ├── augmenters/ # Input augmentation (Identity, Gaussian)
│   ├── vitakka/    # Semantic initialization
│   ├── vicara/     # Refinement (Standard, Weighted, ProbeSpecific)
│   ├── refiners/   # Core networks (MLP, GRU) used by Vicara
│   ├── sati/       # Convergence monitors (FixedStep, Threshold)
│   ├── vipassana/  # Meta-cognition modules
│   ├── decoders/   # Output transformers
│   └── objectives/ # Training objectives
├── configs/        # Type-safe dataclass configs
│   ├── system.py   # SystemConfig (root)
│   ├── factory.py  # create_*_config() factories
│   └── *.py        # Component-specific configs
├── train/          # Training
│   └── trainer.py  # SatipatthanaTrainer (4-stage curriculum)
└── utils/          # Utilities
    └── logger.py   # Logging setup
```

## Training Strategy (4-Stage Curriculum)

| Stage | Name | Trainable | Objective |
|-------|------|-----------|-----------|
| 0 | Adapter Pre-training | Adapter, AdapterReconHead | Reconstruction |
| 1 | Samatha Training | Adapter, Vitakka, Vicara, Sati, SamathaReconHead, AuxHead | Stability + Recon + (Guidance) |
| 2 | Vipassana Training | Vipassana | Triple Score BCE (trust, conformity, confidence) |
| 3 | Decoder Fine-tuning | ConditionalDecoder | Task-specific |

**Important:** `AuxHead` (Stage 1, input: S\* only, dim=d) and `ConditionalDecoder` (Stage 3, input: S\* ⊕ V\_ctx, dim=d+c) are physically separate modules. AuxHead is discarded after Stage 1.

```python
from satipatthana.train import SatipatthanaTrainer
from satipatthana.core.system import TrainingStage

trainer = SatipatthanaTrainer(model=system, args=args, train_dataset=dataset)

# Run full curriculum
results = trainer.run_curriculum(
    stage0_epochs=5,
    stage1_epochs=10,
    stage2_epochs=5,
    stage3_epochs=5,
)
```

## Key Patterns

### SatipatthanaSystem Forward Pass

```python
result = system(x)
# result.output: decoded result
# result.s_star: converged latent state
# result.v_ctx: context vector from Vipassana
# result.trust_score: OOD detection score (0.0-1.0), static metrics only
# result.conformity_score: trajectory anomaly score (0.0-1.0), dynamic context only
# result.confidence_score: comprehensive score (0.0-1.0), static + dynamic
# result.santana: SantanaLog with trajectory
# result.severity: augmentation severity
```

### Triple Score System

Vipassana outputs three complementary scores for different use cases:

| Score | Input | GRU Gradient | Purpose |
|-------|-------|--------------|---------|
| `trust_score` | Static metrics only (8 Grounding Metrics) | No | Pure OOD detection |
| `conformity_score` | Dynamic context only (GRU output) | Yes | Trajectory anomaly detection |
| `confidence_score` | Static + Dynamic (both) | Yes | Comprehensive assessment |

**Static Metrics (8 Grounding Metrics)**:

- `s0_energy`, `s_star_energy`: Energy at initial/final state
- `s0_min_dist`, `s_star_min_dist`: Distance from training probes
- `delta_norm`, `drift_magnitude`: State change metrics
- `num_steps`: Convergence steps
- `recon_error`: Reconstruction quality

**Usage Guidelines**:

- Use `trust_score` for OOD detection (no GRU gradient pollution)
- Use `conformity_score` for trajectory monitoring
- Use `confidence_score` for general confidence estimation

### Dataset Format

See [docs/data_format.md](docs/data_format.md) for comprehensive data format specification.

```python
def __getitem__(self, idx):
    return {
        "x": self.data[idx],   # Required: input tensor
        "y": self.labels[idx]  # Optional: labels (for Stage 1 guidance, Stage 3)
    }
```

### VoidDataset (Stage 2 OOD Data)

```python
from satipatthana.data import VoidDataset, FilteredNoiseVoid, GaussianNoiseVoid

# Wrap existing dataset
void_dataset = VoidDataset(ood_dataset)

# Or generate filtered noise far from training data
void_dataset = FilteredNoiseVoid(
    reference_data=train_data_tensor,
    shape=(input_dim,),
    length=5000,
    min_distance=0.3,
    noise_range=(-1.5, 1.5),
)

# Pass to trainer for Stage 2
trainer = SatipatthanaTrainer(..., void_dataset=void_dataset)
```

### Mandatory Config Fields

- **MlpAdapterConfig**: `input_dim`
- **LstmAdapterConfig**: `input_dim`, `seq_len`
- **CnnAdapterConfig**: `img_size`, `channels`
- **ReconstructionDecoderConfig**: `input_dim`
- **ConditionalDecoderConfig**: `dim`, `context_dim`, `output_dim`
- **StandardVipassanaConfig**: `context_dim` (optional: `trust_weight`, `conformity_weight`, `confidence_weight`)

## Stage 2 Noise Generation Strategy

Four strategies to teach Vipassana meta-cognition:

1. **Augmented Path**: Input noise → Target: `1.0 - severity`
2. **Drunk Path**: Internal perturbation (`drunk_mode=True`) → Target: `0.0`
   - Implementations: Dropout rate increase, weight noise, temperature disturbance
3. **Mismatch Path**: Shuffle S* and SantanaLog → Target: `0.0`
4. **Void Path**: OOD samples (VoidDataset) → Target: `0.0`
   - Trains Grounding Metrics (`s0_min_dist`, `drift_magnitude`, `recon_error`) for OOD detection

## Documentation

| Document | Description |
|----------|-------------|
| [docs/specification_en.md](docs/specification_en.md) | v4 Architecture Specification |
| [docs/theory_en.md](docs/theory_en.md) | Theoretical Background & Buddhist Psychology Mapping |
| [docs/training_strategy.md](docs/training_strategy.md) | 4-Stage Training Strategy & Pattern Selection (A-F) |
| [docs/data_format.md](docs/data_format.md) | Data Format & VoidDataset Specification |
| [docs/config_summary.md](docs/config_summary.md) | Config Reference & Required Fields |
| [docs/workflow_guide.md](docs/workflow_guide.md) | Implementation Cookbook (Copy-Paste Code) |
| [docs/logging.md](docs/logging.md) | Logging Configuration |
| [docs/diagrams/](docs/diagrams/) | PlantUML Diagrams |

## Testing Guidelines

- Place tests in `tests/` mirroring source structure
- Use `pytest` fixtures from `tests/conftest.py`
- Test naming: `test_<function_name>_<scenario>`
- Each component should have unit tests for:
  - Forward pass shape verification
  - Edge cases (empty input, batch size 1)
  - Config validation

## Issue Management

- Create a **GitHub Issue** when new problems arise
- Use labels: `bug`, `training`, `enhancement`, `v4-architecture`, etc.
- Close issues after resolution and link related commits/PRs
- Archived resolved issues: [docs/archive/issues/](docs/archive/issues/)
