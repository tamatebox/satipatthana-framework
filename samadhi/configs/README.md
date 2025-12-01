# Configuration System for Samadhi Framework

This directory contains the new Type-Safe Dataclass-based configuration system for the Samadhi Framework, replacing the old `Dict[str, Any]` based configuration.

## Overview

The goal of this refactoring is to improve readability, prevent runtime errors due to typos or type mismatches, and enhance maintainability by introducing a structured, type-safe approach to configuration management.

### Key Principles
-   **Type Safety**: All configuration parameters are defined within Python dataclasses with explicit type hints.
-   **Modularity**: Configurations are split into logical units (e.g., adapters, vicara, vitakka, decoders, objectives).
-   **Validation**: Each configuration dataclass can implement a `validate()` method for custom validation and type conversions (e.g., `list` to `tuple`).
-   **Factory Pattern**: Factory functions are used to instantiate the correct configuration class based on a `type` field (e.g., `AdapterType.MLP`, `VicaraType.STANDARD`).
-   **Single Source of Truth**: `SamadhiConfig` acts as the root configuration, encapsulating all component-specific configurations.
-   **Backward Compatibility**: During migration, components can still accept dictionary-based configurations, which are then automatically converted to the corresponding dataclass using the factory functions.

## Directory Structure

-   `base.py`: Contains `BaseConfig`, the abstract base class for all configurations, providing `from_dict` and `validate` methods.
-   `enums.py`: Defines `Enum` classes for component types (e.g., `AdapterType`, `VicaraType`, `DecoderType`) to prevent string-based typos.
-   `adapters.py`: Dataclasses for Adapter component configurations (e.g., `MlpAdapterConfig`, `CnnAdapterConfig`).
-   `vitakka.py`: Dataclasses for Vitakka component configurations (e.g., `StandardVitakkaConfig`).
-   `vicara.py`: Dataclasses for Vicara component configurations (e.g., `StandardVicaraConfig`, `ProbeVicaraConfig`).
-   `decoders.py`: Dataclasses for Decoder component configurations (e.g., `ReconstructionDecoderConfig`, `CnnDecoderConfig`).
-   `objectives.py`: Dataclasses for Objective configurations (e.g., `ObjectiveConfig`).
-   `factory.py`: Contains factory functions (e.g., `create_adapter_config`) responsible for dynamically creating the correct configuration dataclass instance from a dictionary based on its `type` field.
-   `main.py`: Contains `SamadhiConfig`, the top-level configuration dataclass that holds instances of all other component configurations.

## Usage Example

### 1. Creating a `SamadhiConfig`

YouYou can create a `SamadhiConfig` directly or from a dictionary:

```python
from samadhi.configs.main import SamadhiConfig
from samadhi.configs.enums import AdapterType, VicaraType, DecoderType
from samadhi.configs.objectives import ObjectiveConfig

# From dictionary (common for loading from YAML/JSON)
config_data = {
    "dim": 128,
    "adapter": {"type": AdapterType.MLP.value, "input_dim": 256},
    "vicara": {"type": VicaraType.PROBE_SPECIFIC.value, "n_probes": 10},
    "decoder": {"type": DecoderType.RECONSTRUCTION.value, "input_dim": 256},
    "objective": {"stability_coeff": 0.05, "anomaly_margin": 5.0}
}
my_config = SamadhiConfig.from_dict(config_data)

# Or directly using dataclass constructors (for programmatic setup)
from samadhi.configs.adapters import MlpAdapterConfig
from samadhi.configs.vicara import ProbeVicaraConfig
from samadhi.configs.decoders import ReconstructionDecoderConfig

my_config_direct = SamadhiConfig(
    dim=128,
    adapter=MlpAdapterConfig(dim=128, input_dim=256),
    vicara=ProbeVicaraConfig(dim=128, n_probes=10),
    decoder=ReconstructionDecoderConfig(dim=128, input_dim=256),
    objective=ObjectiveConfig(stability_coeff=0.05, anomaly_margin=5.0)
)

assert my_config == my_config_direct
```

### 2. Accessing Configuration Parameters

Access parameters using dot notation, benefiting from static type checking:

```python
# In SamadhiEngine or other components:
class MyComponent:
    def __init__(self, config: SamadhiConfig):
        self.config = config

    def do_something(self):
        latent_dim = self.config.dim
        adapter_input_dim = self.config.adapter.input_dim
        vicara_refine_steps = self.config.vicara.refine_steps
        stability_weight = self.config.objective.stability_coeff

        print(f"Latent Dim: {latent_dim}")
        print(f"Adapter Input Dim: {adapter_input_dim}")
        print(f"Vicara Refine Steps: {vicara_refine_steps}")
        print(f"Stability Weight: {stability_weight}")
```

### 3. Extending Configurations

To add a new component type or parameter:
1.  Create a new `Enum` member in `enums.py` if it's a new component type (e.g., for a new adapter type).
2.  Define a new `@dataclass` in the relevant component file (e.g., `adapters.py`, `objectives.py`) inheriting from its `Base*Config`.
3.  Implement any custom `validate()` logic in the new dataclass.
4.  Update the corresponding `create_*_config` factory function in `factory.py` to handle the new type.
5.  If it's a top-level parameter or a new nested component (like `objective`), update `SamadhiConfig` in `main.py`.

This structured approach ensures consistency, reduces errors, and makes the configuration system more robust and easier to evolve.