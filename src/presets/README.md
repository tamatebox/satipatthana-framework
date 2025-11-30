# Presets Documentation

This directory contains factory functions, or "presets," for easily constructing common configurations of the Samadhi Model. These presets encapsulate the process of assembling various Adapters, Decoders, Vitakka, and Vicara modules using the `SamadhiBuilder`.

## Purpose

Presets are designed to:
*   **Simplify Model Creation**: Provide ready-to-use model configurations for typical use cases (e.g., tabular data, image data, sequence data).
*   **Promote Best Practices**: Demonstrate how to correctly combine components to form a functional Samadhi model.
*   **Reduce Boilerplate**: Abstract away the detailed component instantiation and wiring logic.

---

## How to Use Existing Presets

Each preset function (e.g., `create_mlp_samadhi`, `create_conv_samadhi`) takes a `config` dictionary and returns a fully constructed `SamadhiEngine` (which is a `torch.nn.Module`).

```python
from src.presets.tabular import create_mlp_samadhi
from src.presets.vision import create_conv_samadhi

config = {
    "dim": 128,
    "n_probes": 16,
    "refine_steps": 5,
    # ... other model-specific configurations
}

# Create a model for tabular data
mlp_model = create_mlp_samadhi(config)
print(mlp_model)

# Create a model for vision data
conv_model = create_conv_samadhi(config)
print(conv_model)
```

---

## How to Create New Presets

When you have a recurring combination of components that you want to make easily accessible, you can create a new preset. This involves defining a function that uses the `SamadhiBuilder` to set up the desired components.

1.  **Create a New File**: Add a new Python file (e.g., `src/presets/my_new_preset.py`) in this directory.

2.  **Define a Factory Function**: Inside the file, create a function that takes a `config` dictionary as input.

3.  **Instantiate Components**: Import and instantiate the specific `Adapter`, `Decoder`, `Vitakka`, `Vicara`, and `Refiner` components that your preset will use.

4.  **Use `SamadhiBuilder`**: Use `SamadhiBuilder` to assemble these components into an `SamadhiEngine` instance.

    ```python
    from typing import Dict, Any
    import torch.nn as nn
    from src.core.builder import SamadhiBuilder
    from src.components.adapters.sequence import LstmAdapter
    from src.components.decoders.sequence import LstmDecoder

    def create_lstm_samadhi(config: Dict[str, Any]) -> nn.Module:
        """
        Creates an LSTM-based Samadhi model for sequence data.
        """
        adapter = LstmAdapter(config)
        decoder = LstmDecoder(config)

        engine = (
            SamadhiBuilder(config)
            .set_adapter(adapter)
            .set_vitakka()  # Example: use default Vitakka
            .set_vicara(refiner_type="mlp") # Example: use default Vicara with MlpRefiner
            .set_decoder(decoder)
            .build()
        )
        return engine
    ```

5.  **Add Tests**: Ensure you add corresponding unit tests for your new preset in `tests/presets/` to verify that it correctly constructs the model with the intended components.
