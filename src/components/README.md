# Components Documentation

This directory contains the modular components that make up the Samadhi Model. The architecture is designed to be highly extensible, allowing you to plug in custom implementations for different stages of the processing pipeline.

## Directory Structure

*   **`adapters/`**: Input adapters that convert raw data (images, text, tabular) into the initial latent space `z`.
*   **`decoders/`**: Output decoders that reconstruct data or generate predictions from the purified latent state `s_final`.
*   **`vitakka/`**: "Applied Thought" or Search modules. Responsible for finding the initial "resonance" or concept `s0` from the adapted input.
*   **`vicara/`**: "Sustained Thought" or Refinement modules. Responsible for recursively purifying the state `s` to remove noise.
*   **`refiners/`**: Internal transformation networks used by Vicara to calculate state updates (e.g., an MLP or RNN block).

---

## How to Add New Modules

To extend the framework, create a new class inheriting from the appropriate Base class.

### 1. Adapters (Input)
Create a file in `src/components/adapters/`.

```python
from src.components.adapters.base import BaseAdapter
import torch

class MyCustomAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your layers here
        # self.net = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform x (Batch, InputDim) -> z (Batch, LatentDim)
        return self.net(x)
```

### 2. Decoders (Output)
Create a file in `src/components/decoders/`.

```python
from src.components.decoders.base import BaseDecoder
import torch

class MyCustomDecoder(BaseDecoder):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your layers here

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Transform s (Batch, LatentDim) -> output (Batch, OutputDim)
        return self.net(s)
```

### 3. Refiners (Internal Logic)
Refiners are the "brains" inside the Vicara loop. Create a file in `src/components/refiners/`.

```python
from src.components.refiners.base import BaseRefiner
import torch

class MyCustomRefiner(BaseRefiner):
    def __init__(self, config):
        super().__init__(config)
        # Define the transformation logic

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Calculate the "residual" or update vector
        # s (Batch, Dim) -> residual (Batch, Dim)
        return self.net(s)
```

### 4. Vitakka (Search Strategy)
If you need a new way to initialize the state (e.g., different attention mechanism).

```python
from src.components.vitakka.base import BaseVitakka
import torch

class MyVitakka(BaseVitakka):
    def __init__(self, config):
        super().__init__(config)
        # self.probes is already initialized by BaseVitakka

    def forward(self, z_adapted: torch.Tensor):
        # Implement search logic to find s0
        # ...
        return s0, {"log_info": ...}
```

### 5. Vicara (Refinement Strategy)
If you need to change *how* the refinement loop works (though usually, just changing the `Refiner` is enough).

```python
from src.components.vicara.base import BaseVicara
import torch

class MyVicara(BaseVicara):
    def _refine_step(self, s_t: torch.Tensor, context) -> torch.Tensor:
        # Implement a single step of purification
        # Typically calls self.refiners
        # Returns the residual to be added to s_t
        pass
```

## Integration

After creating your component, you can use it in two ways:

1.  **Direct Builder Usage**: Pass the instance to `SamadhiBuilder`.
    ```python
    adapter = MyCustomAdapter(config)
    builder = SamadhiBuilder(config)
    builder.set_adapter(adapter)
    model = builder.build()
    ```

2.  **Preset Creation**: Create a new factory function in `src/presets/` that instantiates your new component.

## Testing New Components

When adding new components, it is crucial to also add corresponding unit tests to ensure their correctness and integration. Tests should be placed in the `tests/components/` directory, mirroring the structure of `src/components/`.

For example:
*   An adapter in `src/components/adapters/my_adapter.py` should have its tests in `tests/components/adapters/test_my_adapter.py`.
*   A decoder in `src/components/decoders/my_decoder.py` should have its tests in `tests/components/decoders/test_my_decoder.py`.

Refer to existing test files in `tests/components/` for examples of how to structure your tests. Ensure your tests cover various scenarios, including edge cases and error handling.
