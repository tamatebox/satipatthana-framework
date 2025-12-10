# Components Documentation

This directory contains the modular components that make up the Samadhi Model. The architecture is designed to be highly extensible, allowing you to plug in custom implementations for different stages of the processing pipeline.

## Directory Structure

### Core Components (v3.x)

* **`adapters/`**: Input adapters that convert raw data (images, text, tabular) into the initial latent space `z`.
* **`decoders/`**: Output decoders that reconstruct data or generate predictions from the purified latent state `s_final`.
* **`vitakka/`**: "Applied Thought" or Search modules. Responsible for finding the initial "resonance" or concept `s0` from the adapted input.
* **`vicara/`**: "Sustained Thought" or Refinement modules. Responsible for single-step state updates `s_t → s_{t+1}`.
* **`refiners/`**: Internal transformation networks used by Vicara to calculate state updates (e.g., an MLP or RNN block).

### New Components (v4.0)

* **`augmenters/`**: Input augmentation modules that apply environmental noise to raw data for robust training. Returns `(x_augmented, severity)`.
* **`sati/`**: "Mindfulness" or Gating modules. Monitors the state trajectory (SantanaLog) and determines when to stop the Vicara loop.
* **`vipassana/`**: "Insight" or Meta-cognition modules. Analyzes the thinking process to produce context vectors and trust scores.
* **`objectives/`**: Training objective components defining loss functions. Moved from `samadhi/train/objectives/` for better organization.

---

## How to Add New Modules

To extend the framework, create a new class inheriting from the appropriate Base class.

**Crucially, when adding new components, you must also define their corresponding configuration dataclass within the `samadhi/configs/` directory.**

### 1. Adapters (Input)

Create a file in `samadhi/components/adapters/`.

```python
from samadhi.components.adapters.base import BaseAdapter
import torch
from samadhi.configs.adapters import MyCustomAdapterConfig # Import your config

class MyCustomAdapter(BaseAdapter):
    # Type hint with your specific config class
    def __init__(self, config: MyCustomAdapterConfig):
        super().__init__(config)
        # Initialize your layers here, accessing config via dot notation:
        # self.net = nn.Linear(self.config.input_dim, self.config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform x (Batch, InputDim) -> z (Batch, LatentDim)
        return self.net(x)
```

**Corresponding Configuration (`samadhi/configs/adapters.py` example):**

```python
from dataclasses import dataclass
from samadhi.configs.base import BaseConfig
from samadhi.configs.enums import AdapterType

@dataclass
class MyCustomAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.MY_CUSTOM_TYPE # Define new enum in enums.py
    my_custom_param: int = 100
```

### 2. Decoders (Output)

Create a file in `samadhi/components/decoders/`.

```python
from samadhi.components.decoders.base import BaseDecoder
import torch
from samadhi.configs.decoders import MyCustomDecoderConfig

class MyCustomDecoder(BaseDecoder):
    def __init__(self, config: MyCustomDecoderConfig):
        super().__init__(config)
        # Initialize your layers here

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Transform s (Batch, LatentDim) -> output (Batch, OutputDim)
        return self.net(s)
```

### 3. Refiners (Internal Logic)

Refiners are the "brains" inside the Vicara loop. Create a file in `samadhi/components/refiners/`.

```python
from samadhi.components.refiners.base import BaseRefiner
import torch
# Refiners typically use a general BaseConfig or VicaraConfig for shared params like 'dim'
from samadhi.configs.base import BaseConfig

class MyCustomRefiner(BaseRefiner):
    def __init__(self, config: BaseConfig):
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
from samadhi.components.vitakka.base import BaseVitakka
import torch
from samadhi.configs.vitakka import MyVitakkaConfig

class MyVitakka(BaseVitakka):
    def __init__(self, config: MyVitakkaConfig):
        super().__init__(config)
        # self.probes is already initialized by BaseVitakka

    def forward(self, z_adapted: torch.Tensor):
        # Implement search logic to find s0
        # ...
        return s0, {"log_info": ...}
```

### 5. Vicara (Refinement Strategy)

If you need to change *how* the refinement loop works (though usually, just changing the `Refiner` is enough).

**Note (v4.0)**: Vicara now only performs single-step state updates. Loop control is delegated to SamathaEngine.

```python
from samadhi.components.vicara.base import BaseVicara
import torch
from samadhi.configs.vicara import MyVicaraConfig

class MyVicara(BaseVicara):
    def __init__(self, config: MyVicaraConfig, refiners):
        super().__init__(config, refiners)
        # ...

    def _refine_step(self, s_t: torch.Tensor, context) -> torch.Tensor:
        # Implement a single step of purification
        # Typically calls self.refiners
        # Returns the residual to be added to s_t
        pass
```

### 6. Augmenters (Input Augmentation) - v4.0

Augmenters apply environmental noise to raw input data for robust training.

```python
from samadhi.components.augmenters.base import BaseAugmenter
import torch
from typing import Tuple
from samadhi.configs.augmenter import MyAugmenterConfig

class MyAugmenter(BaseAugmenter):
    def __init__(self, config: MyAugmenterConfig):
        super().__init__(config)
        # Initialize augmentation parameters

    def forward(self, x: torch.Tensor, noise_level: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply augmentation to input
        # x: (Batch, *) -> x_augmented: (Batch, *)
        # severity: (Batch,) - per-sample noise intensity
        return x_augmented, severity
```

### 7. Sati (Convergence Monitoring) - v4.0

Sati monitors the state trajectory and determines when to stop the Vicara loop.

```python
from samadhi.components.sati.base import BaseSati
from samadhi.core.santana import SantanaLog
import torch
from typing import Tuple, Dict, Any
from samadhi.configs.sati import MySatiConfig

class MySati(BaseSati):
    def __init__(self, config: MySatiConfig):
        super().__init__(config)
        # Initialize stopping criteria

    def forward(self, current_state: torch.Tensor, santana: SantanaLog) -> Tuple[bool, Dict[str, Any]]:
        # Evaluate whether to stop the Vicara loop
        # Returns: (should_stop, info_dict)
        return should_stop, {"reason": "converged", "energy": ...}
```

### 8. Vipassana (Meta-cognition) - v4.0

Vipassana analyzes the thinking process to produce context vectors and trust scores.

```python
from samadhi.components.vipassana.base import BaseVipassana
from samadhi.core.santana import SantanaLog
import torch
from typing import Tuple
from samadhi.configs.vipassana import MyVipassanaConfig

class MyVipassana(BaseVipassana):
    def __init__(self, config: MyVipassanaConfig):
        super().__init__(config)
        # Initialize log encoder and confidence monitor

    def forward(self, s_star: torch.Tensor, santana: SantanaLog) -> Tuple[torch.Tensor, float]:
        # Analyze thinking trajectory
        # s_star: (Batch, Dim) - converged state
        # santana: SantanaLog - trajectory history
        # Returns: (v_ctx, trust_score)
        #   v_ctx: (Batch, context_dim) - context vector ("doubt" embedding)
        #   trust_score: float (0.0-1.0) - confidence score
        return v_ctx, trust_score
```

### 9. Objectives (Training Loss Functions)

Objectives define the loss functions used during training. They control what the model learns and which components are active during training.

```python
from samadhi.components.objectives.base_objective import BaseObjective
import torch
from typing import Dict, Any, Tuple, Optional

class MyCustomObjective(BaseObjective):
    # Define which components are needed
    needs_vitakka = True
    needs_vicara = True

    def compute_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        s0: torch.Tensor,
        s_final: torch.Tensor,
        decoded_s_final: torch.Tensor,
        metadata: Dict[str, Any],
        num_refine_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Calculate your custom loss
        # Return (total_loss, loss_components_dict)
        pass
```

**Available Objectives:**

* `AutoencoderObjective`: Pre-training (skips Vitakka/Vicara)
* `UnsupervisedObjective`: Reconstruction + stability + entropy regularization
* `SupervisedRegressionObjective`: Regression with MSE loss
* `SupervisedClassificationObjective`: Classification with CrossEntropy loss
* `RobustRegressionObjective`: Regression with Huber loss (robust to outliers)
* `CosineSimilarityObjective`: Semantic alignment via cosine similarity
* `AnomalyObjective`: Anomaly detection with margin-based loss

## Integration into the Framework

After creating your component and its corresponding configuration, follow these steps to integrate it:

1. **Update `samadhi/configs/enums.py`**: Add a new member to the relevant `Enum` (e.g., `AdapterType`) for your new component type.
2. **Update `samadhi/configs/factory.py`**: Modify the appropriate `create_*_config` function (e.g., `create_adapter_config`) to include logic for instantiating your new configuration dataclass based on its `type`.
3. **Update `samadhi/core/builder.py`**: If your component is directly settable via the `SamadhiBuilder` (like Adapter, Vitakka, Vicara, Decoder), update the relevant `set_*` method to instantiate your component using its new configuration.
4. **Preset Creation (Optional)**: Create a new factory function in `samadhi/presets/` that instantiates your new component as part of a predefined model configuration.

## Testing New Components

When adding new components, it is crucial to also add corresponding unit tests to ensure their correctness and integration. Tests should be placed in the `tests/components/` directory, mirroring the structure of `samadhi/components/`.

For example:

* An adapter in `samadhi/components/adapters/my_adapter.py` should have its tests in `tests/components/adapters/test_my_adapter.py`.
* A decoder in `samadhi/components/decoders/my_decoder.py` should have its tests in `tests/components/decoders/test_my_decoder.py`.

Refer to existing test files in `tests/components/` for examples of how to structure your tests. Ensure your tests cover various scenarios, including edge cases and error handling.
