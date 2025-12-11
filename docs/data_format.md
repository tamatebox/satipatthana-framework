# Data Format Specification

This document defines the data format requirements for the Satipatthana Framework.

---

## 1. Dataset Interface

All datasets must implement PyTorch's `Dataset` interface and return a dictionary from `__getitem__`.

### 1.1. Required Format

```python
def __getitem__(self, idx) -> dict:
    return {
        "x": torch.Tensor,  # Required: input data
        "y": Any,           # Optional: label (for Stage 1 guidance, Stage 3)
    }
```

### 1.2. Supported Key Aliases

The framework automatically normalizes these keys to `"x"`:

| Key | Use Case |
|-----|----------|
| `"x"` | Standard input key |
| `"input_values"` | HuggingFace audio models |
| `"pixel_values"` | HuggingFace vision models |

If none of these keys exist, the first value in the dict is used.

### 1.3. Tuple/List Format

For compatibility with torchvision-style datasets:

```python
# (data, label) tuple -> {"x": data}
def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
```

---

## 2. Tensor Shape Requirements

### 2.1. By Adapter Type

| Adapter | Expected Shape | Example |
|---------|---------------|---------|
| `MlpAdapter` | `(Batch, input_dim)` | `(32, 784)` for MNIST |
| `CnnAdapter` | `(Batch, C, H, W)` | `(32, 3, 32, 32)` for CIFAR |
| `LstmAdapter` | `(Batch, seq_len, input_dim)` | `(32, 100, 16)` |
| `TransformerAdapter` | `(Batch, seq_len, input_dim)` | `(32, 512, 768)` |

### 2.2. Label Shape

| Task Type | Label Shape | Example |
|-----------|-------------|---------|
| Classification | `(Batch,)` or `(Batch, 1)` | `torch.tensor([0, 1, 2, 0])` |
| Regression | `(Batch, output_dim)` | `torch.tensor([[0.5], [0.8]])` |
| Multi-label | `(Batch, num_classes)` | `torch.tensor([[1,0,1], [0,1,0]])` |

---

## 3. Normalization Guidelines

### 3.1. Recommended Preprocessing

**Standard Normalization (Recommended):**
```python
# Mean=0, Std=1
x_normalized = (x - x.mean()) / x.std()
```

**Min-Max Normalization:**
```python
# Range [0, 1]
x_normalized = (x - x.min()) / (x.max() - x.min())
```

**Domain-Specific:**
```python
# Image: ImageNet normalization
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Audio: typically [-1, 1] or mean/std per sample
```

### 3.2. Consistency Requirements

| Component | Expectation |
|-----------|-------------|
| Train Dataset | Normalized consistently |
| Eval Dataset | Same normalization as train |
| VoidDataset | Match train data range (see Section 4.2) |
| ReconstructionDecoder | Output matches input normalization |

---

## 4. VoidDataset (Stage 2 OOD Data)

### 4.1. Classes

| Class | Purpose | Use Case |
|-------|---------|----------|
| `VoidDataset` | Unified wrapper | Wrap any Dataset or generator |
| `GaussianNoiseVoid` | Gaussian noise | Simple OOD baseline |
| `UniformNoiseVoid` | Uniform noise | Bounded OOD samples |
| `FilteredNoiseVoid` | Rejection sampling | Guaranteed far-from-train OOD |

### 4.2. Matching Train Data Range

**Critical:** VoidDataset noise range should match training data normalization.

```python
# If train data is normalized to [-1, 1]:
void_dataset = GaussianNoiseVoid(
    shape=(input_dim,),
    length=10000,
    scale=1.0,  # Similar std to train data
    mean=0.0,
)

# Or use FilteredNoiseVoid with matching range:
void_dataset = FilteredNoiseVoid(
    reference_data=train_data_tensor,
    shape=(input_dim,),
    length=10000,
    min_distance=0.3,
    noise_range=(-1.5, 1.5),  # Slightly beyond train range
)
```

### 4.3. Static vs Dynamic Mode

**Static Mode:** Wrap existing dataset
```python
# From existing dataset
ood_images = torchvision.datasets.CIFAR100(...)
void_dataset = VoidDataset(ood_images)

# From tensor list
ood_tensors = [torch.randn(784) for _ in range(1000)]
void_dataset = VoidDataset(ood_tensors)
```

**Dynamic Mode:** Generate on-the-fly
```python
# Lambda generator (requires length)
void_dataset = VoidDataset(
    lambda: {"x": torch.randn(784)},
    length=10000
)

# Convenience classes
void_dataset = GaussianNoiseVoid(shape=(784,), length=10000)
```

---

## 5. Stage-Specific Requirements

### 5.1. Stage 0 (Adapter Pre-training)

| Field | Required | Notes |
|-------|----------|-------|
| `"x"` | Yes | Input tensor |
| `"y"` | No | Labels not used |

### 5.2. Stage 1 (Samatha Training)

| Field | Required | Notes |
|-------|----------|-------|
| `"x"` | Yes | Input tensor |
| `"y"` | If `use_label_guidance=True` | For AuxHead guidance loss |

### 5.3. Stage 2 (Vipassana Training)

**Train Dataset:**
| Field | Required | Notes |
|-------|----------|-------|
| `"x"` | Yes | Clean input for contrastive learning |
| `"y"` | No | Labels not used |

**VoidDataset:**
| Field | Required | Notes |
|-------|----------|-------|
| `"x"` | Yes | OOD samples (target: trust=0.0) |

### 5.4. Stage 3 (Decoder Fine-tuning)

| Field | Required | Notes |
|-------|----------|-------|
| `"x"` | Yes | Input tensor |
| `"y"` | Yes | Task-specific labels |

---

## 6. Examples

### 6.1. Custom Dataset Implementation

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)

        sample = {"x": x}
        if self.labels is not None:
            sample["y"] = self.labels[idx]
        return sample
```

### 6.2. Using torchvision Datasets

```python
import torchvision
from torchvision import transforms

# Wrapper for torchvision datasets
class TorchvisionWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return {"x": x, "y": y}

# Usage
mnist = torchvision.datasets.MNIST(
    root="./data",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten
    ])
)
train_dataset = TorchvisionWrapper(mnist)
```

### 6.3. VoidDataset with FilteredNoiseVoid

```python
from satipatthana.data import FilteredNoiseVoid

# Get training data tensor for reference
train_tensors = torch.stack([train_dataset[i]["x"] for i in range(len(train_dataset))])

# Create OOD dataset far from training data
void_dataset = FilteredNoiseVoid(
    reference_data=train_tensors[:500],  # Subsample for efficiency
    shape=(784,),
    length=5000,
    min_distance=0.5,
    noise_range=(-2.0, 2.0),
)

# Pass to trainer
trainer = SatipatthanaTrainer(
    model=system,
    args=training_args,
    train_dataset=train_dataset,
    void_dataset=void_dataset,
    stage=TrainingStage.VIPASSANA_TRAINING,
)
```

---

## 7. Troubleshooting

### 7.1. Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'x'` | Dataset returns wrong format | Ensure `__getitem__` returns `{"x": tensor}` |
| `Shape mismatch` | Adapter input_dim doesn't match data | Check `MlpAdapterConfig(input_dim=...)` |
| `Poor OOD detection` | VoidDataset range mismatched | Match `noise_range` to train data range |

### 7.2. Debugging Tips

```python
# Check dataset output
sample = dataset[0]
print(f"Keys: {sample.keys()}")
print(f"x shape: {sample['x'].shape}")
print(f"x range: [{sample['x'].min():.2f}, {sample['x'].max():.2f}]")

# Check VoidDataset output
void_sample = void_dataset[0]
print(f"Void x range: [{void_sample['x'].min():.2f}, {void_sample['x'].max():.2f}]")
```
