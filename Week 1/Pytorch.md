# 1 Pytorch and Numpy Integration


Defining Tensor and Array
```python
tensor = torch.tensor([1,2,3])
array = np.array([1,2,3])

```

To conververt Numpy array too tensor:
```python
torch.from_numpy(array)
```
To convert tensor to numpy array;
```python
tensor.numpy()   
```
---
# 2 Reproducibility in Pytorch
In machine learning, reproducibility means that you (or someone else) can rerun the same experiment using the same code and obtain the same results, within expected stochastic variation.

*How to Set Random Seeds*

**For CPU**
``` python
import torch

random_seed = 42
torch.manual_seed(random_seed)

```

**For GPU (CUDA)**
```python
import torch

random_seed = 42
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

```

>**Why Set Seeds Separately for CPU and GPU?**
CPU and GPU are separate devices with separate random number generators (RNGs). Setting one does not automatically set the other.


**For full reproducibility across devices**
```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)                    # Python's random module
    np.random.seed(seed)                 # NumPy
    torch.manual_seed(seed)              # PyTorch CPU
    torch.cuda.manual_seed_all(seed)     # PyTorch all GPUs

set_seed(42)
```

---
# Hardware Acceleration in Pytorch
