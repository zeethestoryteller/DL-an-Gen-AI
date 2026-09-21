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
# 3 Hardware Acceleration in Pytorch

## 1. Check GPU availability (hardware level)

Before touching PyTorch, confirm a GPU is visible to the system:

```bash
!nvidia-smi
```

This shows your GPU name, driver status, and memory usage. If it fails, you'll get an error saying it "couldn't communicate with the NVIDIA driver" — meaning either no GPU is attached or drivers/CUDA aren't installed correctly.

- On **Google Colab**, drivers are pre-configured, so this just works.
- On a **local machine**, you need to install a compatible CUDA toolkit yourself.

## 2. Check if PyTorch can see CUDA

```python
import torch

print(torch.cuda.is_available())  # True/False
```

If `True`, PyTorch can use the GPU.

## 3. Inspect GPU details

```python
print(torch.cuda.device_count())        # number of GPUs
print(torch.cuda.current_device())      # index of current GPU (usually 0)
print(torch.cuda.get_device_name(0))    # e.g. "A100"
```

### Memory info

```python
print(torch.cuda.memory_allocated(0))
print(torch.cuda.memory_reserved(0))
print(torch.cuda.get_device_properties(0).total_memory)
```

These return byte values — divide by 1e9 to get GB. Useful for checking how much memory is actually in use vs. reserved vs. total available.

## 4. Check CUDA/cuDNN/Python versions

```python
import sys
print(sys.version)              # Python version
print(torch.version.cuda)       # CUDA version PyTorch was built with
print(torch.backends.cudnn.version())   # cuDNN version
print(torch.backends.cudnn.enabled)     # True/False
```

This matters because open-source code you download often only works with **specific CUDA/cuDNN versions**. Knowing your versions helps you debug compatibility issues or decide if you need a container/environment with a matching setup.

## 5. Moving tensors to/from the GPU

Create a tensor (defaults to CPU):

```python
tensor = torch.tensor([1, 2, 3])
print(tensor.device)   # cpu
```

Move it to the GPU:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu.device)   # cuda:0
```

The `0` in `cuda:0` refers to the GPU index — if you had multiple GPUs, they'd be numbered `0, 1, 2, ...`.

## 6. Converting GPU tensors to NumPy

You **cannot** convert a GPU tensor directly to NumPy:

```python
tensor_on_gpu.numpy()
# RuntimeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() first.
```

You must move it back to CPU memory first:

```python
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
```

This creates a **copy** of the tensor in CPU/host memory — the original GPU tensor is untouched and still lives on the GPU.

## Key takeaways

| Task | Command |
|---|---|
| Check GPU hardware | `!nvidia-smi` |
| Check PyTorch sees CUDA | `torch.cuda.is_available()` |
| Number of GPUs | `torch.cuda.device_count()` |
| GPU name | `torch.cuda.get_device_name(0)` |
| Move tensor to GPU | `tensor.to("cuda")` or `tensor.cuda()` |
| Move tensor back to CPU | `tensor.cpu()` |
| GPU tensor → NumPy | `tensor.cpu().numpy()` |

**Why this matters:** GPUs (especially NVIDIA, the most widely supported) are hardware-optimized for the parallel matrix operations deep learning relies on, giving **10–100x speedups** over CPU, enabling larger batch sizes, and making production-scale training feasible.

---
# 4 Common Pytorch Errors

## 1. Shape Mismatch Errors (Matrix Multiplication)

Most PyTorch errors come from **shape incompatibilities**, since matrix operations have strict rules about which dimensions can be combined.

**Example problem:**
```python
tensor_a = torch.tensor([[1,2],[3,4],[5,6]], dtype=torch.float32)   # shape: 3x2
tensor_b = torch.tensor([[7,8],[9,10],[11,12]], dtype=torch.float32) # shape: 3x2

torch.matmul(tensor_a, tensor_b)  # ERROR: inner dimensions don't match (2 ≠ 3)
```

**Fix:** Transpose one tensor so the inner dimensions align:
```python
tensor_b_t = tensor_b.T   # shape becomes 2x3

torch.matmul(tensor_a, tensor_b_t)  # works: (3x2) @ (2x3) → 3x3 output
# torch.mm() is a shortcut for matmul() and gives the same result
```

**Rule of thumb:** For `(m x n) @ (n x p)`, the inner dimensions (`n`) must match. Output shape will be `(m x p)`.

## 2. Data Type (dtype) Mismatch Errors

Operating on two tensors with different data types (e.g., `float16` vs `float32`) causes errors.

**Example problem:**
```python
tensor1 = torch.arange(10, 100, 10, dtype=torch.float16)
tensor2 = torch.arange(10, 100, 10)  # defaults to float32

tensor1 @ tensor2  # ERROR: dtype mismatch
```

**Fix:** Cast one tensor to match the other's dtype using `.type()`:
```python
tensor2 = tensor2.type(torch.float16)
tensor1 @ tensor2  # now works — computes the dot product (both are 1D)
```

You can check a tensor's dtype anytime with `tensor.dtype`, and cast to other types similarly:
```python
tensor_int8 = tensor.type(torch.int8)
```

## 3. Device Mismatch Errors (CPU vs GPU)

Operating on tensors that live on **different devices** (one on CPU, one on GPU) throws an error.

**Example problem:**
```python
tensor1 = torch.arange(10, 100, 10, device="cpu")
tensor2 = torch.arange(10, 100, 10, device="cuda")

tensor1 @ tensor2  # ERROR: expected all tensors to be on the same device
```

**Fix:** Move tensors to the same device before operating:
```python
tensor1 = tensor1.to("cuda")
tensor1 @ tensor2  # now works
```

## Summary Table

| Error Type | Cause | Fix |
|---|---|---|
| **Shape mismatch** | Incompatible matrix dimensions | Print `tensor.shape` before operations; use `.T` or `.view()` to reshape |
| **Dtype mismatch** | Mixed precision (e.g., float16 + float32) | Use `tensor.type(torch.float32)` to align types |
| **Device mismatch** | Tensors on CPU and GPU simultaneously | Use `tensor.to(device)` to move tensors to the same device |

## Best Practices for Production Code

- **Be device-agnostic** — write code that works on both CPU and GPU (e.g., using a `device` variable rather than hardcoding).
- **Monitor memory usage** — implement proper cleanup so large tensors don't sit in memory unnecessarily and crash your system.
- **Ensure reproducibility** — set random seeds for consistent experimental results.
- **Add error handling** — especially around tensor operations, to catch shape/dtype/device issues early.

**Quick debugging checklist when you hit a PyTorch error:**
1. Print `tensor.shape` for all tensors involved — are the dimensions compatible?
2. Print `tensor.dtype` — do they match?
3. Print `tensor.device` — are they on the same device (CPU/GPU)?
---



