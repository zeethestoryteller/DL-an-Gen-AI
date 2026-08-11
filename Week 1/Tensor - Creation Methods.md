# Lecture Notes: PyTorch Tensor Creation Methods

Welcome to your study guide on PyTorch tensors! Tensors are the fundamental data structures in PyTorch used to encode the inputs, outputs, and parameters (weights/biases) of our models. Think of them as multi-dimensional arrays, very similar to NumPy arrays, but with a superpower: they can run on GPUs or other hardware accelerators for blazing-fast parallel computing.

---

## 1. What is a Tensor?

* **Definition:** A container for numerical data of any dimensionality (scalars, vectors, matrices, n-dimensional tensors).
* **Key Attributes:** Every tensor has a specific shape (`shape`), data type (`dtype`), and device placement (`device`—CPU or GPU).

---

## 2. Core Tensor Creation Methods

Here are the primary ways you will initialize tensors when building deep learning models.

### A. From Existing Python Data (`torch.tensor` vs `torch.as_tensor`)

You can convert lists, tuples, or nested collections directly into tensors.

```python
import torch

# 1D Tensor (Vector)
vector_tensor = torch.tensor([1, 2, 3])

# 2D Tensor (Matrix)
matrix_tensor = torch.tensor([[1, 2], [3, 4]])

```

> **Teaching Tip:** `torch.tensor()` always *copies* the data. If you want to avoid unnecessary memory copies when working with existing NumPy arrays, look into `torch.from_numpy()` or `torch.as_tensor()`.

---

### B. Initializing with Pre-defined Shapes (Zeros, Ones, and Constants)

Often, you need to set up structural placeholders or weights initialized to a constant value before training begins.

```python
# Create a tensor filled entirely with zeros (useful for biases or padding masks)
zeros_tensor = torch.zeros((2, 3))  # Shape: 2 rows, 3 columns

# Create a tensor filled entirely with ones
ones_tensor = torch.ones((3, 3))

# Create a tensor filled with a specific scalar value (e.g., filling with 7)
full_tensor = torch.full((2, 2), fill_value=7.0)

```

---

### C. Identity and Sequence Tensors

Useful for indexing, linear algebra operations, or creating sequential time steps.

```python
# Identity Matrix (Square matrix with ones on the main diagonal)
identity_matrix = torch.eye(3)  # 3x3 identity matrix

# Sequence of numbers (similar to Python's range or NumPy's arange)
seq_tensor = torch.arange(start=0, end=10, step=2)  # tensor([0, 2, 4, 6, 8])

# Evenly spaced numbers over a specified interval (great for plotting curves/activation inputs)
lin_tensor = torch.linspace(start=0, end=1, steps=5)  # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

```

---

### D. Random Tensor Generation (Crucial for Weight Initialization)

Neural network weights must be randomized initially to break symmetry during training.

```python
# Uniform distribution between [0, 1)
rand_tensor = torch.rand((2, 3))

# Normal (Gaussian) distribution with mean=0 and variance=1 (Standard Normal)
randn_tensor = torch.randn((2, 3))

# Random integers within a specific range [low, high)
randint_tensor = torch.randint(low=0, high=10, size=(3, 3))

```

---

### E. "Like" Methods (Copying Shape & Properties)

If you have an existing tensor and want to create a new one with the **same shape** (and optionally the same data type or device) without manually typing out dimensions:

```python
base_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Creates a zero-filled tensor with the exact same shape and dtype as base_tensor
zeros_like_tensor = torch.zeros_like(base_tensor)

# Creates a randomly populated tensor matching the shape of base_tensor
rand_like_tensor = torch.rand_like(base_tensor)

```

---

## 3. Explicit Data Type (`dtype`) Specification

By default, standard integer lists convert to 64-bit integers (`torch.int64`) and floating-point lists convert to 32-bit floats (`torch.float32` / `torch.float`). You can override this explicitly during creation to manage memory footprint:

```python
# Creating a double-precision (64-bit float) tensor
double_tensor = torch.tensor([1.5, 2.5], dtype=torch.float64)

# Creating a boolean mask tensor
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)

```

---

## Quick Summary Checklist for Your Notebook:

1. **`torch.tensor()`**: General creation from data (copies memory).
2. **`torch.zeros()` / `torch.ones()` / `torch.full()**`: Constant structural fillers.
3. **`torch.arange()` / `torch.linspace()**`: Sequence generation.
4. **`torch.rand()` / `torch.randn()**`: Stochastic distributions for model weights.
5. **`*_like()` variants**: Shape-preserving shortcuts based on existing tensors.
