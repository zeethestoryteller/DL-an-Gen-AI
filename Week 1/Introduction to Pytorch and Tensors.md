## 📝 Lecture Notes: [1.2 Introduction to Pytorch and Tensors](https://www.google.com/search?q=url%3Fid%3Dhttps://www.youtube.com/watch%3Fv%3DjG_YEftvVdE%26list%3DPLZ2ps__7DhBa9hqi20allqocTSUUt3nWX%26index%3D2)

**Course:** [IIT Madras - B.S. Degree Programme](https://www.google.com/search?q=url%3Fid%3Dhttps://www.youtube.com/watch%3Fv%3DjG_YEftvVdE%26list%3DPLZ2ps__7DhBa9hqi20allqocTSUUt3nWX%26index%3D2)

**Topic:** PyTorch Overview & Introduction to Tensors

---

### 1. What is PyTorch?

* **Definition:** PyTorch is an open-source deep learning framework based on the Torch library, primarily developed by Meta's AI Research lab (FAIR).
* **Why PyTorch?**
* **Dynamic Computation Graphs:** Unlike static frameworks, PyTorch builds and computes graphs on-the-fly (`define-by-run`), making debugging intuitive and straightforward using standard Python control flows.
* **Hardware Acceleration:** Seamlessly moves computations from CPU to GPU/TPU to speed up massive matrix operations.
* **Pythonic Nature:** Feels natural to write, closely mimicking NumPy syntax while adding powerful deep learning primitives.



---

### 2. Understanding Tensors

* **What is a Tensor?**
* A tensor is a generalization of scalars, vectors, and matrices into arbitrary dimensions ($N$-dimensional arrays).
* 0D Tensor = Scalar (single number)
* 1D Tensor = Vector (array of numbers)
* 2D Tensor = Matrix (table of numbers)
* 3D+ Tensor = Higher-dimensional tensors (e.g., batch of RGB images: `[Batch, Channels, Height, Width]`)


* **Core Difference from NumPy Arrays:** Tensors can track gradients (via `requires_grad=True`) and run on hardware accelerators (GPUs), whereas standard NumPy arrays are limited to CPU memory and lack automatic differentiation support.

---

### 3. Code Example: Creating and Inspecting Tensors in PyTorch

Let's look at how we create tensors from scratch and check their properties like shape and data type:

```python
import torch

# 1. Create a 1D tensor (Vector) from a list
tensor_1d = torch.tensor([1.5, 2.0, 3.5])
print("1D Tensor:", tensor_1d)

# 2. Create a 2D tensor (Matrix) filled with zeros
tensor_zeros = torch.zeros((2, 3))
print("\n2D Zeros Matrix:\n", tensor_zeros)

# 3. Create a tensor with random values drawn from a uniform distribution [0, 1)
tensor_rand = torch.rand((3, 3))
print("\nRandom Tensor (3x3):\n", tensor_rand)

# 4. Check tensor metadata
print("\n--- Tensor Properties ---")
print("Shape:", tensor_rand.shape)
print("Data Type:", tensor_rand.dtype)
print("Device (CPU/GPU):", tensor_rand.device)

```

---

### 4. Code Example: Basic Tensor Operations & Reshaping

Deep learning models frequently manipulate tensor shapes (e.g., flattening an image or batching data). Here is how you can perform basic operations and reshape tensors:

```python
import torch

# Create a tensor of numbers from 0 to 5
x = torch.arange(6)
print("Original 1D Tensor (x):", x)
print("Shape of x:", x.shape)

# Reshape 1D tensor into a 2D matrix (2 rows, 3 columns)
x_reshaped = x.view(2, 3)
print("\nReshaped Tensor (2x3):\n", x_reshaped)

# Element-wise operations
y = torch.tensor([[1, 2, 3], [4, 5, 6]])
z = x_reshaped + y  # Element-wise addition
print("\nElement-wise Addition (x_reshaped + y):\n", z)

# Matrix Multiplication (Dot Product)
# Transpose y to match shapes: (2, 3) @ (3, 2) -> (2, 2)
matrix_product = torch.matmul(x_reshaped, y.T)
print("\nMatrix Multiplication Result:\n", matrix_product)

```
