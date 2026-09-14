
---
# MODULE NOTES: Tensor Generation & Basic Tensor Operations

## 1. What is a Tensor?

* **Definition:** A container for numerical data of any dimensionality (scalars, vectors, matrices, n-dimensional tensors).
* **Key Attributes:** Every tensor has a specific shape (`shape`), data type (`dtype`), and device placement (`device`—CPU or GPU).
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

---

## 1. Why Do We Need Random Tensors?

In deep learning, neural networks start by initializing their parameters (weights and biases) with **random values** [2.2.6, 2.2.8]. If all weights started at zero or a single constant value, every neuron in a layer would compute the exact same output during the forward pass, receive the exact same gradient during backpropagation, and update identically—rendering multi-layer networks useless (symmetry problem). Random initialization breaks this symmetry.

---

## 2. Generating Random Tensors in PyTorch

PyTorch provides several built-in functions to create tensors drawn from different statistical distributions [2.2.2].

### A. Uniform Distribution (`torch.rand`)

Creates a tensor with random numbers sampled from a **continuous uniform distribution** over the interval $[0, 1)$ [2.2.1, 2.2.2].

* **Syntax:** `torch.rand(size)` or `torch.rand(size=(dim1, dim2, ...))` [2.2.6, 2.2.8]

```python
import torch

# Create a 2x3 matrix with random values between 0 and 1
random_tensor = torch.rand(size=(2, 3))
print("Random Uniform Tensor:\n", random_tensor)
print("Shape:", random_tensor.shape)

```

### B. Standard Normal Distribution (`torch.randn`)

Creates a tensor with random numbers drawn from a **standard normal (Gaussian) distribution** with a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$ [2.2.2]. This is frequently used for initializing neural network weights [2.2.6, 2.2.8].

* **Syntax:** `torch.randn(size)`

```python
# Create a 3x3 matrix with values from a normal distribution
normal_tensor = torch.randn(size=(3, 3))
print("Random Normal Tensor:\n", normal_tensor)

```

### C. Random Integers (`torch.randint`)

Generates random integers drawn from a discrete uniform distribution within a specified range [2.2.2].

* **Syntax:** `torch.randint(low, high, size)` *(Note: `low` is inclusive, `high` is exclusive).*

```python
# Create a 2x4 tensor with random integers from 0 up to (but not including) 10
int_tensor = torch.randint(low=0, high=10, size=(2, 4))
print("Random Integer Tensor:\n", int_tensor)

```

---

## 3. Reproducibility and Random Seeds (`torch.manual_seed`)

Because random generation relies on pseudo-random number generators, your values will change every time you run the script. In research and debugging, you want **reproducibility** [2.2.4]. Setting a seed ensures you get the exact same sequence of random numbers every time [2.2.2].

```python
import torch

# Set the manual seed
torch.manual_seed(42)
tensor_a = torch.rand(size=(2, 2))

torch.manual_seed(42)
tensor_b = torch.rand(size=(2, 2))

# tensor_a and tensor_b will be identical
print("Are tensors identical?", torch.equal(tensor_a, tensor_b))

```

---

## 4. Basic Tensor Operations

Once you have tensors, you can perform arithmetic and structural manipulations. Operations can generally be categorized into element-wise operations and matrix/linear algebra operations.

### A. Element-Wise Operations

Arithmetic operators (`+`, `-`, `*`, `/`) or PyTorch functions operate **element-by-element** by default (unlike strict matrix multiplication).

```python
tensor1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
tensor2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 1. Addition (Multiple ways)
print("Addition (+):\n", tensor1 + tensor2)
print("Addition (torch.add):\n", torch.add(tensor1, tensor2))

# 2. Subtraction
print("Subtraction (-):\n", tensor2 - tensor1)

# 3. Element-wise Multiplication (Hadamard Product)
print("Element-wise Multiplication (*):\n", tensor1 * tensor2)

# 4. Division
print("Division (/):\n", tensor2 / tensor1)

```

### B. In-Place Operations

By default, operations like `tensor1 + tensor2` allocate a **new region of memory** for the result tensor [2.2.4]. If you are dealing with massive deep learning models, memory efficiency matters.

* In-place operations modify the original tensor directly and are suffixed with an underscore (`_`) [2.2.3, 2.2.4].

```python
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32)

# Normal addition creates a new tensor object in memory
z = x + y 

# In-place addition: adds y to x directly without extra memory allocation
x.add_(y) 
print("Modified x in-place:\n", x)

```

### C. Matrix Multiplication (Dot Product)

To perform actual linear algebra matrix multiplication (rows columns dot product) instead of element-wise multiplication, use `torch.matmul()` or the `@` operator [2.2.1, 2.2.3].

```python
# Shape: (2, 3)
mat_a = torch.tensor([[1, 2, 3], 
                      [4, 5, 6]], dtype=torch.float32)

# Shape: (3, 2)
mat_b = torch.tensor([[7, 8], 
                      [9, 10], 
                      [11, 12]], dtype=torch.float32)

# Matrix multiplication: (2, 3) @ (3, 2) -> resulting shape: (2, 2)
result_matmul = torch.matmul(mat_a, mat_b)
# Alternatively: result_matmul = mat_a @ mat_b

print("Matrix Multiplication Result:\n", result_matmul)
print("Result Shape:", result_matmul.shape)

```

---

## 5. Quick Summary Checklist for Your Notebook

* **`torch.rand(*size)`**: Uniform distribution values on $[0, 1)$ [2.2.1, 2.2.2].
* **`torch.randn(*size)`**: Normal distribution ($\mu=0, \sigma=1$) [2.2.2].
* **`torch.randint(low, high, size)`**: Random discrete integers [2.2.2].
* **`torch.manual_seed(seed)`**: Locks the random number generator for reproducible code experiments [2.2.2, 2.2.4].
* **Element-wise vs. Matrix Product**: Use `*` for element-wise scaling [2.2.1] and `torch.matmul()` / `@` for network layer linear transformations.
* **In-place operations (`_` suffix)**: Saves memory by modifying existing tensor data blocks directly [2.2.3, 2.2.4].

---

# 📚 MODULE NOTES: Tensor Indexing and Slicing

## 1. Introduction: Why Do We Need Indexing?

When working with deep learning models, tensors can easily grow to 3D, 4D, or higher dimensions (e.g., batches of images with shape `[Batch_Size, Channels, Height, Width]`). **Indexing and slicing** allow us to inspect specific data points, extract regions of interest (like cropping an image patch), or separate training data from labels without altering the original data block.

---

## 2. Standard 1D Tensor Indexing (The Basics)

Just like Python lists, PyTorch tensors are **zero-indexed**. The first element is at index `0`, and negative indices count backward from the end (`-1` is the last element).

```python
import torch

# Create a 1D tensor
t = torch.tensor([10, 20, 30, 40, 50, 60])

print("Element at index 0:", t[0].item())   # Output: 10
print("Element at index 3:", t[3].item())   # Output: 40
print("Last element (-1):", t[-1].item())   # Output: 60

```

---

## 3. Multi-Dimensional Tensor Indexing

For 2D tensors (matrices) and above, indexing uses comma-separated values inside the brackets: `tensor[row_index, column_index]`.

```python
# Create a 3x3 2D tensor
mat = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

# Access the element at Row 1, Column 2 (number '6')
print("Element at [1, 2]:", mat[1, 2].item())

# Access an entire row (Row index 2)
print("Entire Row 2:", mat[2])

# Access an entire column (Column index 1)
# ':' means "take everything along this axis"
print("Entire Column 1:", mat[:, 1])

```

---

## 4. Slicing Sequences (`start:end:step`)

Slicing extracts a continuous subset of a tensor. The general syntax is `[start:end:step]`, keeping in mind that **`end` is exclusive** (it stops right before that index).

### A. 1D Slicing Examples

```python
x = torch.tensor([10, 20, 30, 40, 50, 60, 70])

# Slice from index 1 up to 4 (exclusive) -> elements at index 1, 2, 3
print("x[1:4]:", x[1:4])        # tensor([20, 30, 40])

# Slice from the beginning up to index 4
print("x[:4]:", x[:4])          # tensor([10, 20, 30, 40])

# Slice from index 3 to the end
print("x[3:]:", x[3:])          # tensor([40, 50, 60, 70])

# Step slicing (every second element)
print("x[::2]:", x[::2])        # tensor([10, 30, 50, 70])

```

### B. Multi-Dimensional Slicing (Sub-matrices / Cropping)

You can apply slicing across multiple dimensions simultaneously. This is heavily used in Computer Vision to crop specific bounding boxes or patches from images.

```python
# Create a 4x4 tensor
grid = torch.tensor([[1,  2,  3,  4],
                     [5,  6,  7,  8],
                     [9,  10, 11, 12],
                     [13, 14, 15, 16]])

# Extract the central 2x2 sub-matrix (Rows 1 to 2, Columns 1 to 2)
# Remember: 'end' index 3 is exclusive, so it stops at index 2.
patch = grid[1:3, 1:3]
print("Central 2x2 Patch:\n", patch)
# Output:
# tensor([[ 6,  7],
#         [10, 11]])

```

---

## 5. Advanced Indexing: Fancy Indexing & Boolean Masks

Sometimes regular sequential slices aren't enough. We need specific, non-contiguous elements.

### A. Fancy Indexing (Integer Array Indexing)

You can pass a list or tensor of specific indices to pull out non-consecutive rows or columns.

```python
t = torch.tensor([10, 20, 30, 40, 50])

# Fetch elements at indices 0, 2, and 4
indices = torch.tensor([0, 2, 4])
print("Fancy indexed elements:", t[indices])  # tensor([10, 30, 50])

```

### B. Boolean Masking (Filtering by Condition)

You can pass a conditional statement inside the brackets to filter out data dynamically. This returns a 1D tensor containing only the elements that satisfy the condition.

```python
scores = torch.tensor([[55, 80],
                       [92, 45]])

# Create a boolean mask for scores greater than 60
mask = scores > 60
print("Boolean Mask:\n", mask)

# Extract values matching the condition
filtered_values = scores[mask]
print("Scores > 60:", filtered_values)  # tensor([80, 92])

```

---

## 6. Slicing Shares Memory (Important Gotcha!)

Unlike standard Python lists where slicing creates a *copy*, **PyTorch slices share the exact same memory storage** as the original tensor for performance efficiency.

* **Warning:** If you modify a sliced tensor view, it will alter the original tensor! If you want an independent copy, use `.clone()`.

```python
original = torch.tensor([[1, 2], [3, 4]])
view_slice = original[0, :]

# Modify the slice
view_slice[0] = 99

print("Original tensor is modified too!\n", original)
# Output shows original[0,0] is now 99!

# Safe approach if you want isolation:
safe_copy = original[0, :].clone()

```

---

## 📝 Summary Quick-Reference Table

| Operation | Syntax Example | Description |
| --- | --- | --- |
| **Basic Indexing** | `t[i, j]` | Accesses a single scalar element at row `i`, col `j`. |
| **Row/Col Extraction** | `t[1, :]` or `t[:, 2]` | Extracts an entire row or column using the colon (`:`). |
| **Range Slicing** | `t[start:end]` | Extracts a subset from `start` up to `end-1`. |
| **Boolean Filtering** | `t[t > 5]` | Filters elements dynamically based on conditions. |
| **Memory Isolation** | `t.slice.clone()` | Creates an independent physical copy in memory. |

---

# 📚 MODULE NOTES: Reshaping, Squeezing, and Un-squeezing Tensors

## 1. Introduction: Why Do We Need Shape Transformations?

In deep learning, data flows through different layers that expect specific input shapes. For instance, a **Convolutional Neural Network (CNN)** typically expects a 4-dimensional tensor `[Batch_Size, Channels, Height, Width]`, while a **Linear (Dense) Layer** requires a 2-dimensional tensor `[Batch_Size, Features]`.

If your data dimensions don't match up, PyTorch will throw a shape mismatch error. That's where reshaping tools come in!

---

## 2. Reshaping Tensors (`torch.reshape` or `.view`)

Reshaping allows you to change the dimensions of a tensor without altering its underlying data values or total number of elements.

* **The Golden Rule:** The total number of elements before reshaping must equal the total number of elements after reshaping ($Rows \times Columns = Total\ Elements$).
* **Wildcard (`-1`):** You can use `-1` for one dimension to let PyTorch automatically calculate that size based on the rest of the tensor.

```python
import torch

# Create a 1D tensor with 12 elements
t = torch.arange(12)
print("Original Tensor:\n", t)
print("Original Shape:", t.shape)

# 1. Reshape into a 3x4 matrix
t_23 = torch.reshape(t, (3, 4))
# Alternatively: t.view(3, 4)
print("\nReshaped to 3x4:\n", t_23)

# 2. Reshape into a 2x2x3 3D tensor
t_3d = t.reshape(2, 2, 3)
print("\nReshaped to 2x2x3:\n", t_3d.shape)

# 3. Using the wildcard (-1)
# "Hey PyTorch, make it 4 rows, and figure out the columns automatically"
auto_reshaped = t.reshape(4, -1) 
print("\nAuto-calculated Shape using -1:", auto_reshaped.shape) # Output: torch.Size([4, 3])

```

---

## 3. Squeezing Tensors (`torch.squeeze`)

What is a **singleton dimension**? It's any dimension with a size of `1` (e.g., shape `[1, 3, 1, 4]`). Sometimes, operations or data pipelines leave behind these extra, unnecessary size-1 dimensions.

`torch.squeeze()` removes **all** dimensions of size 1 from a tensor. You can also target a specific axis by passing the `dim` argument.

```python
# Create a tensor with extra singleton dimensions of size 1
x = torch.zeros(size=(1, 3, 1, 4))
print("Original shape:", x.shape)  # torch.Size([1, 3, 1, 4])

# Squeeze out all dimensions of size 1
squeezed_x = torch.squeeze(x)
print("Squeezed shape (all 1s removed):", squeezed_x.shape)  # torch.Size([3, 4])

# Target a specific dimension
# If we only want to remove dimension at index 0 if it's 1:
specific_squeeze = torch.squeeze(x, dim=0)
print("Squeezed at dim=0 shape:", specific_squeeze.shape)  # torch.Size([3, 1, 4])

```

---

## 4. Un-squeezing Tensors (`torch.unsqueeze`)

The exact opposite of squeezing! `torch.unsqueeze()` **adds** a dimension of size `1` at a specific, designated index location.

This is extremely common when you have a single data sample (like one image or one sentence) and need to artificially add a **Batch Dimension** at index `0` so the model accepts it.

* **Syntax:** `torch.unsqueeze(tensor, dim)`

```python
# Create a simple 1D feature vector with 3 elements
feature = torch.tensor([2.5, 3.1, 1.8])
print("Original vector shape:", feature.shape)  # torch.Size([3])

# 1. Add a batch dimension at the very beginning (dim=0)
batched_feature = torch.unsqueeze(feature, dim=0)
print("Un-squeezed at dim=0 (Batch Added):\n", batched_feature)
print("New shape:", batched_feature.shape)  # torch.Size([1, 3])

# 2. Add a dimension as a column vector instead (dim=1)
column_vector = torch.unsqueeze(feature, dim=1)
print("Un-squeezed at dim=1 shape:", column_vector.shape)  # torch.Size([3, 1])

```

---

## 📝 Quick-Reference Summary Table for Your Notebook

| Function | What it Does | Common Use Case |
| --- | --- | --- |
| **`torch.reshape(t, shape)`** (or `.view()`) | Changes dimensions while keeping total element count constant. | Transitioning data from a CNN feature map into a Linear/Dense layer. |
| **`torch.squeeze(t, dim=None)`** | Removes all or specified dimensions of size `1`. | Cleaning up redundant output dimensions after pooling or calculations. |
| **`torch.unsqueeze(t, dim)`** | Inserts a new dimension of size `1` at the given index. | Adding a batch dimension (`dim=0`) to a single inference sample. |

---

# 📚 MODULE NOTES: Mathematical and Arithmetic Manipulation Methods

## 1. Introduction: Why PyTorch-Level Math Matters

When building deep learning models, we rarely write manual `for` loops to process numbers element by element in Python—it would be painfully slow. PyTorch leverages **C++ and CUDA backends** to execute mathematical operations on massive tensors concurrently (vectorization), making computations blazing fast and GPU-friendly.

---

## 2. Basic Element-Wise Arithmetic Operations

Arithmetic operators in PyTorch (`+`, `-`, `*`, `/`) work **element-wise** by default. This means every operation is applied directly to matching elements at the same position in the tensors.

Alternatively, PyTorch provides explicit functional syntax for these operations, which is useful when passing functions as arguments or working with computational graphs.

```python
import torch

# Create two tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

# 1. Addition
print("Addition (+):", a + b)                  # tensor([11., 22., 33.])
print("Using torch.add():", torch.add(a, b))

# 2. Subtraction
print("Subtraction (-):", b - a)                 # tensor([9., 18., 27.])
print("Using torch.sub():", torch.sub(b, a))

# 3. Multiplication (Element-wise / Hadamard)
print("Multiplication (*):", a * b)              # tensor([10., 40., 90.])
print("Using torch.mul():", torch.mul(a, b))

# 4. Division
print("Division (/):", b / a)                    # tensor([10., 10., 10.])
print("Using torch.div():", torch.div(b, a))

```

---

## 3. Exponents, Powers, and Logarithms

Neural networks frequently apply non-linear transformations, exponentials (like in Softmax layers), and logarithms (like in Cross-Entropy Loss calculations).

```python
x = torch.tensor([1.0, 2.0, 3.0])

# 1. Power / Exponentiation (e.g., squaring each element)
print("Squared (x^2):", torch.pow(x, 2))          # tensor([1., 4., 9.])
print("Alternative syntax:", x ** 2)

# 2. Exponential (e^x)
print("Exponential (e^x):", torch.exp(x))

# 3. Natural Logarithm (ln(x))
print("Natural Log:", torch.log(x))

```

---

## 4. Reduction Operations (Aggregations)

Reduction methods take a multi-dimensional tensor and **collapse** one or more dimensions down into a single aggregate value (like finding the total sum, mean, or extreme values).

* **Crucial Argument (`dim`):** If you don't specify a dimension, PyTorch reduces the *entire* tensor into a single scalar. If you specify a `dim`, it collapses just that specific axis.

```python
# Create a 2x3 matrix
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

# 1. Summation
print("Total Sum of all elements:", torch.sum(matrix).item())      # 21.0
print("Sum across columns (dim=0):", torch.sum(matrix, dim=0))     # tensor([5., 7., 9.])
print("Sum across rows (dim=1):", torch.sum(matrix, dim=1))        # tensor([ 6., 15.])

# 2. Mean (Average)
print("Mean of all elements:", torch.mean(matrix).item())          # 3.5

# 3. Maximum and Minimum
print("Global Maximum:", torch.max(matrix).item())                 # 6.0
print("Global Minimum:", torch.min(matrix).item())                 # 1.0

```

---

## 5. Finding Extremes with Locations (`argmax` and `argmin`)

In classification tasks, your model outputs a vector of raw scores (logits) or probabilities for different classes. To find *which* class won, you need the **index** of the highest value.

* `torch.argmax()` returns the index position of the maximum value.
* `torch.argmin()` returns the index position of the minimum value.

```python
# Model output probabilities for 4 different classes
probabilities = torch.tensor([0.05, 0.85, 0.08, 0.02])

# Find the index of the highest probability
predicted_class = torch.argmax(probabilities).item()

print("Probabilities:", probabilities)
print("Predicted Class Index (Argmax):", predicted_class)  # Output: 1 (since 0.85 is at index 1)

```

---

## 📝 Quick-Reference Summary Table for Your Notebook

| Mathematical Operation | Function / Operator | Description |
| --- | --- | --- |
| **Element-wise Math** | `+`, `-`, `*`, `/` (or `torch.add`, etc.) | Computes arithmetic operations matching position-to-position. |
| **Powers & Roots** | `torch.pow(t, n)` or `t ** n` | Raises tensor elements to a given power. |
| **Exponentials & Logs** | `torch.exp(t)`, `torch.log(t)` | Applies natural exponential or logarithmic functions element-wise. |
| **Reductions** | `torch.sum()`, `torch.mean()` | Collapses tensor dimensions to calculate totals or averages. |
| **Peak Location** | `torch.argmax(t)`, `torch.argmin(t)` | Locates the **index position** of the maximum or minimum value. |

---

# 📚 MODULE NOTES: Combining and Splitting Tensors

## 1. Introduction: Why Do We Need Tensor Combination?

In deep learning pipelines, data rarely arrives all at once. Often, you need to:

* Batch individual image samples together into a mini-batch before feeding them to a neural network.
* Combine outputs or feature maps from different layers or branches of an architecture.

To handle these scenarios, PyTorch provides powerful tools to **glue tensors together** (Concatenation and Stacking) or **tear them apart** (Splitting and Chunking).

---

## 2. Concatenation (`torch.cat`)

Concatenation joins two or more existing tensors along an **already existing dimension** (axis).

* **The Golden Rule:** All tensors being concatenated must have the exact same shape, *except* along the specific dimension you are joining them on.
* **Syntax:** `torch.cat((tensor1, tensor2, ...), dim=0)`

```python
import torch

# Create two 2x3 matrices
t1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

t2 = torch.tensor([[7, 8, 9],
                   [10, 11, 12]])

# 1. Concatenate along Rows (dim=0 / Vertically)
# Resulting shape: (4, 3) -> 2 rows + 2 rows = 4 rows
cat_dim0 = torch.cat((t1, t2), dim=0)
print("Concatenated along dim=0 (Vertical):\n", cat_dim0)

# 2. Concatenate along Columns (dim=1 / Horizontally)
# Resulting shape: (2, 6) -> 3 cols + 3 cols = 6 cols
cat_dim1 = torch.cat((t1, t2), dim=1)
print("\nConcatenated along dim=1 (Horizontal):\n", cat_dim1)

```

---

## 3. Stacking Tensors (`torch.stack`)

While concatenation glues tensors along an existing axis, **stacking creates a brand-new dimension** to join them. This is the exact tool you use to build a batch out of individual samples.

* **The Golden Rule:** All tensors must have the **exact same shape** in every single dimension.
* **Syntax:** `torch.stack((tensor1, tensor2, ...), dim=0)`

```python
# Imagine these are two separate 1D feature vectors for individual samples (length 3)
sample1 = torch.tensor([1.0, 2.0, 3.0])
sample2 = torch.tensor([4.0, 5.0, 6.0])

# Stack them together along a new batch dimension at index 0
# Shape of sample1: torch.Size([3])
# Shape after stacking: torch.Size([2, 3]) -> 2 samples in the batch!
batch_tensor = torch.stack((sample1, sample2), dim=0)

print("Stacked Batch Tensor:\n", batch_tensor)
print("New Batch Shape:", batch_tensor.shape)

```

---

## 4. Splitting Tensors (`torch.split`)

Sometimes you have a large tensor (like a mini-batch or a combined feature matrix) and you need to break it back down into smaller, individual segments.

* **Syntax:** `torch.split(tensor, split_size_or_sections, dim=0)`
* You can either specify an **equal split size** or pass a list of specific lengths for each section.

```python
# Create a 1D tensor with 6 elements
large_tensor = torch.tensor([10, 20, 30, 40, 50, 60])

# 1. Split into equal chunks of size 2
# This yields 3 smaller tensors of size 2
chunks = torch.split(large_tensor, split_size_or_sections=2, dim=0)
print("Equal splits of size 2:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:", chunk)

# 2. Split into custom uneven lengths (e.g., sections of size 1, 3, and 2)
uneven_chunks = torch.split(large_tensor, split_size_or_sections=[1, 3, 2], dim=0)
print("\nUneven splits ([1, 3, 2]):")
for i, chunk in enumerate(uneven_chunks):
    print(f"Uneven Chunk {i+1}:", chunk)

```

---

## 5. Chunking Tensors (`torch.chunk`)

`torch.chunk()` is a close cousin to `torch.split()`. The primary difference is how you specify the division:

* In `torch.split`, you specify **how many elements** each chunk should contain.
* In `torch.chunk`, you specify **how many total chunks** you want to break the tensor into.

```python
t = torch.tensor([1, 2, 3, 4, 5, 6])

# Break the tensor into exactly 3 equal chunks
# (PyTorch automatically calculates that each chunk will have 2 elements)
chunked_t = torch.chunk(t, chunks=3, dim=0)

print("\nChunking into 3 parts:")
for i, c in enumerate(chunked_t):
    print(f"Part {i+1}:", c)

```

---

## 📝 Quick-Reference Summary Table for Your Notebook

| Operation | Function | Key Behavior / Rule |
| --- | --- | --- |
| **Concatenation** | `torch.cat((t1, t2), dim)` | Joins tensors along an **existing dimension**. Tensors must match in all other dimensions. |
| **Stacking** | `torch.stack((t1, t2), dim)` | Joins tensors by **creating a brand-new dimension**. Tensors must have identical shapes. |
| **Splitting** | `torch.split(t, split_size, dim)` | Breaks a tensor apart based on **chunk size** or explicit section lengths. |
| **Chunking** | `torch.chunk(t, chunks, dim)` | Breaks a tensor apart into a **fixed total number of chunks**. |

---

# 📚 MODULE NOTES: In-Place Operations in PyTorch

## 1. Introduction: Out-of-Place vs. In-Place (The Memory Story)

By default, almost every standard mathematical operation in PyTorch (like addition, multiplication, or subtraction) is **out-of-place**.

* **Out-of-Place Operations:** When you write `z = x + y`, PyTorch allocates a **brand-new block of memory** in RAM (or VRAM) to store the result tensor `z`. The original tensors `x` and `y` remain completely untouched in their original memory locations.
* **In-Place Operations:** These operations modify the existing tensor **directly in its current memory address** without allocating any new space.

---

## 2. How to Spot an In-Place Operation (The Underscore Rule)

In PyTorch, the convention is simple and strict: **any operation that ends with an underscore (`_`) is an in-place operation**.

Let’s look at the functional equivalents:

* Out-of-place addition: `torch.add(x, y)` or `x + y`
* In-place addition: `torch.add_(x, y)` or `x.add_(y)`

---

## 3. Code Example: Seeing the Memory Difference

Let's write a small script to see how in-place operations modify data directly and preserve memory locations.

```python
import torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print("Initial memory address of x:", id(x))

# 1. OUT-OF-PLACE OPERATION (+)
# This creates a completely new tensor object in memory
y = x + 10
print("After out-of-place (x + 10):", y)
print("Memory address of y (New!):", id(y))
print("x is unchanged:", x)

# 2. IN-PLACE OPERATION (_ suffix)
# This modifies 'x' directly at its original memory location
x.add_(10)
print("\nAfter in-place (x.add_(10)):", x)
print("Memory address of x (Unchanged!):", id(x))

```

---

## 4. Common In-Place Methods in PyTorch

Almost all standard operations have an in-place counterpart:

| Standard (Out-of-Place) | In-Place Counterpart | Description |
| --- | --- | --- |
| `x + y` / `torch.add(x, y)` | `x.add_(y)` | Adds `y` to `x` in-place |
| `x * y` / `torch.mul(x, y)` | `x.mul_(y)` | Multiplies `x` by `y` in-place |
| `x - y` / `torch.sub(x, y)` | `x.sub_(y)` | Subtracts `y` from `x` in-place |
| `x / y` / `torch.div(x, y)` | `x.div_(y)` | Divides `x` by `y` in-place |
| `torch.abs(x)` | `x.abs_()` | Computes absolute values in-place |

---

## 5. Why Use In-Place Operations? (The Pros)

* **Memory Efficiency:** If you are training massive deep learning models with gigabytes of parameters, avoiding unnecessary memory allocations saves valuable GPU VRAM and speeds up garbage collection.

---

## ⚠️ 6. The Danger Zone: Why You Must Be Careful (Crucial for Backpropagation)

While in-place operations save memory, they are a **major hazard** during deep learning training loops because of **Autograd (Automatic Differentiation)**.

* **The Problem:** Neural networks rely on saving intermediate tensor values computed during the *forward pass* to calculate gradients during the *backward pass* (backpropagation).
* If an in-place operation modifies a tensor value *before* backpropagation finishes using it, PyTorch will throw a runtime error:
> *`RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.`*


* **Rule of Thumb:** Avoid using in-place operations (`+=`, `add_()`, etc.) on tensors that require gradients (like model weights or intermediate activations) unless you know exactly what you are doing. They are generally safe to use during data preprocessing, loop counters, or non-gradient tracking phases.
