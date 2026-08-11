
---
# MODULE NOTES: Random Tensor Generation & Basic Tensor Operations

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

