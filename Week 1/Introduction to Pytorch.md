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


**Topic:** PyTorch Installation, Virtual Environments, and Checking GPU Availability

---

### 1. Setting Up a Virtual Environment

* **Why use a virtual environment?**
Think of a virtual environment as an isolated sandbox for your project. If Project A needs PyTorch version 2.0 and Project B needs version 2.4, keeping them in separate virtual environments ensures their packages don't crash or conflict with each other.
* **How to create and activate one using Conda or venv:**
```bash
# Option A: Using Python's built-in venv
python -m venv my_dl_env

# Activate it (Mac/Linux):
source my_dl_env/bin/activate

# Activate it (Windows Command Prompt):
# my_dl_env\Scripts\activate

```



---

### 2. Installing PyTorch

* PyTorch makes installation super easy. Depending on whether you have an NVIDIA GPU, an Apple Silicon Mac, or just a standard CPU, head over to the official PyTorch website to grab your specific install command.
* **General pip installation command (for CPU / standard setup):**
```bash
pip install torch torchvision torchaudio

```



---

### 3. Code Example: Verifying Installation & Checking GPU Availability

Once everything is installed, the very first thing we always want to do in a notebook or script is check if PyTorch is ready to roll and whether it can detect our GPU. Using a GPU (via CUDA or Apple's MPS) makes training deep neural networks lightning fast!

Let's write a quick script to check this:

```python
import torch

# 1. Check PyTorch Version
print(f"PyTorch Version: {torch.__version__}")

# 2. Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

if cuda_available:
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Found GPU: {gpu_name}")
    
    # Check how many GPUs are accessible
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("No CUDA-compatible GPU detected. Running on CPU.")

# 3. Best Practice: Setting up a device-agnostic code snippet
# This automatically routes your tensors to the GPU if available, else falls back to CPU!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device for computations: {device}")

# Let's test allocating a tensor on our chosen device
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Test tensor successfully placed on: {x.device}")

```

---

### 💡 Pro-Tip :

Whenever you write code in PyTorch, try to define a `device` variable early on (like we did above). When you build large neural networks or load datasets, you can push them to `device` dynamically. That way, your code will seamlessly run on your laptop's CPU during testing and fly on a cloud GPU when you train heavy models!

---



