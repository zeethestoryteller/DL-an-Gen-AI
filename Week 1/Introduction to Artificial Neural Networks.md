## 📝 Lecture Notes: 1.1 Introduction to Artificial Neural Networks

**Course:** [IIT Madras - B.S. Degree Programme](https://www.youtube.com/watch?v=Z7t12UH0cyY&list=PLZ2ps__7DhBa9hqi20allqocTSUUt3nWX&index=1)

**Topic:** Introduction to Machine Learning, Terminologies, Limitations, and Artificial Neural Networks (ANNs)

---

### 1. Overview of Machine Learning (ML)

* **Definition:** Machine learning is a subset of artificial intelligence (AI) focused on building systems that learn from data rather than being explicitly programmed for every rule.
* **Key Terminologies:**
* **Features ($x$):** The input variables, attributes, or characteristics fed into a model.
* **Labels/Targets ($y$):** The desired output or ground truth we want the model to predict.
* **Model:** A mathematical function or representation parameterized by weights and biases that maps inputs to outputs.
* **Training:** The process of optimizing the model's parameters using data so that its predictions closely match the true labels.



---

### 2. Limitations of Traditional Machine Learning

* **Feature Engineering Bottleneck:** Traditional ML algorithms (like Linear/Logistic Regression, Support Vector Machines) require manual domain expertise to extract meaningful features from raw data (e.g., pixels, audio waveforms, or raw text).
* **Scaling with High-Dimensional Data:** As data complexity increases (such as images, video, or natural language), manual feature engineering becomes extremely difficult and sub-optimal.
* **Representation Learning:** Deep Learning and Artificial Neural Networks solve this limitation by automatically discovering useful representations and hierarchical features straight from raw data.

---

### 3. Introduction to Artificial Neural Networks (ANN)

* **Biological Inspiration:** ANNs are loosely modeled after biological neural networks in the human brain, where interconnected neurons receive signals, process them, and pass activations downstream.
* **The Artificial Neuron (Perceptron Building Block):**
* Receives multiple inputs: $x_1, x_2, \dots, x_n$
* Each input is multiplied by a corresponding weight: $w_1, w_2, \dots, w_n$
* A bias term ($b$) is added to shift the decision boundary.
* **Linear Combination:**

$$z = \sum_{i=1}^{n} w_i x_i + b = W^T X + b$$


* **Activation Function ($f$):** Applied to $z$ to introduce non-linearity:

$$\hat{y} = f(z)$$





---

### 4. Basic Code Example: Simple Neuron Computation (Python / NumPy)

```python
import numpy as np

# Define inputs and corresponding weights for a single neuron
inputs = np.array([0.5, 1.2, -0.3])
weights = np.array([0.2, -0.4, 0.5])
bias = 0.1

# 1. Compute the linear combination (weighted sum + bias)
linear_output = np.dot(weights, inputs) + bias
print(f"Linear Output (z): {linear_output:.4f}")

# 2. Apply a non-linear activation function (e.g., Sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

activation_output = sigmoid(linear_output)
print(f"Neuron Activation (y_hat): {activation_output:.4f}")

```
