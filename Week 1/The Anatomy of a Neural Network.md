# The Anatomy of a Neural Network


### From Biology to Mathematics 
This section tells one story: how a rough idea from biology turned into the mathematical neuron used in every modern neural network. It goes in four steps.

**Brain neuron → McCulloch-Pitts (1943) → Perceptron (1958) → Modern neuron**
#

### 1. Inspiration from the brain

A biological neuron is a brain cell, and the slides simplify it into four parts:

- **Dendrites** receive signals from other neurons.
- **Soma (cell body)** adds up all those signals.
- If the total reaches a **threshold**, the neuron **"fires"**.
- The **axon** sends the output signal onward.

So the pattern is: collect signals, add them up, fire if the total is big enough, pass it on.

The slides also give a warning. This is an **inspiration, not a replica**:
- The real brain is far more complex.
- Biological learning is **not** backpropagation.
- The neuron model is a useful simplification, not an exact copy.


#
### 2. The McCulloch-Pitts neuron (1943)

This was the **first mathematical model of a neuron**. It is a simple logic gate.

**How it works:**
1. It takes several **binary inputs** (each is 0 or 1).
2. It **sums** them.
3. If the sum is **≥ a fixed threshold θ**, the output is **1**. Otherwise it is **0**.

**Example (my own, not from the slides):** with two inputs:
- If θ = 2, the neuron fires only when both inputs are 1. That is an **AND** gate.
- If θ = 1, it fires when at least one input is 1. That is an **OR** gate.

**Why it mattered:** networks of these simple units can compute **any logical function** (AND, OR, and so on).

### Its limitations
The slides call it "a calculator, not a learner." It can compute but cannot learn, for three reasons:
- **No feature importance:** every input counts equally, so it can't say "this input matters more." What's missing is **weights**.
- **No automatic learning:** the threshold θ is fixed and must be set by hand. What's missing is a **learning rule**.
- **Binary only:** inputs and outputs must be 0 or 1, so it can't handle real-valued data like height or temperature.


#
### 3. The Perceptron (1958)

The perceptron fixed the M-P neuron's problems with two innovations.

**Innovation 1: Weighted inputs.** Each input is multiplied by a **weight** (w₁, w₂, …, wₙ). A bigger weight means that input matters more, so the model can prioritize important signals.

**Innovation 2: A learning rule.**
1. Compare the prediction ŷ with the true label y.
2. If it is wrong, **adjust the weights** to reduce the error.
3. Repeat over the data.

**Why it matters:** for the first time, a machine could **learn to classify patterns on its own**. The slides call this the foundation of all modern deep learning.

### Strengths
- **It learns:** weights are adjusted automatically from data.
- **It learns feature importance** through the weights.
- **Convergence guarantee:** if the data is linearly separable, it is guaranteed to find a solution.

### Limitations
- **Linear separability only.** It can only solve problems where **one straight line** (or a flat plane in higher dimensions) separates the classes. This is the "fatal flaw." Later in this module you'll meet the **XOR problem**, which is exactly the kind of problem it cannot solve.
- **Harsh threshold (step function).** The output jumps from 0 to 1 with no smooth transition, so it is **not differentiable**. That blocks modern **gradient-based training**.


#
### 4. The modern neuron: the engine of deep learning

The modern neuron keeps the good ideas (weights, plus a learning rule that uses gradients) and fixes the step-function problem. It works in two steps:

**Step 1: Linear combination**
> **z = w · x + b**

That is, z = w₁x₁ + w₂x₂ + … + wₙxₙ + b.
- **w** are the weights (importance of each input).
- **b** is the **bias**, an extra adjustable number that shifts the result up or down.

**Step 2: Non-linear activation**
> **a = g(z)**

Here **g** is an **activation function** that is smooth and **differentiable**. It replaces the harsh step function.

**Example (my own):** x = [2, 3], w = [0.5, −1], b = 1
z = (0.5×2) + (−1×3) + 1 = 1 − 3 + 1 = **−1**
Then a = g(−1), whichever activation function g you choose.

### Why differentiability is the "superpower"
- Differentiable means you can compute a **gradient**, which tells you which direction to nudge each weight to reduce the error.
- Gradients are what **backpropagation** uses to train the network.
- A step function has no useful gradient, so it can't be trained this way. A smooth activation can.

#

### Quick summary

| | Inputs | Weights | Learns? | Output |
|---|---|---|---|---|
| **Biological neuron** | signals | — | (not modeled) | fires or not |
| **McCulloch-Pitts (1943)** | binary | none | No (fixed threshold) | 0 or 1 |
| **Perceptron (1958)** | real numbers | learnable | Yes (learning rule) | 0 or 1 (step) |
| **Modern neuron** | real numbers | learnable + bias | Yes (gradients) | smooth value a = g(z) |

**One sentence to remember:** every neuron in a deep network does the same two steps, **weighted sum plus bias, then a smooth non-linear activation**, and it is smooth so that gradients can flow and the network can learn.

---
### Fundamental Tricks in Deep Learning 
Its main idea is that deep learning takes a real-world problem and turns it into a math problem a computer can solve. It does this with a chain of "tricks":

**Turn everything into numbers → Treat the solution as a function → Give the function tunable dials → Measure the error and find the best dials**

#

## Trick 1: The Numerization Trick

### The problem
Humans experience the world through **sensations** such as sight, sound and touch. Computers only understand **numbers**. So we must convert every piece of information about a problem into a **list of numbers, called a vector**.

### Example: a self-driving car
The car must see the road, recognize objects and decide what to do in real time. Everything it sees and does becomes a vector.

**Inputs (what the car "sees"):**
- **Camera data:** a huge vector of pixel values (R, G, B for every pixel).
- **LiDAR data:** 3D coordinates (x, y, z) of nearby surfaces.
- **Radar data:** distances and velocities of objects.
- **Car's state:** current speed, acceleration, GPS location.

**Outputs (what the car "does"):**
- **Control actions:** a vector like `[Steering Angle, Acceleration, Brake]`.
- **Object predictions:** for each detected object, a vector like `[Class, X-Pos, Y-Pos, Velocity]`.

### How do we turn categories into numbers?
Most of the car's data is already numeric. But a category like "Pedestrian" or "Stop Sign" is a word, not a number. The slides give two methods.

**Method 1: One-Hot Encoding**
- Each class becomes a binary vector with a single "hot" (1) position and 0 everywhere else.
- Pedestrian → `[1, 0, 0]`, Vehicle → `[0, 1, 0]`, Stop Sign → `[0, 0, 1]`.
- It is simple, but **inefficient for many classes**. With 50,000 classes you would need vectors of length 50,000, almost all zeros. (This is my own example.)

**Method 2: Embedding Layers**
- The network **learns** a short, dense vector for each class, such as Stop Sign → `[0.9, -0.2, ...]`.
- **Similar concepts get similar vectors.**
- It is **efficient** and captures **meaning and relationships** between classes.

#

## Trick 2: The "Learn the Function" Trick

Once everything is numbers, the problem takes this shape:

1. The **problem** is data: inputs and outputs.
2. The **solution** is a **function** that maps inputs to outputs.
3. The **learning task** is to **find this function**.

A useful example from the slides: **ChatGPT is one massive function.** Your prompt goes in and the reply comes out.

### Every function has two parts

**The Form (Architecture):**
- This is the function's template or structure.
- Learning a function with a fixed structure is much easier ("more tractable") than learning a completely arbitrary function, because you narrow the search to one shape.

**The Parameters (w):**
- These are the "dials" that tune the function.
- They are learned automatically from data.

**Example (my own):** for the function y = w·x + b, the form is "a straight line". The parameters are the numbers w and b that decide which straight line it is.

#

## Trick 3: The Parameterization Trick

This is the same idea written formally. We define the model as a function with **learnable parameters** (weights and biases), called **θ (theta)**. Training means finding the best values of θ.

> **ŷ = f(x ; θ)**

| Symbol | Meaning |
|---|---|
| **x** | the numerical input data |
| **f** | the network architecture (layers, activation functions, etc.) |
| **θ** | all the weights and biases in the network |
| **ŷ** | the model's prediction |

The semicolon in f(x ; θ) separates the input x from the parameters θ.

#

## Trick 4: From Learning to Optimization (The Loss Landscape)

Parameterization turns the vague idea of "learning" into a clear problem: **optimization**, which means finding the best θ.

### Step 1: Measure the error
A **loss function** gives a score for how bad the model is:

> **L(y, ŷ) = L(y, f(x ; θ)) ≈ L(θ)**

- **High loss = bad model. Low loss = good model.**
- The data is fixed, so the loss depends only on θ. That is why it is written as L(θ).
- **Example (my own):** if the true answer is y = 10 and the model predicts ŷ = 7, one simple loss is (10 − 7)² = 9. A better prediction gives a smaller number.

### Step 2: The goal
Find the parameters **θ\*** that give the **lowest possible loss**.

### The Loss Landscape
Imagine plotting the loss for every possible θ. You get a landscape:
- **Mountains** are high error (bad parameters).
- **Valleys** are low error (good parameters).

It is "high-dimensional" because real networks have millions or billions of parameters. Training is like walking downhill to find a deep valley. (The next part of the module, **Gradient Descent**, explains how.)

#

## Big picture

| Step | Trick | What it does |
|---|---|---|
| 1 | **Numerization** | Converts the real world into vectors of numbers (one-hot or embeddings for categories) |
| 2 | **Learn the Function** | Treats the solution as a function with a form and parameters |
| 3 | **Parameterization** | Writes it as ŷ = f(x ; θ), with θ learnable |
| 4 | **Loss / Optimization** | Measures error with L(θ) and searches for the θ\* with the lowest loss |

**One sentence to remember:** turn everything into numbers, model the solution as a function with adjustable dials (θ), score its mistakes with a loss, and then hunt for the dial settings that make the loss smallest.

---
### Network ArchitecturesThis 
Earlier you learned what a single neuron does. Here we connect many neurons into a network and see how a whole layer is computed.

#

## 1. Building a feedforward network

A feedforward network is built from three kinds of layers.

**1. Input layer**
- It receives the data and **just passes the values along**. No calculation happens here.

**2. Hidden layers**
- These are the network's **computational engine**.
- They learn **increasingly abstract features**. For an image, early layers might pick up simple things like edges and later layers more complex shapes. (This example is mine, not from the slides.)

**3. Output layer**
- It produces the **final result**, such as a prediction.

### Two key properties
- **Feedforward flow:** information moves in **one direction only, from input to output. No loops.**
- **Fully connected:** **every neuron connects to every neuron in the next layer.**

#

## 2. What each neuron calculates

Every neuron does the same two steps you saw in the last section:

1. **Linear part:** z = (Σ wᵢxᵢ) + b, a weighted sum of the inputs plus a bias.
2. **Non-linear part:** a = g(z), which passes z through an activation function g.

The slides call this "Linear Sum → Non-Linear Activation" the **fundamental building block**, and **every single neuron** in a deep network performs it.

#

## 3. Linear model vs neural network

- **Linear model:** ŷ = w₀x₀ + w₁x₁ = Σ wᵢxᵢ. This is just a weighted sum of the input features. There is no non-linear step and no hidden layer.
- **Neural network:** each neuron has a **linear and a non-linear operation**, a = g(Σ wᵢxᵢ). These neurons are arranged in input, hidden and output layers, and each layer's output feeds the next layer.

The non-linear step is what gives the network more power than a plain linear model. The next section covers this in detail.

#

## 4. The forward pass: computing a whole layer at once

### The problem
Calculating neuron by neuron is very slow.

### The solution
Compute an **entire layer at once** using **vectorized operations**, meaning **matrix multiplication**.

### The terms for a layer *l*

| Symbol | Meaning |
|---|---|
| **a[l−1]** | vector of activations coming from the previous layer |
| **W[l]** | weight matrix. Wᵢⱼ is the weight from neuron *j* in the previous layer to neuron *i* in the current layer |
| **b[l]** | bias vector for the current layer |
| **z[l]** | vector of weighted sums for the current layer |
| **a[l]** | vector of activations for the current layer |

### The two core equations
1. **Weighted sum:** z[l] = W[l] · a[l−1] + b[l]
2. **Activation:** a[l] = g(z[l])

This is efficient and is **perfectly suited for GPUs**, which are built to do matrix math in parallel.

**Shape tip (my own):** W[l] has one row per neuron in the current layer and one column per neuron in the previous layer. For 2 inputs going into 3 neurons, W is 3×2.

#

## 5. Worked example (from the slides)

A hidden layer with **3 neurons** receives input from **2 neurons**, and the activation is **ReLU: g(z) = max(0, z)**.

**Step 1: Define everything**
- Inputs: x = [0.5, 1.0]
- Weights (3×2):
  ```
  W = [ 0.2   0.7 ]
      [-0.4   0.1 ]
      [ 0.9  -0.3 ]
  ```
- Biases: b = [0.1, 0.2, −0.5]

**Step 2: Weighted sum z = Wx + b**
- Neuron 1: (0.2×0.5) + (0.7×1.0) + 0.1 = **0.9**
- Neuron 2: (−0.4×0.5) + (0.1×1.0) + 0.2 = **0.1**
- Neuron 3: (0.9×0.5) + (−0.3×1.0) − 0.5 = **−0.35**

**Step 3: Apply ReLU to each value**
- max(0, 0.9) = **0.9**
- max(0, 0.1) = **0.1**
- max(0, −0.35) = **0**. ReLU turns negatives into 0.

**Result:** a = [0.9, 0.1, 0]. This vector becomes the input to the next layer.

#

## 6. The Multilayer Perceptron (MLP)

### What is an MLP?
Simply **a feedforward network with one or more hidden layers.**

### Why it overcomes the perceptron's limits
Recall that a single perceptron **fails on non-linear data**, such as XOR. The hidden layer fixes this:

- It acts as an **automatic feature engineer**.
- It **transforms the data into a new representation** that it learns by itself.
- In that new space, the data becomes **linearly separable**, so a straight line can now split it.
- This lets the MLP build **complex decision boundaries**.

The slide's picture shows this: in the **original space** the two classes can't be separated by one line, but after the **hidden layer** ("learned representation") they can be.

**Example (my own):** XOR outputs 1 when the two inputs differ. The points (0,0) and (1,1) belong to class 0, and (0,1) and (1,0) belong to class 1. No single straight line separates them. A hidden layer can reshape the points so one line does work. The module comes back to this in "The XOR Problem".

#

## Summary

| Idea | Key point |
|---|---|
| **Layers** | Input (passes data), hidden (learns features), output (gives the result) |
| **Feedforward** | One direction only, no loops |
| **Fully connected** | Every neuron links to every neuron in the next layer |
| **Neuron** | Linear sum, then non-linear activation |
| **Vectorized layer** | z = Wa + b, then a = g(z), fast on GPUs |
| **MLP** | Feedforward network with hidden layers that learn new representations, solving problems a single perceptron can't |

**One sentence to remember:** a neural network is layers of neurons where each layer computes z = Wa + b, then a = g(z), and the hidden layers reshape the data until the problem becomes easy.

---
### Activation Functions: Introducing Non-Linearity
"The Anatomy of a Neural Network". It answers two questions: why do neurons need an activation function at all, and which one should you use?

#

## 1. Why do we need activation functions?

### The problem: stacking linear layers is useless
Without a non-linear activation, a deep network **collapses into a single linear model**, no matter how many layers it has.

**Without non-linearity:**
- One layer is a linear operation: Wx + b.
- Stack two layers: L₂(L₁(x)) = W₂(W₁x + b₁) + b₂.
- Simplify: **(W₂W₁)x + (W₂b₁ + b₂)**.
- That is just another linear function, W′x + b′.

So **a 100-layer linear network has the same power as a 1-layer network.** The extra layers add nothing.

**With non-linearity (g):**
- The equation becomes g(W₂ · g(W₁x + b₁) + b₂).
- This **cannot be simplified**.
- So the network can learn arbitrarily complex, "wiggly" functions.

**In one line:** the activation function is what makes depth worthwhile.

#

## 2. Conventional activation functions

### Sigmoid
**Formula:** σ(z) = 1 / (1 + e⁻ᶻ)

**Properties:**
- It squeezes any real number into the range **(0, 1)**.
- It was historically popular because it can be read as a neuron's "firing rate".
- It is still useful in **output layers for binary classification**, where it gives a probability.

**Problems:**
- **Vanishing gradients:** the curve is flat at both ends. For large positive or negative inputs the gradient is nearly zero, which effectively stops learning in deep networks.
- **Not zero-centered:** outputs are always positive, which can slow down learning.

### Tanh
**Formula:** tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)

**Properties:**
- It squeezes values into the range **(−1, 1)**.
- It is **zero-centered**, its main advantage over sigmoid. That helps center the data for the next layer and often speeds up convergence.

**Problem:**
- **Vanishing gradients**, just like sigmoid. It flattens at both ends, so it is hard to use in very deep networks.

### ReLU (Rectified Linear Unit)
**Formula:** ReLU(z) = max(0, z). Negatives become 0 and positives stay as they are.

**Properties:**
- **Computationally efficient:** it is just a threshold.
- **No vanishing gradient for z > 0:** the gradient is a constant 1 for positive inputs, so learning signals can travel deep into the network.
- **Sparsity:** it outputs 0 for negative inputs, which can make the network sparse (many neurons are inactive at once).

**Problem: the "Dying ReLU":**
- Suppose a neuron's weights get updated so that z is **always negative**.
- Then it always outputs 0, and its gradient is also 0.
- It can never recover, so it effectively "dies".

#

## 3. The ReLU family: fixing the dying problem

**Leaky ReLU**
- f(z) = z if z > 0, and f(z) = αz if z ≤ 0, where α is a small constant like 0.01.
- **The fix:** negative inputs still get a small non-zero gradient, so "dead" neurons can be revived.
- It is a very common and effective choice.

**GELU (Gaussian Error Linear Unit)**
- A smoother, probabilistic alternative to ReLU.
- It became the standard in state-of-the-art **Transformer** models such as **GPT and BERT**.

**The gallery (slide 47).** This slide shows plots of 15 activation functions side by side: ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, SELU, GELU, Swish, Softplus, Hardtanh, Tanhshrink, Softsign, LogSigmoid, Hardshrink and Mish. The point is that many variants exist and most are small twists on the same ideas: squash into a range, or stay linear for positive values with a tweak for negative ones.

#

## 4. Practical guide: which one to use?

### Hidden layers
1. **Start with ReLU.** It is the most common and fastest, and usually works.
2. If you see **dying neurons**, switch to **Leaky ReLU** or **Parametric ReLU** (where the small slope is learned).
3. For **Transformer-based models**, consider **GELU or Swish**.

### Output layer (depends on the task)

| Task | Activation | Why |
|---|---|---|
| **Binary classification** | **Sigmoid** | Outputs a probability between 0 and 1 |
| **Multiclass classification** | **Softmax** | Outputs a probability distribution over all classes, and all outputs sum to 1 |
| **Regression** | **None (linear)** | The output can be any real number |

### Rule of thumb
**Never use sigmoid or tanh in the hidden layers of modern deep networks**, because of the vanishing gradient problem. Today they are used almost exclusively in output layers.

#

## Summary

| Activation | Range | Good | Bad |
|---|---|---|---|
| **Sigmoid** | (0, 1) | Probabilities, binary output | Vanishing gradients, not zero-centered |
| **Tanh** | (−1, 1) | Zero-centered | Vanishing gradients |
| **ReLU** | [0, ∞) | Fast, no vanishing gradient for z > 0, sparse | Dying ReLU |
| **Leaky ReLU** | (−∞, ∞) | Fixes dying ReLU | (none listed) |
| **GELU / Swish** | ≈ smooth ReLU | Standard in Transformers | — |

**One sentence to remember:** without activation functions, a deep network is secretly just one linear layer, so use **ReLU in hidden layers** (or a variant) and choose the output activation by the task: **sigmoid, softmax or none**.

