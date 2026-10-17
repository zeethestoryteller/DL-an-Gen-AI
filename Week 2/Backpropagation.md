# **Backpropagation's Engine: The Chain Rule**

#### **The Question**

* How does a small change in an early parameter ($`w_1`$) affect the final loss ($L$) at the end of a long chain of computations?

#### **The Core Concept**

* The **Chain Rule** provides the answer by multiplying local derivatives along the path.

#### **Mathematical Formulation**

* Imagine a function $L(y(x))$. We want to find how $L$ changes with respect to $x$, which is $\frac{dL}{dx}$.
* **Formula:**

$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$$



> **Figure Note:** The influence of $x$ on $L$ is the product of the local influences along the path ($x \xrightarrow{\frac{dy}{dx}} y(x) \xrightarrow{\frac{dL}{dy}} L(y)$).
