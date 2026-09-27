### Sigmoid
**Formula:** σ(z) = 1 / (1 + e⁻ᶻ)
#

### Tanh
**Formula:** tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)

#

### ReLU (Rectified Linear Unit)
**Formula:** ReLU(z) = max(0, z). Negatives become 0 and positives stay as they are.

# 

**Leaky ReLU**
- f(z) = z if z > 0, and f(z) = αz if z ≤ 0, where α is a small constant like 0.01.

#
**Softmax** :

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Here is a breakdown of what each part means:

* **$z_i$**: The raw output score (also called a logit) given by the neural network for a specific class.
* **$e$**: Euler's number, a mathematical constant (approximately 2.718). Raising $e$ to the power of the raw score ($e^{z_i}$) ensures that every number becomes positive, even if the raw score was negative.
* **$\sum_{j=1}^{K} e^{z_j}$**: The sum of the exponential scores for all $K$ classes. Dividing the individual score by this total sum is the "normalization" step—it guarantees that all the final probabilities add up to exactly 1.

#
