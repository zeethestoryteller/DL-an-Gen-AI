# Index
### 1 [ Universal Approximation Theorem](https://github.com/zeethestoryteller/DL-an-Gen-AI/blob/main/Week%202/The%20Anatomy%20of%20a%20Neural%20Network.md#%EF%B8%8F-universal-approximation-theorem)
### 2 [ Forward Pass and Loss Functions]()
### 3 [Gradient Descent: The Intuition ](https://github.com/zeethestoryteller/DL-an-Gen-AI/edit/main/Week%202/The%20Anatomy%20of%20a%20Neural%20Network.md#%EF%B8%8F-gradient-descent-the-intuition)




---
# ➡️ Universal Approximation Theorem
---

## 1. The core idea

A neural network with **one hidden layer**, a **nonlinear activation**, and **enough neurons** can get as close as you like to **any continuous function** on a **bounded region**.

The transcript is right that this is a statement about **capacity** (what the network *can* represent), not about whether training will find it.

## 2. Why linearity isn't enough

Linear regression gives ŷ = w₁x₁ + w₂x₂ + b, which is a line or hyperplane.

If you stack linear layers, you get another linear function:

W₂(W₁x) = (W₂W₁)x

Ten layers collapse into one matrix. Depth adds nothing without nonlinearity. A nonlinear activation like sigmoid, tanh, or ReLU stops this collapse and lets the network bend the input space.

## 3. The formal statement

A one-hidden-layer network computes:

**G(x) = Σᵢ₌₁ᴺ vᵢ · σ(wᵢx + bᵢ)**

- N is the number of hidden neurons
- wᵢ, bᵢ are the weights and bias of hidden neuron i
- vᵢ is the output weight of that neuron
- σ is the activation function

**Theorem:** For any continuous function f on a compact (closed and bounded) domain, and any ε > 0, there exist N, vᵢ, wᵢ, bᵢ such that

**|G(x) − f(x)| < ε for all x in the domain.**

"Arbitrarily close" means you pick the error tolerance, and some network meets it. Smaller ε usually needs a larger N.

## 4. Why it's true: the construction in 1D

This is the "bumps" idea from the video, step by step.

**Step 1: a sigmoid is a soft step.**
σ(wx + b) goes from 0 to 1 around the point x = −b/w. Making w very large makes it nearly a hard step function. The bias b moves where the step happens.

**Step 2: two steps make a bump (a "tower").**
Take a step up at x = a and subtract a step up at x = c (where c > a):

σ(w(x − a)) − σ(w(x − c))

This is about 1 between a and c and about 0 elsewhere. It's a rectangular bump, and it uses 2 neurons.

**Step 3: scale the bump.**
Multiply by the output weight vᵢ to set the bump's height to any value you want.

**Step 4: tile the domain.**
Split the domain into many thin slices. In each slice, place a bump whose height equals f's value there. Add them all up:

G(x) = Σ (height of f at slice k) × (bump k)

The result is a staircase that hugs f. Because f is continuous, it barely changes across a thin slice, so the staircase error is small. More slices means thinner steps and smaller error, which means more neurons.

**Tiny example:** approximate f(x) = x² on [0, 1] with 4 slices. Bumps at heights 0.03, 0.19, 0.53, 0.97 (f at slice midpoints) give a 4-step staircase. With 100 slices it's visually indistinguishable from the parabola.

**ReLU version:** a ReLU network builds *piecewise linear* functions instead. Each neuron adds a "kink," and connecting kinks gives a polyline that approximates any curve.

## 5. Extending beyond 1D

In d dimensions, each neuron σ(w·x + b) is a ridge: it's constant along directions perpendicular to w. Combinations of ridges can form localized bumps in higher dimensions, and the same tiling argument works. The catch is that the number of neurons needed can grow **exponentially** with dimension (the curse of dimensionality).

## 6. What the theorem requires and what it doesn't say

**Conditions**
- **Continuous target f** (extensions exist for discontinuous and measurable functions, in a weaker sense)
- **Compact domain.** Outside it, no guarantee. A network trained on [0,1] can behave arbitrarily on [5, 10].
- **Non-polynomial activation.** Cybenko (1989) proved it for sigmoids, and Hornik (1991) and Leshno et al. (1993) showed it holds for essentially any non-polynomial activation, including ReLU. A polynomial activation fails because the network can only produce polynomials up to a fixed degree.

**What it does NOT guarantee**

| Question | UAT answer |
|---|---|
| Can the network *represent* an approximation? | Yes (existence) |
| How many neurons are needed? | Not stated (could be astronomically many) |
| Can gradient descent *find* the weights? | No guarantee (trainability) |
| Will it work on unseen data? | No guarantee (generalization) |
| Is it efficient? | No |

So there are three separate issues: **representation** (UAT), **optimization** (can we find the weights?), and **generalization** (does it work on new data?). Even a perfect fit on training data can overfit.

## 7. If one layer is enough, why use deep networks?

A single hidden layer may need exponentially many neurons for some functions, while a deeper network can represent them with far fewer. Depth lets the network reuse and compose features (edges → shapes → objects), which is much more parameter-efficient. There are also "depth" versions of the theorem (Lu et al., 2017) showing narrow-but-deep networks are universal approximators too.

## 8. Summary

1. Linear models can only draw lines and planes.
2. Nonlinear neurons create bumps or kinks.
3. Weighted sums of many bumps can tile any continuous curve.
4. With enough neurons, the error can be made as small as you like.
5. This says the answer *exists*, not that training will find it or that it will generalize.

## Check your understanding

1. Why does stacking 5 linear layers not help?
2. How many sigmoid neurons make one bump in 1D?
3. If you want half the error, roughly what must you do to the network?
4. Your network fits the training data perfectly but fails on test data. Does that contradict the UAT?
5. Why can't a network with a polynomial activation be universal?

*(Answers: 1. They collapse to one linear map. 2. Two. 3. Add more neurons and thinner bumps. 4. No, the UAT says nothing about generalization. 5. It only produces polynomials up to a fixed degree, so it can't approximate, say, sin(x) arbitrarily well over a wide range.)*

If you'd like, I can build an interactive demo where you add neurons and watch a curve get approximated, or walk through a numerical example with actual weights.

---
---
# ➡️ Forward Pass and Loss Functions

## 1. The big picture

Training a neural network is a loop:

1. **Forward pass:** feed input x through the network and get a prediction ŷ.
2. **Loss:** compare ŷ with the true answer y and get one number saying how wrong it was.
3. **Backward pass:** work out how to change the weights to reduce that number.
4. **Update:** nudge the weights, then repeat.

Your question covers steps 1 and 2. Everything else depends on them.

## 2. The forward pass

### One neuron: two steps

From your slides, every neuron does the same thing:

- **Step 1, linear:** z = (Σ wᵢxᵢ) + b
- **Step 2, nonlinear:** a = g(z)

This is the "linear then nonlinear" combination from the video you pasted earlier. Without g, stacked layers would collapse into a single linear map.

### One layer: vectorized

Computing neuron by neuron is slow, so a whole layer is computed at once:

**z[l] = W[l] a[l−1] + b[l]**
**a[l] = g(z[l])**

- a[l−1] is the output of the previous layer (the input to this one)
- W[l] is the weight matrix and b[l] is the bias vector for layer l
- g is applied **element-wise** to every entry of z[l]

Shapes, which are the most common source of bugs: if layer l−1 has n neurons and layer l has m neurons, then W[l] is **m × n**, a[l−1] is n × 1, and b[l] and z[l] are m × 1. Row j of W holds the weights feeding neuron j.

### The whole network

Start with a[0] = x, then repeat the two equations layer by layer:

x = a[0] → (z[1], a[1]) → (z[2], a[2]) → … → **ŷ = a[L]**

The last activation is the prediction. Information flows in one direction only, which is why this is called a *feedforward* network.

### Choosing the output activation

Your slides give this guide:

| Task | Output activation | Output |
|---|---|---|
| Regression | None (linear) | Any real number |
| Binary classification | Sigmoid | Probability in (0, 1) |
| Multiclass classification | Softmax | Probabilities that sum to 1 |

Hidden layers usually use ReLU. The slides advise against sigmoid and tanh there because of vanishing gradients.

## 3. Worked example (from your slides)

**Network:** 2 inputs → 2 hidden sigmoid neurons → 1 sigmoid output.
**Input:** x = [1, 0]. **Target:** y = 1.

**Weights:**
- Hidden: w₁₁=1, w₂₁=1 into h₁; w₁₂=−1, w₂₂=−1 into h₂
- Hidden biases: b₁=0, b₂=1
- Output: w₃₁=2, w₃₂=−1, b₃=−1

**Hidden layer**
- z₁ = (1)(1) + (1)(0) + 0 = **1.0**
- z₂ = (−1)(1) + (−1)(0) + 1 = **0.0**
- h₁ = σ(1.0) = 1/(1+e⁻¹) ≈ **0.731**
- h₂ = σ(0.0) = **0.500**

**Output layer**
- z₃ = 2(0.731) + (−1)(0.500) + (−1) = 1.462 − 0.500 − 1 = **−0.038**
- ŷ = σ(−0.038) ≈ **0.491**

**Loss (binary cross-entropy)**
- L = −[1·ln(0.491) + 0·ln(1−0.491)] = −(−0.712) = **0.712**

The network said "49% chance of class 1" when the truth was class 1, so the loss is fairly high.

**Notation gotcha:** the slides write w₁₁ x₁ + w₂₁ x₂ for neuron 1, so the first index is the *input* and the second is the *neuron*. In the matrix form z = Wa, you need rows to be neurons, so W[1] = [[1, 1], [−1, −1]]. That is the transpose of the matrix printed on the slide. Check: [[1,1],[−1,−1]]·[1,0] + [0,1] = [1, 0] ✓.

## 4. Loss functions

### What a loss function is

A loss L(y, ŷ) takes the truth and the prediction and returns **one number**, where lower means better. It turns "how good is this network?" into something we can minimize. A good loss has two properties:

- It is smallest when ŷ matches y.
- It is **differentiable**, so gradient descent can use its slope.

### Regression: Mean Squared Error

<img width="417" height="106" alt="image" src="https://github.com/user-attachments/assets/cc536791-ebeb-4bb4-9899-6228d591dadf" />


It is the average squared gap between truth and prediction.

- **Why square?** It makes every error positive so errors can't cancel, and it is smooth, so the gradient is easy.
- **Penalizes large errors heavily:** an error of 10 costs 100, while ten errors of 1 cost only 10 in total. The model will work hard to avoid big misses.
- **Downside:** one outlier can dominate the loss.

**Example:** y = [3, 5], ŷ = [2, 9]. Errors are 1 and −4, so MSE = (1 + 16)/2 = **8.5**.

### Classification: Cross-Entropy

<img width="342" height="105" alt="image" src="https://github.com/user-attachments/assets/6876ab3b-36a1-4ad0-aca1-978821e77e8a" />


Here y is a **one-hot** vector (1 for the true class, 0 elsewhere) and ŷ is a vector of predicted probabilities. Because y is zero everywhere except the true class, the sum collapses to:

**L = −log(probability the model gave to the correct class)**

Your slides' example, with the true class Cat = [1, 0, 0]:

| Prediction | Probability on Cat | Loss = −ln(p) |
|---|---|---|
| [0.8, 0.1, 0.1] (good) | 0.8 | **0.22** |
| [0.1, 0.2, 0.7] (bad) | 0.1 | **2.30** |

**Why the log works:** −log(p) is 0 when p = 1, grows slowly as p drops, and explodes as p → 0. That is what "penalizes being confident and wrong" means: p = 0.01 on the right answer gives a loss of 4.6.

### Binary cross-entropy (used in your worked example)

With one output ŷ = P(class 1), the two-class version is:

**L = −[ y log(ŷ) + (1−y) log(1−ŷ) ]**

- If y = 1, only the first term survives: L = −log(ŷ).
- If y = 0, only the second term survives: L = −log(1−ŷ).

In both cases the loss is the negative log of the probability assigned to the correct answer.

### Why not use MSE for classification?

Suppose y = 1 and the model is confidently wrong with ŷ = 0.01:

- MSE: (1 − 0.01)² ≈ **0.98**, which is capped near 1
- Cross-entropy: −ln(0.01) ≈ **4.6**, which keeps growing

Cross-entropy punishes confident mistakes much harder, and with sigmoid or softmax outputs it also gives stronger gradients when the model is wrong. MSE tends to learn very slowly in that situation.

### Softmax (the multiclass output)

Softmax turns raw scores (logits) z into probabilities:

**ŷ_c = e^(z_c) / Σ_k e^(z_k)**

Example: z = [2, 1, 0] gives e^z = [7.39, 2.72, 1.00], which sum to 11.11, so ŷ ≈ [0.665, 0.245, 0.090]. They sum to 1, and softmax followed by cross-entropy is the standard pairing.

## 5. Loss vs. average loss

For a single example the loss is L(y, ŷ). In training we average over a batch of N examples, which is why MSE has the 1/N. The training objective is therefore the average loss over the data, and that is the number gradient descent tries to push down.

## 6. How this connects to what comes next

The forward pass makes ŷ, and the loss turns it into a single score. Slides 139-145 describe the "loss landscape": the loss is a function of all the weights, and learning means finding weights where it is low. Backpropagation then uses the chain rule to measure how the loss changes as each weight changes, and your slides continue the same example into that step (pp. 370-376 use δ₃ = −0.509 from this exact forward pass).

## 7. Summary

1. Each layer: z = Wa + b, then a = g(z). Chain the layers to get ŷ.
2. Shapes matter: W[l] is (neurons in l) × (neurons in l−1).
3. The output activation matches the task: linear for regression, sigmoid for binary, softmax for multiclass.
4. The loss reduces prediction quality to one differentiable number.
5. **MSE** for regression, which punishes large errors heavily.
6. **Cross-entropy** for classification, which equals −log of the probability on the correct class.

## Check your understanding

1. Layer 1 has 4 neurons and the input has 3 features. What is the shape of W[1], b[1], and z[1]?
2. In the worked example, why is the (1−y) term zero?
3. A 3-class model outputs [0.5, 0.3, 0.2] and the truth is class 2. What is the cross-entropy loss?
4. Why does cross-entropy punish ŷ = 0.01 on the correct class so much more than MSE does?
5. If you removed all activation functions, what would the forward pass reduce to?

*(Answers: 1. W is 4×3, b and z are 4×1. 2. y = 1, so 1−y = 0. 3. −ln(0.3) ≈ 1.20. 4. −log grows without bound as p→0, while squared error is capped near 1. 5. A single linear map.)*

---
---
# ➡️ Gradient Descent: The Intuition

I'll pull the gradient descent sections from your slides first so the guide matches your course.# Gradient Descent: A Complete Guide

This follows your slides (pp. 139-145 and 244-260) and adds the intuition and numbers behind them.

## 1. The problem it solves

The forward pass and loss give you L(θ), a single number that depends on all the weights and biases (together called θ). Learning means finding **θ\*** that makes L as small as possible.

You can't solve for θ\* directly. A network has millions of parameters and the loss surface is a complicated, non-linear landscape. Gradient descent doesn't try to see the whole landscape. It takes small steps downhill using only local information.

## 2. The intuition: a hiker in fog

You're on a mountain in thick fog and want the lowest valley. You can't see far, so at every step you ask two questions:

1. **Which way is downhill?** Feel the slope under your feet. The **gradient ∇L** points in the direction of steepest *uphill*, so you go the opposite way.
2. **How big a step?** This is the **learning rate η**. Too small and it takes forever. Too big and you overshoot the valley.

## 3. The gradient

For one parameter, the gradient is the ordinary derivative **dL/dθ**: the slope of the loss at the current point.

- Slope positive → loss rises as θ increases → **decrease θ**
- Slope negative → loss falls as θ increases → **increase θ**
- Slope near zero → flat, so you're near a minimum, maximum, or saddle

For many parameters, the gradient is the vector of all partial derivatives:

**∇L = [∂L/∂θ₁, ∂L/∂θ₂, …, ∂L/∂θₙ]**

Each entry says how sensitive the loss is to that one parameter. The vector as a whole points uphill, and its length says how steep the slope is.

## 4. The update rule

**θ_new = θ_old − η ∇L(θ_old)**

or, per parameter, **θⱼ := θⱼ − η · ∂L/∂θⱼ**

The minus sign is the "go downhill" part. The loop is:

1. Compute the loss (forward pass)
2. Compute the gradient (backpropagation, covered in your next slides)
3. Update every parameter using the rule above
4. Repeat until the loss stops improving

## 5. The simplest example: L(θ) = θ²

Here dL/dθ = 2θ, so the update is θ ← θ − η(2θ) = θ(1 − 2η). Start at θ = 4:

| Learning rate | Steps | What happens |
|---|---|---|
| η = 0.01 | 4 → 3.92 → 3.84 → 3.77 | Safe but very slow |
| **η = 0.1** | 4 → 3.2 → 2.56 → 2.05 | Smooth descent toward 0 |
| η = 1.0 | 4 → −4 → 4 → −4 | Bounces forever, never converges |
| η = 1.1 | 4 → −4.8 → 5.76 → −6.9 | **Diverges**, getting worse each step |

Notice also that the steps shrink as you approach the minimum even though η is fixed. That happens because the gradient itself shrinks near a flat bottom.

## 6. Worked example: linear regression (as in your slides)

Model: ŷ = mx + c, with parameters θ = {m, c}.
Loss: L = (1/N) Σ (ŷᵢ − yᵢ)²

Taking derivatives (chain rule: the 2 comes from the square):

- ∂L/∂m = (2/N) Σ (ŷᵢ − yᵢ) · xᵢ
- ∂L/∂c = (2/N) Σ (ŷᵢ − yᵢ)

Update rules (from your slides):

- m := m − η ∂L/∂m
- c := c − η ∂L/∂c

**Numbers:** data (1, 2), (2, 4), (3, 6). The true line is y = 2x. Start at m = 0, c = 0, with η = 0.1.

**Step 0:**
- Predictions: all 0, so errors (ŷ − y) = −2, −4, −6
- Loss = (4 + 16 + 36)/3 = **18.67**
- ∂L/∂m = (2/3)(−2·1 − 4·2 − 6·3) = **−18.67**
- ∂L/∂c = (2/3)(−12) = **−8.0**
- m = 0 − 0.1(−18.67) = **1.867**, c = 0 − 0.1(−8) = **0.8**

**Step 1:** loss drops to **0.296**, with gradients of about 1.96 and 1.07. The update gives m ≈ 1.67 and c ≈ 0.69.

**Step 2:** loss is **0.073**.

Both gradients were negative at the start, meaning "increasing m and c lowers the loss", and the updates moved them up. The loss fell from 18.67 to under 0.1 in two steps. It then settles in a long flat valley where m and c trade off (many (m, c) pairs fit these three points almost equally well), so it keeps creeping toward m = 2, c = 0 slowly.

## 7. The learning rate

| Too small | Just right | Too large |
|---|---|---|
| Slow, may stall | Steady decrease | Oscillates or diverges |
| Wastes compute | Converges | Loss may become NaN |

**Practical signs:** if the loss goes **up** or becomes NaN, η is too high. If it barely moves, η is too low. In practice, people try values like 0.1, 0.01, 0.001 and often use a **schedule** that decreases η over time.

## 8. Local vs. global minima

From your slides:

- **Global minimum:** the true lowest point of the landscape
- **Local minimum:** a valley that is lowest nearby but not overall, where gradient descent can get trapped (gradient is zero there)

Two clarifications that are useful beyond the slides:

- **Convex losses** (like MSE for linear regression) have one valley, so gradient descent reliably finds it.
- In **deep networks**, true bad local minima turn out to be less of a problem than people once feared. **Saddle points** (flat in some directions, curved in others) and flat plateaus are the more common difficulty. Noise from mini-batches (next section) also helps escape them.

## 9. Batch, stochastic, and mini-batch

The gradient of the loss is an average over the data, so a key question is how much data to use per step. Your slides compare three options:

| Aspect | Batch GD | Stochastic GD (SGD) | Mini-batch GD |
|---|---|---|---|
| Data per update | Entire dataset | One example | A small batch |
| Updates per epoch | One | N (one per example) | N / batch size |
| Speed per update | Slow | Fast | Moderate |
| Stability | Smooth | Noisy, erratic | Fairly smooth |
| Efficiency | Poor on big data | Efficient | Very efficient (GPU) |
| Memory | High | Low | Moderate |

**Mini-batch is the standard.** It balances the stability of batch GD with the speed of SGD, and GPUs process a batch in parallel. Typical batch sizes are 32, 64, 128, or 256.

**Vocabulary:**
- **Epoch:** one full pass through the training data
- **Iteration:** one parameter update
- Example: 1,000 examples with batch size 100 → 10 iterations per epoch

## 10. Where backpropagation fits

Gradient descent says "move opposite to the gradient." It does not say *how to compute* the gradient of a loss through millions of weights. That is what **backpropagation** does, using the chain rule. The two work together:

- **Backpropagation:** computes ∇L
- **Gradient descent:** uses ∇L to update θ

Your next slides (pp. 263-304) cover this.

## 11. Common pitfalls

- **Features on very different scales:** the landscape becomes a long, narrow valley, and gradient descent zigzags. Normalizing inputs helps a lot.
- **Bad initialization:** all-zero weights make every neuron in a layer identical, so they never differentiate. Use small random values.
- **Forgetting that gradient descent only finds a *good-on-training-data* solution.** As the UAT lesson noted, low training loss does not guarantee good performance on unseen data.
- **Not shuffling data:** with mini-batches, shuffle every epoch so batches aren't biased.

## 12. Beyond plain gradient descent

Modern training uses variants that add smarter step rules. Your course will likely cover these later, so here is just the idea:

- **Momentum:** keeps a running average of past gradients so you roll through flat or noisy regions instead of stopping and starting
- **Adam (and RMSprop):** adapts the step size for each parameter individually, so η effectively changes per weight

Adam is the common default today.

## 13. Summary

1. Learning is minimizing the loss L(θ).
2. The gradient points uphill, so we step the opposite way: **θ ← θ − η∇L**.
3. η controls step size: too small is slow, too large diverges.
4. Convex problems have one valley, while deep networks have saddles and plateaus.
5. Mini-batch GD is the practical standard.
6. Backpropagation computes the gradient and gradient descent uses it.

## Check your understanding

1. L(θ) = θ², θ = 3, η = 0.25. What is θ after one update?
2. In the linear regression example, both gradients were negative. Which way did m and c move, and why?
3. Why does gradient descent slow down near a minimum even with fixed η?
4. 6,000 examples, batch size 200. How many iterations per epoch?
5. Your loss suddenly becomes NaN after a few steps. What is the most likely cause?

*(Answers: 1. Gradient = 6, so θ = 3 − 0.25·6 = 1.5. 2. Both increased, because the negative gradient times −η gives a positive change. 3. The gradient shrinks toward zero near a flat bottom, so steps get smaller. 4. 30. 5. The learning rate is too high and the updates are diverging.)*

---
---
