# PyTorch Univariate Linear Regression — Learning Guide

This walks through everything in the transcript, organized into clear steps with working code. I've filled in the parts that were implied but not fully typed out, and added the pieces you'll need next (model, training loop) since the video sets those up.
#
### a. Setup

```python
import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

- `nn` holds PyTorch's neural network building blocks (layers, loss functions, etc.)
- Checking CUDA tells you whether training will run on GPU (fast) or CPU (fine for this toy problem, but good habit for later)
#
### b. Define the "ground truth" and generate synthetic data

The whole point of a toy problem is that *you* know the real weight and bias, so you can check whether the model learns them correctly.

```python
weight = 0.7   # slope
bias = 0.3     # intercept

start, end, step = 0, 1, 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)   # shape [50, 1]
y = weight * X + bias                                   # shape [50, 1]

print(X[:10], y[:10])
print(X.shape, y.shape)
```

**Why `unsqueeze`?** `torch.arange` gives a 1D tensor of shape `[50]`. Matrix operations (and later, `nn.Linear`) expect a 2D tensor of shape `[batch, features]`. `unsqueeze(dim=1)` turns `[50]` into `[50, 1]` — 50 samples, 1 feature each.

**Indexing trick mentioned in the video:**
```python
X[:10]   # first 10 rows, all columns
X[10:]   # from row 10 to the end
```
#
### c. Train/test split

```python
train_split = int(0.8 * len(X))   # 40

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(X_test)   # 40, 10
```

This is a **sequential** split, not a random one — fine for this ordered synthetic data, but for real datasets you'd usually shuffle first (e.g. `sklearn.model_selection.train_test_split`) to avoid any ordering bias, while still making sure train/test never overlap.
#
### d. Wrap the data: `TensorDataset` + `DataLoader`

```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

BATCH_SIZE = 8
NUM_WORKERS = 0   # 0 = load data in the main process (simplest, fine for small data)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,          # reshuffle every epoch so the model doesn't memorize order
    num_workers=NUM_WORKERS
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # no need to shuffle — we're only evaluating
    num_workers=NUM_WORKERS
)
```

| Piece | What it does |
|---|---|
| `TensorDataset(X, y)` | Pairs each input row with its label so you can index/iterate them together instead of managing two tensors separately |
| `DataLoader` | Wraps a dataset and gives you an iterator that yields **batches** |
| `shuffle=True` (train only) | Randomizes sample order each epoch → prevents the model from learning spurious order patterns |
| `num_workers` | How many subprocesses(halper) load data in parallel; `0` keeps it single-threaded/simple |
|`batch_size`| Decides:in how many parts the data will be devided|

**Check what you built:**
```python
print(len(train_dataloader), len(test_dataloader))  # 5, 2  (40/8=5, 10/8→2 batches)
print(train_dataloader.batch_size)
print(len(train_dataset), len(test_dataset))         # 40, 10

for batch_X, batch_y in train_dataloader:
    print(batch_X.shape, batch_y.shape)   # [8,1] [8,1]
    print(batch_X.flatten(), batch_y.flatten())
    break
```
`.flatten()` just reshapes `[8,1]` → `[8]` for easier reading — it's cosmetic, not required for training.
#
### e. What comes next 

Since the video stops right before modeling, here's the rest of the standard PyTorch workflow so you have the full picture:

### Define the model
```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(42)
model = LinearRegressionModel().to(device)
print(model.state_dict())   # randomly initialized weight & bias
```

### Loss function and optimizer
```python
loss_fn = nn.L1Loss()   # Mean Absolute Error
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
```

### Training loop
```python
epochs = 200

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        y_pred = model(batch_X)
        loss = loss_fn(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # quick eval each epoch
    model.eval()
    with torch.inference_mode():
        test_loss = sum(loss_fn(model(xb.to(device)), yb.to(device)) for xb, yb in test_dataloader)

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Train loss: {loss.item():.4f} | Test loss: {test_loss.item():.4f}")

print(model.state_dict())   # should approach weight=0.7, bias=0.3
```

### Visualize predictions
```python
model.eval()
with torch.inference_mode():
    y_preds = model(X_test.to(device))

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, c="b", s=10, label="Train data")
plt.scatter(X_test, y_test, c="g", s=10, label="Test data")
plt.scatter(X_test, y_preds.cpu(), c="r", s=10, label="Predictions")
plt.legend()
plt.show()
```
#
### Mental model / cheat sheet

- **`TensorDataset`** = pairs X and y so they index together
- **`DataLoader`** = iterates a dataset in batches, optionally shuffled/parallel
- **`shuffle=True`** for train, **`shuffle=False`** for test/eval
- **`batch_size`** = how many samples per gradient update (8 here; 32/64/128 common in practice)
- **Full loop per epoch**: `model.train()` → forward → loss → `zero_grad()` → `backward()` → `step()`, then optionally `model.eval()` + `torch.inference_mode()` to check test loss

---
# Plotting Routine for Linear Regression — Learning Guide (Part 2)

This continues from the data-loading guide. Here's the `plot_predictions()` function the video walks through, reconstructed from what's described, plus notes on how each parameter behaves.

## The `plot_predictions()` function

```python
import matplotlib.pyplot as plt

def plot_predictions(train_data=X_train,
                      train_labels=y_train,
                      test_data=X_test,
                      test_labels=y_test,
                      predictions=None):
    """
    Plots training data, test data, and (optionally) model predictions.
    """
    plt.figure(figsize=(10, 7))

    # Training data — green
    plt.scatter(train_data, train_labels,
                c="g", s=25, alpha=1.0,
                label="Training data", marker="o")

    # Test data — blue
    plt.scatter(test_data, test_labels,
                c="b", s=25, alpha=1.0,
                label="Testing data", marker="o")

    # Predictions — red, only if provided
    if predictions is not None:
        plt.scatter(test_data, predictions,
                    c="r", s=25, alpha=0.8,
                    label="Predictions", marker="s")

    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Linear Regression", fontsize=16, fontweight="bold")
    plt.legend(prop={"size": 12})
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## Parameter-by-parameter breakdown

| Parameter | What it controls | Notes from the video |
|---|---|---|
| `train_data` / `train_labels` | X and Y for the scatter's positions | Default to `X_train`, `y_train` so you can call the function with no args |
| `test_data` / `test_labels` | Same, for test set | Default to `X_test`, `y_test` |
| `predictions` | Model output for `X_test` | Defaults to `None` — the function only plots red prediction points **if** you pass something in |
| `c` | Marker color | Green for train, blue for test, red for predictions — an easy visual convention: two "true" data colors + one "model output" color |
| `s` | Marker size | `25` is a reasonable default; the video demoed bumping it to `45` to make markers much larger — purely cosmetic, doesn't affect the model |
| `alpha` | Transparency (`0`=invisible, `1`=fully opaque) | Useful when scatter plots overlap — e.g. lowering test data's alpha so predictions plotted on top of it remain clearly visible |
| `marker` | Shape of the plotted points | `"o"` (circle) for data, `"s"` (square) worked well as a visual differentiator for predictions — any valid matplotlib marker works |
| `label` | Text shown in the legend | Must be set for `plt.legend()` to display anything meaningful |
| `plt.grid(alpha=...)` | Background grid | Also has its own transparency control, independent of marker alpha |
| `plt.tight_layout()` | Auto-adjusts spacing | Prevents labels/titles from getting clipped |
| `plt.show()` | Actually renders the figure | **Required** — without it, in many environments (like plain scripts) nothing displays |

## Using it — before training (sanity check)

```python
plot_predictions()  # just shows train (green) vs test (blue), no predictions yet
```

## Using it — after training (the payoff)

Once you have a trained model (from Part 1 of this guide):

```python
model.eval()
with torch.inference_mode():
    y_preds = model(X_test.to(device))

plot_predictions(predictions=y_preds.cpu())
```

If the model learned well, the **red squares** (predictions) should land almost exactly on top of the **blue circles** (true test data).

## Experimenting, as shown in the video

Try these tweaks yourself to build intuition:

```python
# Make test data markers huge and predictions barely visible on top:
plt.scatter(test_data, test_labels, c="b", s=45, alpha=0.3, label="Testing data")
plt.scatter(test_data, predictions, c="r", s=45, alpha=1.0, label="Predictions")
```

This kind of alpha/size play is purely about **readability when points overlap** — a bigger, more opaque "Predictions" layer on top of a smaller, more transparent "Testing data" layer makes it easy to see how closely predictions track ground truth.

> load data → build model → train → **visualize with this function**. 

---

# Defining the Linear Regression Model in PyTorch — Learning Guide (Part 3)

This part covers the actual model class, built manually with `nn.Parameter` (the "from scratch" way, as opposed to just using `nn.Linear` — both are valid, and it's worth understanding this version since it shows you what's happening under the hood).

## Imports

```python
import torch
from torch import nn
```

- `torch` — core tensor library
- `torch.nn` — PyTorch's toolkit for building neural network models (layers, parameters, loss functions)

## The model class

```python
class LinearRegressionModel(nn.Module):
    """
    Linear Regression Model: y = weight * x + bias

    Input: univariate (1 feature per sample)
    Output: 1 prediction per sample
    Parameters: weight, bias
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )
        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
```

## Why each piece matters

| Piece | Purpose |
|---|---|
| `class LinearRegressionModel(nn.Module)` | Every custom PyTorch model **must** inherit from `nn.Module` — this gives you parameter tracking, `.parameters()`, GPU-moving (`.to(device)`), saving/loading, and autograd integration for free |
| `super().__init__()` | Calls `nn.Module`'s constructor. **Skip this and your parameters won't get registered** — meaning no gradients, no training. Always the first line in `__init__` |
| `nn.Parameter(...)` | Wraps a tensor and tells PyTorch "this is a learnable parameter — track it, include it in `.parameters()`, update it during optimization" |
| `torch.randn(1, dtype=torch.float)` | Initializes the parameter with a random value drawn from a standard normal distribution — training will nudge this toward the true value over time |
| `requires_grad=True` | Tells autograd to compute gradients for this tensor during backpropagation (this is somewhat redundant with `nn.Parameter`, which sets it automatically, but it's explicit here) |
| `forward(self, x)` | Defines **how input flows through the model** — i.e., how you compute `y` from `x`. This is the one method every `nn.Module` subclass must implement |

**Key relationship:** `nn.Module` (the parent class) handles backprop, parameter bookkeeping, and training-loop integration automatically — as long as you (1) call `super().__init__()` and (2) register parameters via `nn.Parameter`.

## Using the model

```python
torch.manual_seed(42)   # for reproducible random init — optional but good practice
model = LinearRegressionModel()

# See the randomly initialized parameters
print(list(model.parameters()))
print(model.state_dict())
```

### Running a prediction (inference)

```python
x = torch.tensor([[1.0], [2.0], [3.0]])   # shape [3, 1] — 3 samples, 1 feature
print(x, x.shape)

y_pred = model(x)   # calling model(x) automatically runs model.forward(x)
print(y_pred)
```

At this point, predictions are **random and meaningless** — because `weights` and `bias` are still randomly initialized. Training is exactly the process of adjusting them so predictions match reality.

## A note on `model(x)` vs `model.forward(x)`

Always call `model(x)`, not `model.forward(x)` directly. `nn.Module.__call__` does extra bookkeeping (hooks, etc.) before invoking `forward()` — calling `forward()` directly skips that.

## Connecting to Part 1 (data pipeline)

If you built the `DataLoader`s from the earlier guide, you can already sanity-check the model on a real batch:

```python
for batch_X, batch_y in train_dataloader:
    preds = model(batch_X)
    print(preds.shape, batch_y.shape)   # should match: [8, 1] and [8, 1]
    break
```

## Manual vs. `nn.Linear` — worth knowing

The video builds weight/bias by hand for teaching purposes. In practice, for a simple linear layer you'd normally just write:

```python
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)
```

`nn.Linear` does exactly the same `weight * x + bias` computation internally, but is the standard, battle-tested building block you'll reuse for every larger model (it generalizes cleanly to multiple input/output features). Understanding the manual version first is exactly why this lesson is valuable — it demystifies what `nn.Linear` is doing for you.

---






