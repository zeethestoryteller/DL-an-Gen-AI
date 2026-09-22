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
| `num_workers` | How many subprocesses load data in parallel; `0` keeps it single-threaded/simple |

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
### e. What comes next (not yet in this transcript, but the natural next steps)

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

## Mental model / cheat sheet

- **`TensorDataset`** = pairs X and y so they index together
- **`DataLoader`** = iterates a dataset in batches, optionally shuffled/parallel
- **`shuffle=True`** for train, **`shuffle=False`** for test/eval
- **`batch_size`** = how many samples per gradient update (8 here; 32/64/128 common in practice)
- **Full loop per epoch**: `model.train()` → forward → loss → `zero_grad()` → `backward()` → `step()`, then optionally `model.eval()` + `torch.inference_mode()` to check test loss

