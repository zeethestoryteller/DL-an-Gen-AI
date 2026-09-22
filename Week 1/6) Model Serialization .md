# PyTorch Model Serialization — Learning Guide (Part 6)

This covers saving trained model parameters to disk and loading them back — essential once training takes real time and you don't want to retrain from scratch every session.

## 1. Set up the save directory

```python
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

print(MODEL_PATH.absolute())
```

| Piece | What it does |
|---|---|
| `Path("models")` | Creates a `Path` object pointing at `./models` — not the folder itself yet, just a reference to it |
| `.mkdir(parents=True, exist_ok=True)` | Actually creates the folder. `parents=True` also creates any missing parent directories along the way; `exist_ok=True` means it **won't error** if the folder already exists (important — otherwise re-running the cell would crash) |

## 2. Build the full save path

```python
MODEL_NAME = "linear_regression_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)   # models/linear_regression_model_0.pth
```

- `.pth` (or `.pt`) is the standard PyTorch file extension convention for saved weights.
- `MODEL_PATH / MODEL_NAME` uses `pathlib`'s overloaded `/` operator to join paths cleanly — works correctly across operating systems (no manual `"/"` or `"\\"` string concatenation needed).

## 3. Save the model — the recommended way (`state_dict` only)

```python
print(f"Saving model to: {MODEL_SAVE_PATH}")

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
```

**Why `state_dict()` and not the whole model object?**

| Approach | What's saved | Recommended? |
|---|---|---|
| `torch.save(model_0.state_dict(), path)` | Just the dictionary of learned parameter tensors (weights, biases) | **Yes** — this is the standard, recommended practice |
| `torch.save(model_0, path)` | The entire model object (structure + code references + parameters) | Not generally recommended — brittle across different files/environments, since it pickles the actual class definition |

`state_dict()` returns a dictionary mapping parameter names → tensors (e.g. `{"weights": tensor([0.6989]), "bias": tensor([0.3015])}`; for a real neural net you'd see things like `layer1.weight`, `layer1.bias`). `torch.save` serializes that dictionary to bytes and writes it to disk. The advantage of saving only the state dict is that as long as whatever model class you load it into has matching parameter shapes, it works — you're not tied to the exact class definition that produced it.

**One prerequisite:** the model must already be trained *in that same session* before you save it — you're saving whatever values are currently sitting in `model_0`'s parameters.

## 4. Sanity-check the save

```python
import os

assert MODEL_SAVE_PATH.exists(), "Model save failed"

file_size = os.path.getsize(MODEL_SAVE_PATH)
print(f"File size: {file_size} bytes")
print(f"Number of saved parameters: {len(model_0.state_dict())}")
```

Checking the file actually exists and has a sensible size is good practice — a silent save failure otherwise leaves you debugging much later, at the "why is my model still randomly initialized" stage.

## 5. Load the parameters back

```python
saved_state = torch.load(f=MODEL_SAVE_PATH, weights_only=True)
```

- `torch.load` deserializes the file back into a Python object (here, the same dictionary you saved).
- **`weights_only=True`** (available in recent PyTorch versions) restricts loading to tensors and metadata only, and disallows arbitrary code execution via pickling — this is a **security feature**. Always prefer it, especially if you're ever loading a file you didn't create yourself.

### Loading into an actual model instance

```python
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(saved_state)
loaded_model.eval()   # put in eval mode if you're about to run inference
```

This is the piece the video sets up conceptually (inspecting `saved_state`) but doesn't fully spell out — `load_state_dict()` is how you actually get those saved tensors back **into** a live model object so you can use it for predictions again, without retraining.

## 6. Printing saved parameters safely

The video hits a real, common gotcha here worth understanding:

```python
for name, param in saved_state.items():
    print(name, param)
```

Works fine for tiny models like this one (2 scalar parameters). But calling `.item()` on a tensor only works if that tensor has **exactly one element** — for any real neural network (weight matrices with thousands/millions of entries), `.item()` raises:
```
ValueError: only one element tensors can be converted to Python scalars
```
and printing the raw tensor would flood your screen.

### The robust pattern — branch on tensor size

```python
for name, param in saved_state.items():
    if param.numel() == 1:
        print(f"{name}: {param.item()}")
    else:
        print(f"{name}: shape={tuple(param.shape)}, dtype={param.dtype}")
```

- `param.numel()` — number of elements in the tensor. `== 1` means it's a scalar you can safely convert with `.item()`.
- For anything larger, print **metadata** (shape, dtype) instead of the actual values — this is what you'd realistically want to inspect for a large model anyway, since dumping millions of numbers to a notebook cell is neither useful nor readable.

## 7. Full save/load workflow, put together

```python
from pathlib import Path
import os

# --- Save ---
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / "linear_regression_model_0.pth"

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
assert MODEL_SAVE_PATH.exists(), "Model save failed"
print(f"Saved {len(model_0.state_dict())} parameters ({os.path.getsize(MODEL_SAVE_PATH)} bytes) to {MODEL_SAVE_PATH}")

# --- Load ---
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True))
loaded_model.eval()

# --- Verify it matches the original ---
print(loaded_model.state_dict())
with torch.inference_mode():
    original_preds = model_0(X_test)
    loaded_preds = loaded_model(X_test)
print(torch.allclose(original_preds, loaded_preds))   # should print True
```

That last `torch.allclose` check is the real proof the round-trip worked: the loaded model produces **identical** predictions to the original trained model.

---

## Full pipeline recap

| Part | Topic |
|---|---|
| 1 | Data generation, train/test split, `TensorDataset`/`DataLoader` |
| 2 | `plot_predictions()` visualization |
| 3 | `LinearRegressionModel(nn.Module)` |
| 4 | Custom loss functions (`MeanAbsoluteError`, `HuberLoss`) |
| 5 | Optimizer + full training loop |
| 6 (this) | Saving/loading model parameters (`state_dict`, `torch.save`/`torch.load`) |

You now have the complete, standard PyTorch workflow end-to-end: **build data → build model → train → evaluate → save → reload**. This exact pattern (with bigger models and datasets) is what you'll reuse for essentially everything else in the course.

Want me to assemble all six parts into one downloadable notebook (`.ipynb`) so you have a single runnable reference file?
