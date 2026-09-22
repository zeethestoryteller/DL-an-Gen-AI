# The Training Loop — Complete Guide (Part 5)

This is the piece that ties everything together: model + data + loss function + optimizer, all wired into a working training loop. By the end you'll have a model that actually learns `weight ≈ 0.7, bias ≈ 0.3`.

## 1. Initialize the model

```python
torch.manual_seed(42)
model_0 = LinearRegressionModel()

# Inspect raw parameters
print(list(model_0.parameters()))

# Inspect named parameters (much more readable)
print(model_0.state_dict())
# e.g. OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])
```

`state_dict()` is the one to reach for day-to-day — it labels each parameter by name instead of just dumping raw tensors.

## 2. Make predictions *before* training (sanity check)

```python
with torch.inference_mode():
    y_preds = model_0(X_test)

print(len(X_test), len(y_preds))
print(y_preds)

plot_predictions(predictions=y_preds)
```

`torch.inference_mode()` disables gradient tracking for anything inside the block — you use it whenever you're just running the model forward, not training it (faster, uses less memory). At this point predictions will be way off, since `weights`/`bias` are still random — that's expected and exactly what training is going to fix.

## 3. Configure the loss function and optimizer

```python
# Loss function — quantifies how wrong predictions are
loss_fn = nn.L1Loss()   # MAE — good default for regression
# (you could swap in your custom MeanAbsoluteError() or HuberLoss() from Part 4)

# Optimizer — the algorithm that updates parameters using gradients
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
```

| Concept | What it is |
|---|---|
| `loss_fn` | Measures prediction error. `nn.L1Loss` (MAE) and `nn.MSELoss` are common for regression; `nn.CrossEntropyLoss` for classification |
| `torch.optim.SGD` | Stochastic Gradient Descent — updates each parameter by a step proportional to the gradient of the loss w.r.t. that parameter |
| `params=model_0.parameters()` | Tells the optimizer **which** tensors to update — everything registered via `nn.Parameter` inside the model |
| `lr=0.01` | Learning rate (often written α or λ) — how big each update step is |

```python
print(optimizer)
print(optimizer.param_groups)
print(len(list(model_0.parameters())))   # confirms how many parameter tensors are being optimized
```

## 4. The four steps inside every training iteration

Every batch, in this exact order:

1. **Forward pass** — `y_pred = model(x)`
2. **Compute loss** — `loss = loss_fn(y_pred, y)`
3. **Zero gradients** — `optimizer.zero_grad()` (gradients accumulate by default, so you must clear old ones each step)
4. **Backward pass** — `loss.backward()` (computes gradients via autograd)
5. **Update parameters** — `optimizer.step()` (applies the gradient-descent update rule)

```python
model_0.train()   # puts the model in training mode (matters for layers like dropout/batchnorm)

y_pred = model_0(X_train)
loss = loss_fn(y_pred, y_train)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`model.train()` vs `model.eval()` matters because some layers behave differently during training vs inference. Even though this simple model has none of those layers, it's good habit to always set the mode explicitly.

## 5. The full training + evaluation loop (batched, with a DataLoader)

This is the complete version, using the `train_dataloader`/`test_dataloader` from Part 1:

```python
torch.manual_seed(42)

epochs = 350
print_interval = 20

# Tracking variables for plotting loss curves later
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # ---------- Training ----------
    model_0.train()
    epoch_train_loss = 0
    num_train_batches = 0

    for batch_X, batch_y in train_dataloader:
        y_pred = model_0(batch_X)             # 1. forward pass
        batch_loss = loss_fn(y_pred, batch_y) # 2. compute loss

        optimizer.zero_grad()                 # 3. zero gradients
        batch_loss.backward()                 # 4. backward pass
        optimizer.step()                      # 5. update parameters

        epoch_train_loss += batch_loss.item()
        num_train_batches += 1

    avg_train_loss = epoch_train_loss / num_train_batches

    # ---------- Evaluation ----------
    model_0.eval()
    epoch_test_loss = 0
    num_test_batches = 0

    with torch.inference_mode():
        for batch_X, batch_y in test_dataloader:
            test_pred = model_0(batch_X)
            batch_test_loss = loss_fn(test_pred, batch_y)

            epoch_test_loss += batch_test_loss.item()
            num_test_batches += 1

    avg_test_loss = epoch_test_loss / num_test_batches

    # ---------- Logging ----------
    if epoch % print_interval == 0:
        epoch_count.append(epoch)
        train_loss_values.append(avg_train_loss)
        test_loss_values.append(avg_test_loss)
        print(f"Epoch: {epoch:4d} | Train loss: {avg_train_loss:.5f} | Test loss: {avg_test_loss:.5f}")

print("Training completed.")
print(f"Average training loss: {avg_train_loss:.5f}")
print(f"Average test loss: {avg_test_loss:.5f}")
```

### Key details worth noting

- **`batch_loss.item()`** — pulls a Python float out of a single-element tensor. You accumulate with `.item()` (not the raw tensor) so you're not needlessly holding onto the computation graph.
- **Accumulate-then-divide pattern** — sum each batch's loss, then divide by the batch count at the end of the epoch → gives you the *average* loss for that epoch, which is what you actually want to track and plot (comparing raw summed loss across epochs with different batch counts wouldn't be meaningful).
- **`with torch.inference_mode()`** around the eval loop — no gradients needed during evaluation, so this saves memory/compute.
- Only logging every `print_interval` (20) epochs keeps the output readable over 350 epochs.

## 6. Plot the loss curves

```python
plt.figure(figsize=(8, 5))
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and Test Loss Curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
```

Both curves should decline and roughly converge — that's the visual confirmation that training is working and the model isn't overfitting (train loss dropping while test loss stays flat/rises would signal overfitting).

## 7. Compare learned parameters to ground truth

```python
learned_params = model_0.state_dict()
for param_name, param_value in learned_params.items():
    print(f"{param_name}: {param_value}")

print("\nGround truth:")
print(f"weight: {weight}")
print(f"bias: {bias}")
```

After ~350 epochs, the learned `weights` and `bias` should land very close to `0.7` and `0.3` — that's the whole model working as intended.

## 8. Predictions after training (visual confirmation)

```python
model_0.eval()
with torch.inference_mode():
    y_preds_trained = model_0(X_test)

plot_predictions(predictions=y_preds_trained)
```

Now the red prediction markers should sit almost exactly on top of the blue test-data markers — a dramatic contrast to the random, way-off predictions from before training.

---

## The complete picture

| Part | Piece | Purpose |
|---|---|---|
| 1 | Data + `TensorDataset`/`DataLoader` | Generate, split, and batch data |
| 2 | `plot_predictions()` | Visualize train/test/predictions |
| 3 | `LinearRegressionModel(nn.Module)` | Define learnable weight & bias, forward pass |
| 4 | Custom `MeanAbsoluteError`, `HuberLoss` | Quantify prediction error |
| 5 (this) | Optimizer + training loop | Actually learn the parameters |

