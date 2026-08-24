"""
Phase 4, Exercise 1: PyTorch fundamentals, from zero.

Run this section by section (or all at once) and read the printed output --
it's designed to show you exactly what each core PyTorch concept actually
does, not just describe it.

USAGE:
    python phase4_pytorch/pytorch_basics.py
"""

import torch

print("=" * 60)
print("PART 1: Tensors")
print("=" * 60)

# A tensor is PyTorch's core data structure -- think of it as a NumPy array
# that can also live on a GPU and track gradients.
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b (elementwise) = {a * b}")
print(f"a.dot(b) = {a.dot(b)}")

# Tensors have a shape, just like NumPy arrays
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"\nmatrix:\n{matrix}")
print(f"matrix.shape = {matrix.shape}")

# Moving a tensor to GPU is one line -- this is the whole reason PyTorch
# is useful for anything performance-heavy (like training SAM or 3DGS).
if torch.cuda.is_available():
    gpu_tensor = a.to("cuda")
    print(f"\na moved to GPU: {gpu_tensor}, device={gpu_tensor.device}")
else:
    print("\n(No GPU available in this environment -- skipping GPU demo)")


print("\n" + "=" * 60)
print("PART 2: Autograd -- automatic differentiation")
print("=" * 60)

# This is the single most important PyTorch concept. Training a model means
# repeatedly asking "how should I adjust each parameter to reduce the error?"
# That requires computing gradients (derivatives) -- autograd does this
# automatically, for arbitrarily complex computations, without you writing
# any calculus by hand.

# requires_grad=True tells PyTorch: "track every operation done to this
# tensor, so I can later ask for the gradient."
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")

# Let's define y = x^2 + 2x + 1
y = x**2 + 2 * x + 1
print(f"y = x^2 + 2x + 1 = {y}")

# .backward() computes dy/dx automatically, using the chain rule, by
# walking backward through every operation that was recorded.
y.backward()

# The gradient gets stored in x.grad
# By calculus: dy/dx = 2x + 2, so at x=3, dy/dx should be 8
print(f"dy/dx at x=3 (computed by autograd): {x.grad}")
print("(Manually: dy/dx = 2x + 2, at x=3 that's 2*3+2 = 8 -- matches!)")

print("\nThis 'compute gradients automatically' mechanism is EXACTLY what")
print("powers model training: the 'loss' is just a more complex version of")
print("y above, and .backward() tells us how to adjust every parameter in")
print("the model to reduce that loss.")


print("\n" + "=" * 60)
print("PART 3: nn.Module -- defining a model")
print("=" * 60)

import torch.nn as nn

# nn.Module is the base class for all PyTorch models. You define:
#   - what layers/parameters the model has (in __init__)
#   - how data flows through them (in forward)
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A single linear layer: y = weight * x + bias
        # This is literally the same math as the y = x^2+2x+1 example above,
        # except now "weight" and "bias" are learnable parameters instead
        # of something we wrote by hand.
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)


model = TinyModel()
print(f"Model architecture:\n{model}")

# Every model starts with random parameters
print(f"\nInitial weight: {model.linear.weight.item():.4f}")
print(f"Initial bias: {model.linear.bias.item():.4f}")

# Running data through the model is just calling it like a function
test_input = torch.tensor([[5.0]])
output = model(test_input)
print(f"\nmodel(5.0) = {output.item():.4f}  (using random, untrained parameters)")


print("\n" + "=" * 60)
print("PART 4: A real (tiny) training loop")
print("=" * 60)

# Goal: train the model above to learn y = 3x + 1
# We'll generate some data following that rule, then let the model
# discover the "3" and "1" purely from examples, using gradient descent.

torch.manual_seed(42)  # for reproducible results

# Generate training data: y = 3x + 1, with a little noise
X_train_raw = torch.rand(100, 1) * 10  # 100 random x values between 0 and 10
y_train = 3 * X_train_raw + 1 + torch.randn(100, 1) * 0.1  # y = 3x + 1 + noise

# --- Input normalization ---
# Raw x values range 0-10 (mean ~5). This causes UNEVEN gradients: the
# weight's gradient gets multiplied by x internally, so with x averaging 5,
# weight gets a much stronger/faster training signal than bias does. Result:
# weight converges quickly, bias lags far behind (you saw this in your run --
# bias was still climbing from 0.78 toward 1.0 even after 200 epochs).
#
# Standardizing x to mean=0, std=1 removes this imbalance -- both parameters
# now get comparably-scaled gradients, so they converge at similar speed.
X_mean = X_train_raw.mean()
X_std = X_train_raw.std()
X_train = (X_train_raw - X_mean) / X_std
print(f"Raw X range: [{X_train_raw.min():.2f}, {X_train_raw.max():.2f}], mean={X_mean:.2f}")
print(f"Normalized X range: [{X_train.min():.2f}, {X_train.max():.2f}], mean={X_train.mean():.2f}")
print("(Note: since we normalized X, the model now learns weight/bias for the")
print(" NORMALIZED input -- the learned numbers won't be exactly 3.0/1.0 anymore,")
print(" but convergence SPEED is what we're comparing here.)\n")

# Loss function: measures how wrong the model's predictions are.
# Mean Squared Error is standard for this kind of regression task.
loss_fn = nn.MSELoss()

# Optimizer: the algorithm that actually updates the model's parameters
# based on the gradients autograd computes. SGD = Stochastic Gradient Descent.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Training on NORMALIZED data...\n")
for epoch in range(200):
    # 1. Forward pass: run data through the model
    predictions = model(X_train)

    # 2. Compute loss: how far off are the predictions?
    loss = loss_fn(predictions, y_train)

    # 3. Backward pass: compute gradients for every parameter
    optimizer.zero_grad()  # clear old gradients first (they accumulate otherwise)
    loss.backward()

    # 4. Optimizer step: nudge parameters in the direction that reduces loss
    optimizer.step()

    if epoch % 40 == 0 or epoch == 199:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
              f"weight: {w:.4f} | bias: {b:.4f}")

print("\nCompare bias's convergence speed here to your previous (unnormalized) run:")
print("it should reach a stable value MUCH faster now, instead of still visibly")
print("climbing at epoch 199. This is why normalizing inputs is standard practice.")
print("\nThis exact loop (forward -> loss -> backward -> step) is the")
print("skeleton of EVERY PyTorch training script you'll encounter,")
print("including SAM fine-tuning and 3D Gaussian Splatting optimization.")
