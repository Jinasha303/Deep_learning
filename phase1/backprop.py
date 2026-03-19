import torch
import matplotlib.pyplot as plt

# XOR DATA
x = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

# WEIGHTS & BIASES
w1 = torch.randn(2, 3, requires_grad=True)
w2 = torch.randn(3, 1, requires_grad=True)

b1 = torch.zeros(1, 3, requires_grad=True)
b2 = torch.zeros(1, 1, requires_grad=True)

# FORWARD PASS
def forward(x):
    z1 = x @ w1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ w2 + b2
    a2 = torch.sigmoid(z2)
    return a1, a2

# LOSS FUNCTION (BCE)
def compute_loss(a2, y):
    return -(y * torch.log(a2) + (1-y) * torch.log(1-a2)).mean()

# TRAINING
losses = []
lr = 0.3

for epoch in range(1000):
    a1, a2 = forward(x)

    loss = compute_loss(a2, y)
    losses.append(loss.item())

    loss.backward()

    # UPDATE
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        b1 -= lr * b1.grad
        b2 -= lr * b2.grad

        # RESET GRADIENTS
        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()

    # PRINT EVERY 100 EPOCHS
    if epoch % 100 == 0:
        with torch.no_grad():
            acc = ((a2 > 0.5) == y).float().mean()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

# PLOT LOSS
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss (Manual Backprop)")
plt.show()

# FINAL OUTPUT
with torch.no_grad():
    _, out = forward(x)
    print("\nPredictions:", out.round().T)
    print("Expected:   ", y.T)