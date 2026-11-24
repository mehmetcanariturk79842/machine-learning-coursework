import csv
import random
import time
from statistics import median

import numpy as np
import torch
from sklearn.datasets import make_moons

from engine import Value, value_sum

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

hidden = 8
epochs = 50
learning_rate = 0.5

# Same starting weights for both versions.
rng = np.random.default_rng(42)
W1_start = rng.normal(0, 0.5, (2, hidden))
b1_start = np.zeros(hidden)
W2_start = rng.normal(0, 0.5, hidden)
b2_start = 0.0


def train_engine(epochs):
    W1 = [[Value(W1_start[i, j]) for j in range(hidden)] for i in range(2)]
    b1 = [Value(b) for b in b1_start]
    W2 = [Value(w) for w in W2_start]
    b2 = Value(b2_start)
    params = [w for row in W1 for w in row] + b1 + W2 + [b2]

    losses = []
    for epoch in range(epochs):
        sample_losses = []
        for features, label in zip(X, y):
            hidden_out = [
                (W1[0][j] * features[0] + W1[1][j] * features[1] + b1[j]).relu()
                for j in range(hidden)
            ]
            logit = value_sum(W2[j] * hidden_out[j] for j in range(hidden)) + b2
            p = logit.sigmoid()
            sample_losses.append(-(label * p.log() + (1 - label) * (1 - p).log()))

        loss = value_sum(sample_losses) * (1.0 / len(X))

        for param in params:
            param.grad = 0.0
        loss.backward()
        for param in params:
            param.data -= learning_rate * param.grad

        losses.append(loss.data)

    correct = 0
    for features, label in zip(X, y):
        hidden_out = [
            max(0.0, W1[0][j].data * features[0] + W1[1][j].data * features[1] + b1[j].data)
            for j in range(hidden)
        ]
        logit = sum(W2[j].data * hidden_out[j] for j in range(hidden)) + b2.data
        correct += int((logit > 0) == label)

    return losses, correct / len(X)


def train_torch(epochs):
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    W1 = torch.tensor(W1_start, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(b1_start, dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(W2_start, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(b2_start, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([W1, b1, W2, b2], lr=learning_rate)

    losses = []
    for epoch in range(epochs):
        p = torch.sigmoid((X_t @ W1 + b1).relu() @ W2 + b2)
        loss = torch.nn.functional.binary_cross_entropy(p, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    with torch.no_grad():
        logits = (X_t @ W1 + b1).relu() @ W2 + b2
        accuracy = ((logits > 0).double() == y_t).double().mean().item()

    return losses, accuracy


results = {}
for name, train in [("engine", train_engine), ("pytorch", train_torch)]:
    # Timing: full training run, warm-up once, then median of 5 runs.
    train(epochs)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        losses, accuracy = train(epochs)
        times.append(time.perf_counter() - start)

    results[name] = (losses, accuracy, median(times))
    print(
        f"{name}: final loss={losses[-1]:.6f}, accuracy={accuracy:.4f}, "
        f"median training time={median(times):.4f} s"
    )

speedup = results["engine"][2] / results["pytorch"][2]
print(f"PyTorch was {speedup:.1f} times faster than the engine")

max_loss_difference = max(
    abs(a - b) for a, b in zip(results["engine"][0], results["pytorch"][0])
)
print(f"Max loss difference between engine and PyTorch: {max_loss_difference:.2e}")

with open("lab05/results/moons_training_log.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "engine_loss", "pytorch_loss"])
    for epoch in range(epochs):
        writer.writerow([
            epoch + 1,
            f"{results['engine'][0][epoch]:.6f}",
            f"{results['pytorch'][0][epoch]:.6f}",
        ])

with open("lab05/results/moons_timing.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["framework", "final_loss", "accuracy", "median_training_seconds"])
    for name, (losses, accuracy, seconds) in results.items():
        writer.writerow([name, f"{losses[-1]:.6f}", f"{accuracy:.4f}", f"{seconds:.4f}"])