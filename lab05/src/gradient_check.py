import random

import numpy as np
import torch

from engine import Value, value_sum

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

size = 5
a_data = np.random.randn(size)
b_data = np.random.randn(size)
c_data = np.random.randn(size)


def function(a, b, c):
    # ((a*b + c)**2).relu().sum() written with plain numbers.
    return sum(max(0.0, (a[i] * b[i] + c[i]) ** 2) for i in range(size))


# 1) My engine
a = [Value(x) for x in a_data]
b = [Value(x) for x in b_data]
c = [Value(x) for x in c_data]
output = value_sum(((a[i] * b[i] + c[i]) ** 2).relu() for i in range(size))
output.backward()

engine_grads = {
    "a": np.array([x.grad for x in a]),
    "b": np.array([x.grad for x in b]),
    "c": np.array([x.grad for x in c]),
}

# 2) PyTorch autograd
a_t = torch.tensor(a_data, dtype=torch.float64, requires_grad=True)
b_t = torch.tensor(b_data, dtype=torch.float64, requires_grad=True)
c_t = torch.tensor(c_data, dtype=torch.float64, requires_grad=True)
torch_output = ((a_t * b_t + c_t) ** 2).relu().sum()
torch_output.backward()

torch_grads = {
    "a": a_t.grad.numpy(),
    "b": b_t.grad.numpy(),
    "c": c_t.grad.numpy(),
}

# 3) Numerical gradients (central difference)
epsilon = 1e-6
inputs = {"a": a_data, "b": b_data, "c": c_data}
numerical_grads = {}

for name, values in inputs.items():
    grads = np.zeros(size)
    for i in range(size):
        original = values[i]
        values[i] = original + epsilon
        plus = function(a_data, b_data, c_data)
        values[i] = original - epsilon
        minus = function(a_data, b_data, c_data)
        values[i] = original
        grads[i] = (plus - minus) / (2 * epsilon)
    numerical_grads[name] = grads

report = []
report.append(f"Engine output: {output.data:.10f}")
report.append(f"PyTorch output: {torch_output.item():.10f}")

max_torch_difference = 0.0
max_numerical_difference = 0.0

for name in ["a", "b", "c"]:
    torch_difference = np.max(np.abs(engine_grads[name] - torch_grads[name]))
    numerical_difference = np.max(np.abs(engine_grads[name] - numerical_grads[name]))
    max_torch_difference = max(max_torch_difference, torch_difference)
    max_numerical_difference = max(max_numerical_difference, numerical_difference)

    report.append(f"\nGradient of {name}")
    report.append(f"Engine:    {np.round(engine_grads[name], 8)}")
    report.append(f"PyTorch:   {np.round(torch_grads[name], 8)}")
    report.append(f"Numerical: {np.round(numerical_grads[name], 8)}")

report.append(f"\nMax absolute difference, engine vs PyTorch: {max_torch_difference:.2e}")
report.append(f"Max absolute difference, engine vs numerical: {max_numerical_difference:.2e}")
report.append(f"PyTorch check passed (< 1e-6): {max_torch_difference < 1e-6}")
report.append(f"Numerical check passed (< 1e-6): {max_numerical_difference < 1e-6}")

print("\n".join(report))

with open("lab05/results/gradient_check.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(report) + "\n")