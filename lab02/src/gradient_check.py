import numpy as np
import random
import torch
from sklearn.datasets import load_digits

from network import init_params, forward, cross_entropy, backward

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

X, y = load_digits(return_X_y=True)
X = X / 16.0

batch = X[:32]
labels = np.eye(10)[y[:32]]

epsilon = 1e-5
tolerance = 1e-6

for width in [16, 64]:
    params = init_params(width)
    probabilities, cache = forward(batch, params)
    gradients = backward(params, cache, labels)

    passed = 0
    print(f"\nWidth: {width}")

    for check in range(10):
        layer = np.random.choice([0, 2, 4])
        weights = params[layer]
        index = tuple(np.random.randint(size) for size in weights.shape)
        original = weights[index]

        weights[index] = original + epsilon
        plus_probs, _ = forward(batch, params)
        loss_plus = cross_entropy(plus_probs, labels)

        weights[index] = original - epsilon
        minus_probs, _ = forward(batch, params)
        loss_minus = cross_entropy(minus_probs, labels)

        weights[index] = original

        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        analytical = gradients[layer][index]

        denominator = max(abs(analytical), abs(numerical))
        if denominator == 0:
            relative_error = 0.0
        else:
            relative_error = abs(analytical - numerical) / denominator

        if relative_error < tolerance:
            passed += 1

        print(
        f"Check {check + 1}: layer={layer // 2 + 1}, index={index}, "
        f"analytic={analytical:.8e}, numeric={numerical:.8e}, "
        f"error={relative_error:.2e}"
        )

    print(f"Passed: {passed}/10")