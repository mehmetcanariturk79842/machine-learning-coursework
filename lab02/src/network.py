import numpy as np
import random
import torch
from sklearn.datasets import load_digits


def init_params(width):
    W1 = np.random.normal(0, 0.1, (64, width))
    b1 = np.zeros(width)

    W2 = np.random.normal(0, 0.1, (width, width))
    b2 = np.zeros(width)

    W3 = np.random.normal(0, 0.1, (width, 10))
    b3 = np.zeros(10)

    return W1, b1, W2, b2, W3, b3

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def forward(X, params):
    W1, b1, W2, b2, W3, b3 = params

    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    probabilities = softmax(z3)

    cache = (X, z1, a1, z2, a2, probabilities)
    return probabilities, cache

def cross_entropy(probabilities, labels):
    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.mean(np.sum(labels * np.log(safe_probabilities), axis=1))

def backward(params, cache, labels):
    W1, b1, W2, b2, W3, b3 = params
    X, z1, a1, z2, a2, probabilities = cache

    N = labels.shape[0]

    dZ3 = (probabilities - labels) / N
    dW3 = a2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (z2 > 0)

    dW2 = a1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return dW1, db1, dW2, db2, dW3, db3

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    batch = X[:32]
    labels = np.eye(10)[y[:32]]
    for width in [16, 64]:
        W1, b1, W2, b2, W3, b3 = init_params(width)

        print(f"Width: {width}")
        print("W1:", W1.shape, "b1:", b1.shape)
        print("W2:", W2.shape, "b2:", b2.shape)
        print("W3:", W3.shape, "b3:", b3.shape)
        params = (W1, b1, W2, b2, W3, b3)
        probabilities, cache = forward(batch, params)

        print("Output shape:", probabilities.shape)
        print("Row sums close to 1:", np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6
        ))
        loss = cross_entropy(probabilities, labels)
        print(f"Untrained loss: {loss:.4f}")
        gradients = backward(params, cache, labels)

        for param, gradient in zip(params, gradients):
            print("Parameter:", param.shape, "Gradient:", gradient.shape)