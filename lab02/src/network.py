import numpy as np
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

    return probabilities

if __name__ == "__main__":
    np.random.seed(42)
    X, _ = load_digits(return_X_y=True)
    X = X / 16.0
    batch = X[:32]

    for width in [16, 64]:
        W1, b1, W2, b2, W3, b3 = init_params(width)

        print(f"Width: {width}")
        print("W1:", W1.shape, "b1:", b1.shape)
        print("W2:", W2.shape, "b2:", b2.shape)
        print("W3:", W3.shape, "b3:", b3.shape)
        params = (W1, b1, W2, b2, W3, b3)
        probabilities = forward(batch, params)

        print("Output shape:", probabilities.shape)
        print("Row sums close to 1:", np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6
        ))