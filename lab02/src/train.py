import csv
import random

import numpy as np
import torch

from prepare_data import X_train, X_test, y_test, Y_train
from network import init_params, forward, cross_entropy, backward

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

batch_size = 32
epochs = 50
widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]

training_rows = []
accuracy_rows = []

for width in widths:
    for learning_rate in learning_rates:
        # Use the same seed at the start of each experiment.
        np.random.seed(42)
        params = init_params(width)

        print(f"\nWidth: {width}, learning rate: {learning_rate}")

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(len(X_train))

            for start in range(0, len(X_train), batch_size):
                batch_indices = indices[start:start + batch_size]
                X_batch = X_train[batch_indices]
                Y_batch = Y_train[batch_indices]

                probabilities, cache = forward(X_batch, params)
                gradients = backward(params, cache, Y_batch)

                for param, gradient in zip(params, gradients):
                    param -= learning_rate * gradient

            train_probs, _ = forward(X_train, params)
            train_loss = cross_entropy(train_probs, Y_train)

            training_rows.append([width, learning_rate, epoch, train_loss])
            print(f"Epoch {epoch}: training loss = {train_loss:.4f}")

        test_probs, _ = forward(X_test, params)
        predictions = np.argmax(test_probs, axis=1)
        test_accuracy = np.mean(predictions == y_test)

        accuracy_rows.append([width, learning_rate, test_accuracy])
        print(f"Final test accuracy: {test_accuracy:.4f}")

with open("lab02/results/training_log.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["width", "learning_rate", "epoch", "training_loss"])
    writer.writerows(training_rows)

with open("lab02/results/final_accuracy.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["width", "learning_rate", "test_accuracy"])
    for width, learning_rate, accuracy in accuracy_rows:
        writer.writerow([width, learning_rate, f"{accuracy:.4f}"])