import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_setup import train_set, validation_set, test_set
from models import MLP, CNN, AttentionModel


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    device = torch.device("cpu")
    batch_size = 128
    epochs = 5
    learning_rate = 1e-3

    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    results = []
    training_log = []

    for model_class in [MLP, CNN, AttentionModel]:
        set_seed()

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )

        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        name = model_class.__name__
        print(f"\nTraining {name} on {device}", flush=True)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

            average_loss = total_loss / total_samples
            validation_accuracy = evaluate(model, validation_loader, device)

            training_log.append([
                name,
                epoch,
                f"{average_loss:.6f}",
                f"{validation_accuracy:.6f}",
            ])

            print(
                f"Epoch {epoch}/{epochs}: "
                f"loss={average_loss:.4f}, "
                f"validation accuracy={validation_accuracy:.4f}",
                flush=True,
            )

        test_accuracy = evaluate(model, test_loader, device)
        print(f"{name} test accuracy: {test_accuracy:.4f}", flush=True)

        results.append([name, f"{test_accuracy:.4f}"])

    with open("lab03/results/training_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "epoch", "training_loss", "validation_accuracy"])
        writer.writerows(training_log)

    with open("lab03/results/test_accuracy.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "test_accuracy"])
        writer.writerows(results)

    print("\nResults saved in lab03/results/", flush=True)