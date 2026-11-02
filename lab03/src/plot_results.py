import csv

import matplotlib.pyplot as plt

from models import MLP, CNN, AttentionModel


model_classes = [MLP, CNN, AttentionModel]
model_names = [model_class.__name__ for model_class in model_classes]

parameter_counts = {
    model_class.__name__: sum(
        parameter.numel() for parameter in model_class().parameters()
    )
    for model_class in model_classes
}

with open("lab03/results/test_accuracy.csv", newline="") as file:
    accuracy_rows = list(csv.DictReader(file))

accuracies = {
    row["model"]: float(row["test_accuracy"])
    for row in accuracy_rows
}

with open("lab03/results/latency.csv", newline="") as file:
    latency_rows = list(csv.DictReader(file))

latencies = {
    row["model"]: float(row["median_ms"])
    for row in latency_rows
    if int(row["batch_size"]) == 1
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name in model_names:
    accuracy = accuracies[name] * 100

    axes[0].scatter(parameter_counts[name], accuracy, label=name)
    axes[1].scatter(latencies[name], accuracy, label=name)

axes[0].set_xlabel("Parameters")
axes[0].set_ylabel("Test accuracy (%)")
axes[0].set_title("Accuracy vs parameters")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_xlabel("Median latency for batch 1 (ms)")
axes[1].set_ylabel("Test accuracy (%)")
axes[1].set_title("Accuracy vs latency")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("lab03/results/tradeoffs.png", dpi=150)
print("Saved: lab03/results/tradeoffs.png")