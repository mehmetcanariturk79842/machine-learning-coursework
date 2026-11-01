import csv
import random
import numpy as np
import torch
from memory_profiler import memory_usage
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
def predict_samples(model, sample):
    for _ in range(100):
        model.predict(sample)
if __name__ == "__main__":
    rows = []
    for name, model in models.items():
        current_model = clone(model)

        peak_memory = memory_usage(
            (current_model.fit, (X_train, y_train)),
            interval=0.001,
            max_usage=True,
        )

        print(f"{name} training peak memory: {peak_memory:.2f} MiB")
        current_model.predict(X_test[:1])

        inference_peak = memory_usage(
            (predict_samples, (current_model, X_test[:1])),
            interval=0.001,
            max_usage=True,
        )

        print(f"{name} inference peak memory: {inference_peak:.2f} MiB")
        rows.append([name, f"{peak_memory:.2f}", f"{inference_peak:.2f}"])

    with open("lab01/results/memory.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "training_peak_mib", "inference_peak_mib"])
        writer.writerows(rows)