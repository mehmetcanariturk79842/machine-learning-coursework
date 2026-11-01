import csv
import time
import random
import numpy as np
import torch
from pathlib import Path
import joblib
from statistics import median
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
rows = []
for name, model in models.items():
    # Run once before measuring.
    clone(model).fit(X_train, y_train)

    training_times = []
    for _ in range(5):
        current_model = clone(model)
        start = time.perf_counter()
        current_model.fit(X_train, y_train)
        training_times.append(time.perf_counter() - start)

    print(f"{name} training time: {median(training_times):.6f} seconds")
    current_model.predict(X_test[:1])

    inference_times = []
    for _ in range(100):
        start = time.perf_counter()
        current_model.predict(X_test[:1])
        inference_times.append((time.perf_counter() - start) * 1000)

    print(f"{name} inference time: {median(inference_times):.6f} ms")    
    model_name = name.lower().replace(" ", "_")
    model_path = Path("lab01/results") / f"{model_name}.joblib"

    joblib.dump(current_model, model_path)

    size_bytes = model_path.stat().st_size
    size_kb = size_bytes / 1024

    print(f"{name} model size: {size_bytes} bytes ({size_kb:.2f} KiB)")
    rows.append([
        name,
        f"{median(training_times):.6f}",
        f"{median(inference_times):.6f}",
        size_bytes,
        f"{size_kb:.2f}",
    ])
with open("lab01/results/system_metrics.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "model",
        "training_median_seconds",
        "inference_median_ms",
        "model_size_bytes",
        "model_size_kib",
    ])
    writer.writerows(rows)