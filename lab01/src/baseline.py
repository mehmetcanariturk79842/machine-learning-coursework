import csv
import random
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

accuracy = logistic_model.score(X_test, y_test)
print(f"Logistic Regression accuracy: {accuracy:.4f}")

forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

forest_accuracy = forest_model.score(X_test, y_test)
print(f"Random Forest accuracy: {forest_accuracy:.4f}")

with open("lab01/results/baseline_accuracy.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "test_accuracy"])
    writer.writerow(["Logistic Regression", f"{accuracy:.4f}"])
    writer.writerow(["Random Forest", f"{forest_accuracy:.4f}"])