import csv
import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

numeric_columns = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week",
]
categorical_columns = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
new_columns = ["capital_net", "hours_per_age"]


def add_features(data):
    data = data.copy()
    data["capital_net"] = data["capital-gain"] - data["capital-loss"]
    data["hours_per_age"] = data["hours-per-week"] / (data["age"] - 16)
    return data


def evaluate(train, test, numeric):
    model = Pipeline([
        ("encode", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ])),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    model.fit(train, train["class"] == ">50K")
    predictions = model.predict(test)
    y_test = test["class"] == ">50K"

    return accuracy_score(y_test, predictions), f1_score(y_test, predictions)


train = pd.read_csv("data/processed/adult_train.csv")
test = pd.read_csv("data/processed/adult_test.csv")

train_features = add_features(train)
test_features = add_features(test)

print(train_features[new_columns].describe())

rows = []
accuracy, f1 = evaluate(train, test, numeric_columns)
rows.append(["without derived features", f"{accuracy:.4f}", f"{f1:.4f}"])

accuracy, f1 = evaluate(train_features, test_features, numeric_columns + new_columns)
rows.append(["with derived features", f"{accuracy:.4f}", f"{f1:.4f}"])

for name, accuracy, f1 in rows:
    print(f"{name}: accuracy={accuracy}, F1={f1}")

with open("lab04/results/feature_comparison.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["setting", "test_accuracy", "test_f1"])
    writer.writerows(rows)