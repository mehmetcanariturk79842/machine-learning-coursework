import random

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

schema = {
    "age": "int",
    "workclass": "category",
    "fnlwgt": "int",
    "education": "category",
    "education-num": "int",
    "marital-status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "sex": "category",
    "capital-gain": "int",
    "capital-loss": "int",
    "hours-per-week": "int",
    "native-country": "category",
    "class": "category",
}

ranges = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

data = fetch_openml(name="adult", version=2, as_frame=True).frame
report = []

missing_columns = [column for column in schema if column not in data.columns]
report.append(f"Rows checked: {len(data)}")
report.append(f"Missing columns: {missing_columns}")

report.append("\nType check")
type_violations = 0
for column, expected_type in schema.items():
    if expected_type == "int":
        correct = pd.api.types.is_integer_dtype(data[column])
    else:
        correct = isinstance(data[column].dtype, pd.CategoricalDtype)

    if not correct:
        type_violations += 1

    status = "OK" if correct else "VIOLATION"
    report.append(
        f"{column}: expected {expected_type}, "
        f"found {data[column].dtype} -> {status}"
    )

report.append("\nRange check")
range_violations = 0
for column, (low, high) in ranges.items():
    outside = ((data[column] < low) | (data[column] > high)).sum()
    range_violations += outside
    report.append(
        f"{column}: allowed {low}-{high}, "
        f"found {data[column].min()}-{data[column].max()}, "
        f"violations={outside}"
    )

report.append("\nMissing values")
for column in schema:
    nan_count = data[column].isna().sum()
    question_count = (data[column].astype(str).str.strip() == "?").sum()
    report.append(f"{column}: NaN={nan_count}, '?'={question_count}")

report.append("\nDuplicates")
report.append(f"Exact duplicate rows: {data.duplicated().sum()}")

report.append("\nSummary")
report.append(f"Type violations: {type_violations}")
report.append(f"Range violations: {range_violations}")
report.append("Rows dropped: 0")

print("\n".join(report))

with open("lab04/results/validation_report.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(report) + "\n")