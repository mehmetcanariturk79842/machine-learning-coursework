import random

import numpy as np
import torch
from sklearn.datasets import fetch_openml

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

data = fetch_openml(name="adult", version=2, as_frame=True).frame
feature_columns = [column for column in data.columns if column != "class"]

report = []

# Duplicate-conflict audit: same features, different income labels.
labels_per_group = data.groupby(
    feature_columns, observed=True, dropna=False
)["class"].nunique()
conflicting_groups = labels_per_group[labels_per_group > 1]

conflict_rows = data.merge(
    conflicting_groups.reset_index()[feature_columns],
    on=feature_columns,
    how="inner",
)

report.append("Duplicate-conflict audit")
report.append(f"Feature groups with conflicting labels: {len(conflicting_groups)}")
report.append(f"Rows in these groups: {len(conflict_rows)}")

# Re-check 50 random rows against the data dictionary.
sample = data.sample(n=50, random_state=42)
sample.to_csv("lab04/results/label_audit_sample.csv")

problems = []
for index, row in sample.iterrows():
    if not 17 <= row["age"] <= 90:
        problems.append(f"Row {index}: age {row['age']} outside 17-90")
    if not 1 <= row["hours-per-week"] <= 99:
        problems.append(f"Row {index}: hours {row['hours-per-week']} outside 1-99")
    if row["capital-gain"] < 0 or row["capital-loss"] < 0:
        problems.append(f"Row {index}: negative capital value")
    if row["relationship"] == "Husband" and row["sex"] == "Female":
        problems.append(f"Row {index}: relationship Husband but sex Female")
    if row["relationship"] == "Wife" and row["sex"] == "Male":
        problems.append(f"Row {index}: relationship Wife but sex Male")
    if row["relationship"] in ["Husband", "Wife"] and row["marital-status"] == "Never-married":
        problems.append(f"Row {index}: spouse relationship but never married")

report.append("\nRe-check of 50 random rows")
report.append(f"Rows checked: {len(sample)}")
report.append(f"Rows with missing values: {sample.isna().any(axis=1).sum()}")
report.append(f"Impossible values found: {len(problems)}")
report.extend(problems)

print("\n".join(report))

with open("lab04/results/label_quality.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(report) + "\n")