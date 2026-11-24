import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from clipper import PercentileClipper

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

data = fetch_openml(name="adult", version=2, as_frame=True).frame

# Split before cleaning.
train, test = train_test_split(
    data, test_size=0.30, stratify=data["class"], random_state=42
)

summary = []
summary.append(f"Train rows after split: {len(train)}")
summary.append(f"Test rows after split: {len(test)}")
summary.append(f"Missing values in train: {train.isna().sum().sum()}")
summary.append(f"Missing values in test: {test.isna().sum().sum()}")

# Drop exact duplicates from train only.
duplicates = train.duplicated().sum()
train = train.drop_duplicates()
summary.append(f"Duplicates dropped from train: {duplicates}")
summary.append(f"Train rows after dropping duplicates: {len(train)}")

numeric_steps = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clipper", PercentileClipper(lower=1, upper=99)),
])
categorical_steps = SimpleImputer(strategy="most_frequent")

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_steps, numeric_columns),
        ("categorical", categorical_steps, categorical_columns),
    ],
    verbose_feature_names_out=False,
)
preprocessor.set_output(transform="pandas")

# Fit on train only, then apply to both sets.
train_clean = preprocessor.fit_transform(train)
test_clean = preprocessor.transform(test)

train_clean["class"] = train["class"]
test_clean["class"] = test["class"]

summary.append(f"Missing values in train after cleaning: {train_clean.isna().sum().sum()}")
summary.append(f"Missing values in test after cleaning: {test_clean.isna().sum().sum()}")

clipper = preprocessor.named_transformers_["numeric"].named_steps["clipper"]
summary.append("\nClip bounds from train (1st / 99th percentile)")
for column, low, high in zip(
    numeric_columns, clipper.lower_bounds_, clipper.upper_bounds_
):
    summary.append(f"{column}: {low:.2f} to {high:.2f}")

joblib.dump(preprocessor, "lab04/results/preprocessor.joblib")

output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)
train_clean.to_csv(output_folder / "adult_train.csv", index=False)
test_clean.to_csv(output_folder / "adult_test.csv", index=False)

print("\n".join(summary))

with open("lab04/results/cleaning_summary.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(summary) + "\n")