import random

import numpy as np
import torch
from sklearn.datasets import fetch_openml

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

adult = fetch_openml(name="adult", version=2, as_frame=True)
data = adult.frame

print("Rows:", data.shape[0])
print("Columns:", data.shape[1])
print("Column names:", list(data.columns))

with open("lab04/results/ingestion.txt", "w", encoding="utf-8") as file:
    file.write(f"Rows: {data.shape[0]}\n")
    file.write(f"Columns: {data.shape[1]}\n")
    file.write(f"Column names: {list(data.columns)}\n")