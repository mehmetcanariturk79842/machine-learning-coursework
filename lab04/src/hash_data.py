import hashlib
from pathlib import Path

import yaml


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


files = {
    "train": Path("data/processed/adult_train.csv"),
    "test": Path("data/processed/adult_test.csv"),
}

config = {
    "dataset": {
        "name": "adult",
        "source": "fetch_openml(name='adult', version=2, as_frame=True)",
        "raw_rows": 48842,
        "raw_columns": 15,
    },
    "split": {
        "test_size": 0.30,
        "stratify": "class",
        "random_state": 42,
    },
    "cleaning": {
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "clip_percentiles": [1, 99],
        "drop_duplicates": "train only",
    },
    "features": [
        "capital_net = capital-gain - capital-loss",
        "hours_per_age = hours-per-week / (age - 16)",
    ],
    "seed": 42,
    "processed_data": {},
}

for name, path in files.items():
    file_hash = sha256_of_file(path)
    config["processed_data"][name] = {
        "path": path.as_posix(),
        "sha256": file_hash,
    }
    print(f"{name}: {file_hash}")

with open("lab04/results/config.yaml", "w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, sort_keys=False)

print("Saved: lab04/results/config.yaml")