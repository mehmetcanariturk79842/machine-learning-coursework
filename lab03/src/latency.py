import csv
import random
import time
from statistics import median

import numpy as np
import torch

from models import MLP, CNN, AttentionModel


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cpu")
    results = []

    for model_class in [MLP, CNN, AttentionModel]:
        model = model_class().to(device)
        model.eval()
        name = model_class.__name__

        for batch_size, runs in [(1, 100), (128, 20)]:
            sample = torch.randn(batch_size, 1, 28, 28, device=device)
            times = []

            with torch.inference_mode():
                model(sample)

                for _ in range(runs):
                    start = time.perf_counter()
                    model(sample)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    times.append(elapsed_ms)

            median_ms = median(times)
            results.append([name, batch_size, runs, f"{median_ms:.6f}"])

            print(
                f"{name}, batch={batch_size}: "
                f"median latency={median_ms:.6f} ms"
            )

    with open("lab03/results/latency.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "batch_size", "runs", "median_ms"])
        writer.writerows(results)

    print("Saved: lab03/results/latency.csv")