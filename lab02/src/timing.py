import csv
import random
import time
from statistics import median

import numpy as np
import torch

from prepare_data import X_train, Y_train
from network import init_params, forward, backward

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

X_batch = X_train[:32]
Y_batch = Y_train[:32]

rows = []

for width in [16, 64]:
    params = init_params(width)

    # Warm up both operations.
    _, cache = forward(X_batch, params)
    backward(params, cache, Y_batch)

    forward_times = []
    for _ in range(100):
        start = time.perf_counter()
        _, cache = forward(X_batch, params)
        forward_times.append((time.perf_counter() - start) * 1000)

    backward_times = []
    for _ in range(100):
        start = time.perf_counter()
        backward(params, cache, Y_batch)
        backward_times.append((time.perf_counter() - start) * 1000)

    forward_median = median(forward_times)
    backward_median = median(backward_times)

    rows.append([width, forward_median, backward_median])
    print(
        f"Width {width}: forward={forward_median:.6f} ms, "
        f"backward={backward_median:.6f} ms"
    )

with open("lab02/results/timing.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["width", "forward_median_ms", "backward_median_ms"])
    writer.writerows(rows)