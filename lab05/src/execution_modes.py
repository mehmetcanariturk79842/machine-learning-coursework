import csv
import random
import time
from statistics import median

import numpy as np
import torch
import torch._dynamo
from torch import nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def median_ms(function, runs=100):
    function()  # warm-up
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return median(times)


model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
model.eval()
report = []

# 1) Inspect the graph of a small model.
graph_module = torch.fx.symbolic_trace(model)
report.append("FX graph of the model")
report.append(str(graph_module.graph))

output = model(torch.randn(4, 2))
report.append("Autograd graph (from output back to the parameters)")
stack = [(output.grad_fn, 1)]
while stack:
    node, depth = stack.pop()
    report.append("  " * depth + node.name())
    for next_node, _ in reversed(node.next_functions):
        if next_node is not None:
            stack.append((next_node, depth + 1))

# 2) Check whether torch.compile is supported here.
explanation = torch._dynamo.explain(model)(torch.randn(4, 2))
compile_supported = explanation.graph_count > 0
report.append(f"\ntorch.compile graphs captured: {explanation.graph_count}")
report.append(f"torch.compile supported on this machine: {compile_supported}")

if compile_supported:
    fast_model = torch.compile(model)
    fast_name = "compiled"
else:
    fast_model = torch.jit.trace(model, torch.randn(1, 2))
    fast_name = "traced"
report.append(f"Mode used for comparison: {fast_name}")

# 3) Eager vs compiled/traced, after warm-up.
timing_rows = []
with torch.no_grad():
    for batch_size in [1, 1000]:
        x = torch.randn(batch_size, 2)
        eager_time = median_ms(lambda: model(x))
        fast_time = median_ms(lambda: fast_model(x))
        timing_rows.append([batch_size, "eager", f"{eager_time:.6f}"])
        timing_rows.append([batch_size, fast_name, f"{fast_time:.6f}"])
        print(
            f"Batch {batch_size}: eager={eager_time:.6f} ms, "
            f"{fast_name}={fast_time:.6f} ms"
        )

# 4) Shape, dtype and contiguity.
x = torch.randn(2000, 2000)
view = x.t()

report.append("\nTensor inspection")
for name, tensor in [("original", x), ("transposed view", view)]:
    report.append(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"stride={tensor.stride()}, contiguous={tensor.is_contiguous()}"
    )
report.append(f"Same memory: {x.data_ptr() == view.data_ptr()}")

contiguity_rows = []
operations = [
    ("contiguous", lambda: x.sum(dim=1)),
    ("non-contiguous view", lambda: view.sum(dim=1)),
    ("view made contiguous first", lambda: view.contiguous().sum(dim=1)),
]
for name, operation in operations:
    elapsed = median_ms(operation)
    contiguity_rows.append([name, f"{elapsed:.6f}"])
    print(f"sum(dim=1) on {name}: {elapsed:.6f} ms")

print("\n".join(report))

with open("lab05/results/graph_inspection.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(report) + "\n")

with open("lab05/results/execution_timing.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["batch_size", "mode", "median_ms"])
    writer.writerows(timing_rows)

with open("lab05/results/contiguity_timing.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["tensor", "median_ms"])
    writer.writerows(contiguity_rows)