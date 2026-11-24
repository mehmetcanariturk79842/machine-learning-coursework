# Lab 5: Frameworks and Automatic Differentiation

## Goal

The goal of this lab is to build a scalar automatic-differentiation
engine, check it against PyTorch, train a tiny model with it, and
compare eager execution with traced execution and the cost of
non-contiguous tensors.

## Method

- **Environment:** Python 3.11.8 and PyTorch 2.3.1 on CPU. Package
  versions are in `results/versions.txt`.
- **Seeds:** Python, NumPy, PyTorch and CUDA seeds were set to 42.
- **Value node:** Each `Value` stores its data, its gradient, its
  parent nodes and a local derivative rule.
- **Operations:** Addition, multiplication, powers and ReLU create a
  new node during the forward pass, so the computation graph is built
  while the result is calculated. Sigmoid and log were added for the
  BCE loss.
- **Backward pass:** The graph is sorted topologically, the output
  gradient is set to 1, and the nodes are visited in reverse order.
  Gradients are accumulated with `+=`, so a node used more than once
  receives the sum of all its gradients.
- **Gradient check:** The function `((a*b + c)**2).relu().sum()` was
  computed for vectors of 5 random values. Gradients from my engine
  were compared with PyTorch autograd (float64) and with numerical
  central differences using `epsilon=1e-6`. The required maximum
  absolute difference was below `1e-6`.
- **Tiny model:** `make_moons` with 1,000 samples, noise 0.1 and
  `random_state=42`. Architecture 2 → 8 → 1 with ReLU in the hidden
  layer, sigmoid output and BCE loss. Full-batch gradient descent
  with learning rate 0.5 for 50 epochs. Both versions started from
  the same initial weights.
- **Training speed:** One warm-up run, then the median of 5 full
  training runs for each framework.
- **Execution modes:** A small PyTorch model (2 → 8 → 1) was
  inspected with `torch.fx` and through its autograd graph. I first
  checked whether `torch.compile` captures a graph on this machine.
  It captured 0 graphs, so I used `torch.jit.trace` as the manual
  allows. Eager and traced inference were timed after one warm-up,
  using the median of 100 runs for batch 1 and batch 1000.
- **Contiguity:** I created a 2000 × 2000 float32 tensor and its
  transposed view, inspected shape, dtype, stride and contiguity, and
  timed `sum(dim=1)` with the median of 100 runs after one warm-up.

## Results

### Gradient check

| Comparison | Max absolute difference | Passed (< 1e-6) |
|---|---:|---|
| Engine vs PyTorch autograd | 0.00e+00 | Yes |
| Engine vs numerical gradients | 1.05e-09 | Yes |

The engine and PyTorch both gave an output of 11.6879237903.

### Tiny model on make_moons

| Framework | Final loss | Accuracy | Median training time (s) |
|---|---:|---:|---:|
| My engine | 0.255827 | 0.8750 | 39.0316 |
| PyTorch | 0.255827 | 0.8750 | 0.0191 |

The two loss curves matched; the largest difference over the
50 epochs was 1.89e-15. PyTorch was about 2,045 times faster.
My engine creates a Python object for every scalar operation,
while PyTorch processes the whole batch with a few tensor operations
implemented in compiled code.

### Graph inspection

The FX graph showed four module calls in order: Linear, ReLU, Linear
and Sigmoid. The autograd graph went from `SigmoidBackward0` through
`AddmmBackward0` and `ReluBackward0` back to `AccumulateGrad` nodes
for the weights and biases. The full output is in
`results/graph_inspection.txt`.

### Eager vs traced execution

`torch.compile` captured 0 graphs on this machine (Windows, PyTorch
2.3.1), so tracing was used instead.

| Batch size | Eager (ms) | Traced (ms) |
|---:|---:|---:|
| 1 | 0.012900 | 0.009800 |
| 1000 | 0.036800 | 0.032700 |

### Tensor contiguity

| Tensor | Shape | Stride | Contiguous |
|---|---|---|---|
| Original | (2000, 2000) | (2000, 1) | True |
| Transposed view | (2000, 2000) | (1, 2000) | False |

Both tensors are float32 and share the same memory.

| `sum(dim=1)` on | Median time (ms) |
|---|---:|
| Contiguous tensor | 0.419850 |
| Non-contiguous view | 0.480850 |
| View made contiguous first | 4.082600 |

### When compilation helps

Compilation or tracing removes part of the Python overhead of running
each operation separately, and a compiler can fuse operations. This
helps most when a model is small and called many times, as in the
batch 1 result here, where the fixed cost per call is large compared
with the actual computation. It helps less when most of the time is
spent inside large operations such as big matrix multiplications.
Compilation also has a start-up cost, so it is useful only when the
model is run many times, and it may not be available on every
platform.

## Conclusions

1. My autograd engine matched PyTorch autograd exactly and the
   numerical gradients within 1.05e-09, and it trained the make_moons
   model to the same loss and accuracy as PyTorch.

2. PyTorch trained the same model about 2,045 times faster, because
   it works on whole tensors instead of individual Python scalars.

3. Tracing made inference slightly faster in this experiment. A
   non-contiguous view was a little slower to sum, and copying it to
   a contiguous tensor first was about 10 times slower.

## Limitations

The timings depend on this computer and its workload. `torch.compile`
could not be tested because it did not capture graphs on this
machine. The model was evaluated on the training data only, because
the lab asked for a speed comparison rather than a test score.