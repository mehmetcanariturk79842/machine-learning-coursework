# Lab 1: Environment and First System Measurements

## Goal

The goal of this lab is to set up the Python environment, train two
baseline models, and measure their accuracy and system costs.

## Method

- **Environment:** Python 3.11.8 in a virtual environment. The package
  versions are listed in `results/versions.txt`. PyArrow 15.0.2 was used.
- **Data:** Breast Cancer Wisconsin dataset, split into 398 training and
  171 test samples. The split was stratified with `random_state=42`.
- **Models:** Logistic Regression with `max_iter=1000` and Random Forest
  with 100 trees. Both used `random_state=42`.
- **Seeds:** Python, NumPy, PyTorch and CUDA seeds were set to 42.
- **Training time:** One warm-up run, followed by five timed runs. The
  median is reported in seconds.
- **Inference time:** One warm-up prediction, followed by 100 timed
  single-sample predictions. The median is reported in milliseconds.
- **Model size:** Each model was saved separately using joblib. File
  size was recorded in bytes and KiB.
- **Memory:** Peak Python process memory was sampled with
  `memory_profiler` during training and 100 inference calls.

## Results

| Measurement | Logistic Regression | Random Forest |
|---|---:|---:|
| Test accuracy | 0.9474 | 0.9357 |
| Median training time (s) | 0.227856 | 0.080324 |
| Median inference time (ms) | 0.023300 | 1.050800 |
| Model size (bytes) | 1,055 | 290,889 |
| Model size (KiB) | 1.03 | 284.07 |
| Peak training memory (MiB) | 229.18 | 229.87 |
| Peak inference memory (MiB) | 229.18 | 230.05 |

## Deployment fit

The table compares the measurements on my computer with the resource
budgets in the lab manual. Actual performance may differ on each device.

| Target | Budget: memory / latency / model size | Logistic Regression | Random Forest |
|---|---|---|---|
| Cloud | ≥ 1 GB / ≤ 100 ms / ≤ 500 MB | Within measured limits | Within measured limits |
| Edge | 256–1024 MB / ≤ 50 ms / ≤ 50 MB | Within measured limits | Within measured limits |
| Mobile | 64–256 MB / ≤ 20 ms / ≤ 10 MB | Close to memory limit | Close to memory limit |
| TinyML | ≤ 256 KB / ≤ 10 ms / ≤ 100 KB | File size fits; measured process memory exceeds limit | File size and measured process memory exceed limits |

Both models had inference times below 20 ms and file sizes below 10 MB
on this computer. Their measured peak process memory was about
229–230 MiB, including Python and the loaded libraries. This is below
the upper Mobile budget, but leaves little room for other software.

The Logistic Regression file was 1.03 KiB, below the TinyML file-size
limit. The Random Forest file was 284.07 KiB, above that limit. Neither
measured Python process fit the TinyML memory budget.

These results do not prove that either model will run within the latency
or memory limits on a target device. The measurements should be repeated
on the intended hardware.

## Conclusions

1. Logistic Regression had a test accuracy of 0.9474, while Random
   Forest had an accuracy of 0.9357.

2. In this measurement, Random Forest trained faster. However, Logistic
   Regression made predictions faster and had a smaller file size.

3. The measured Python process memory exceeded the TinyML limit for
   both models. The Random Forest file also exceeded the model size limit.

## Limitations

Logistic Regression reached the 1,000-iteration limit without
converging. Its accuracy should be considered with this warning in mind.

The memory results include the Python process and loaded libraries.
Both models were measured sequentially in the same process, and
sampling may miss short memory peaks.

Training and inference times depend on the computer and its workload.
Measurements on this computer do not guarantee the same performance on
a target device.