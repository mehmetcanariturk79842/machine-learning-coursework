# Machine Learning Coursework

This repository contains my machine learning lab work.

## Environment

The labs use Python 3.11.8 and the packages listed in `requirements.txt`.
To install them in an activated virtual environment:

```powershell
python -m pip install -r requirements.txt
```

## Lab 1

Lab 1 compares Logistic Regression and Random Forest on the Breast
Cancer Wisconsin dataset. The report and results are in `lab01/`.

Run the scripts from the project root:

```powershell
python lab01/src/print_versions.py
python lab01/src/baseline.py
python lab01/src/measure_system.py
python lab01/src/measure_memory.py
```

## Lab 2

Lab 2 builds a two-hidden-layer neural network from scratch using the
digits dataset. The report and results are in `lab02/`.

Run the scripts from the project root:

```powershell
python lab02/src/prepare_data.py
python lab02/src/network.py
python lab02/src/gradient_check.py
python lab02/src/train.py
python lab02/src/plot_loss.py
python lab02/src/timing.py
```

## Lab 3

Lab 3 compares MLP, CNN and attention models on Fashion-MNIST.
The report and results are in `lab03/`.

Run the scripts from the project root:

```powershell
python lab03/src/data_setup.py
python lab03/src/check_models.py
python lab03/src/flop_check.py
python lab03/src/train.py
python lab03/src/latency.py
python lab03/src/plot_results.py
```