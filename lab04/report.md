# Lab 4: Data Pipeline and Quality

## Goal

The goal of this lab is to build a reproducible, leakage-free data
pipeline for the UCI Adult dataset, check the quality of the data
and labels, and document the dataset.

## Problem definition

The task is to predict whether a person earns more than 50K per year
from census information. This is a binary classification problem.

- **Inputs:** 14 features. Six are numerical (age, fnlwgt,
  education-num, capital-gain, capital-loss, hours-per-week) and
  eight are categorical (workclass, education, marital-status,
  occupation, relationship, race, sex, native-country).
- **Output:** The income class, `<=50K` or `>50K`.
- **Success metrics:** F1 score and accuracy on the test set.
  Only about 24% of the rows are `>50K`, so accuracy alone can look
  good even when the model misses many positive cases. F1 shows how
  well the `>50K` class is predicted.
- **Constraints:** The pipeline must not leak information from the
  test set into training. All statistics used for cleaning must be
  learned from the training set only. The pipeline must produce the
  same processed data every time it is run.

## Method

- **Environment:** Python 3.11.8 in a virtual environment. Package
  versions are in `results/versions.txt`.
- **Seeds:** Python, NumPy, PyTorch and CUDA seeds were set to 42.
- **Ingestion:** The raw data was loaded with `fetch_openml` and was
  not modified. The row and column counts were logged.
- **Validation:** I defined a schema with the expected type of each
  column, value ranges for age, hours-per-week, capital-gain and
  capital-loss, and a missing-value policy. The checks only report
  violations and do not drop rows.
- **Cleaning:** The data was split 70/30, stratified by `class`,
  with `random_state=42` before any cleaning. Exact duplicates were
  dropped from the training set only. Missing values were imputed
  with the median for numerical columns and the most frequent value
  for categorical columns. Numerical columns were clipped to the 1st
  and 99th percentiles. The imputers and clipping bounds were fitted
  on the training set and then applied to both sets. The fitted
  transformer is saved in `results/preprocessor.joblib`.
- **Derived features:** `capital_net = capital-gain - capital-loss`
  and `hours_per_age = hours-per-week / (age - 16)`. Both are
  calculated from values in the same row, so they do not use
  information from other rows or from the test set.
- **Baseline:** Logistic Regression with `max_iter=1000`. Numerical
  columns were standardised and categorical columns were one-hot
  encoded, both fitted on the training set.
- **Label quality:** I searched for rows with identical features but
  different labels. I also checked 50 random rows against the data
  dictionary for impossible values, and looked through them by hand.
- **Reproducibility:** The SHA256 hashes of the processed train and
  test files are stored with the pipeline settings in
  `results/config.yaml`. The whole pipeline runs with one command:
  `python lab04/src/run_pipeline.py`.

## Results

### Validation

| Check | Result |
|---|---:|
| Rows checked | 48,842 |
| Type violations | 0 |
| Range violations | 0 |
| Missing values in `workclass` | 2,799 |
| Missing values in `occupation` | 2,809 |
| Missing values in `native-country` | 857 |
| Exact duplicate rows | 52 |
| Rows dropped | 0 |

In this version of the dataset, missing values are stored as NaN.
No `'?'` values were found. The full report is in
`results/validation_report.txt`.

### Cleaning

| Measurement | Train | Test |
|---|---:|---:|
| Rows after split | 34,189 | 14,653 |
| Missing values before cleaning | 4,452 | 2,013 |
| Duplicates dropped | 24 | 0 |
| Rows after cleaning | 34,165 | 14,653 |
| Missing values after cleaning | 0 | 0 |

Clipping bounds learned from the training set:

| Feature | 1st percentile | 99th percentile |
|---|---:|---:|
| age | 17.00 | 74.00 |
| fnlwgt | 27,035.20 | 509,629.00 |
| education-num | 3.00 | 16.00 |
| capital-gain | 0.00 | 15,024.00 |
| capital-loss | 0.00 | 1,980.00 |
| hours-per-week | 8.00 | 80.00 |

### Derived features

| Setting | Test accuracy | Test F1 |
|---|---:|---:|
| Without derived features | 0.8540 | 0.6601 |
| With derived features | 0.8578 | 0.6696 |

### Label quality

- **Duplicate-conflict audit:** 5 groups of rows had identical
  features but different income labels. These groups contain
  10 rows in total.
- **Re-check of 50 random rows:** No impossible values were found.
  Three rows had missing values. Some rows were unusual but possible,
  for example a person working 4 hours per week and a married person
  with the relationship `Own-child`.

### Reproducibility

| File | SHA256 |
|---|---|
| `adult_train.csv` | `ce9f7a8654c80b90388c923c3e5a2b501a94cd76c9b561f9e9b7cce43e4fdf08` |
| `adult_test.csv` | `3c2857cffd9adcb4ecacbfa4983406ccd38d126f58d7bb0f4608524763cd86b8` |

Running the full pipeline again produced the same hashes, and Git
showed no changes in the result files.

The dataset is described in `data_card.md`.

## Conclusions

1. The raw data passed all type and range checks. The main quality
   problems were missing values in three categorical columns and
   52 exact duplicate rows.

2. Splitting before cleaning and fitting all cleaning steps on the
   training set kept test information out of training. The test set
   kept all of its rows.

3. The two derived features gave a small improvement in this
   experiment: accuracy increased from 0.8540 to 0.8578 and F1 from
   0.6601 to 0.6696.

## Limitations

The 10 rows with conflicting labels were reported but not corrected,
because the correct label is not known. The 50-row check can only
find clear errors, not wrong labels that look realistic. The hashes
were computed on Windows; the same files may get different hashes on
another operating system because of different line endings.