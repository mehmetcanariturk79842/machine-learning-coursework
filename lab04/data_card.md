# Data Card: UCI Adult (Census Income)

## Source

- **Dataset:** UCI Adult / Census Income, OpenML dataset 1590.
- **Loaded with:** `fetch_openml(name="adult", version=2, as_frame=True)`.
- **Origin:** Extracted from the 1994 US Census database.
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0).

## Size

- **Raw data:** 48,842 rows and 15 columns (14 features and 1 label).
- **Features:** 6 numerical and 8 categorical.
- **Label:** `class`, income `<=50K` or `>50K`.
- **Processed train:** 34,165 rows, after removing 24 exact duplicates.
- **Processed test:** 14,653 rows.
- **SHA256 hashes:** stored in `results/config.yaml`.

## Missing values

Missing values appear only in `workclass` (2,799), `occupation` (2,809)
and `native-country` (857). They were imputed with the most frequent
value learned from the training set.

## Known biases

- **Class imbalance:** 76.1% of rows are `<=50K` and 23.9% are `>50K`.
- **Sex:** 66.8% of rows are male and 33.2% are female.
- **Race:** 85.5% of rows are White. Other groups are small,
  for example Amer-Indian-Eskimo is 1.0%.
- **Country:** 89.7% of rows are from the United States.
- **Age of data:** The data is from 1994. The 50K threshold and
  income patterns may not match the present day.
- **Label noise:** 5 groups (10 rows) have identical features but
  different income labels.

## Privacy concerns

The data does not contain names or addresses. However, it contains
sensitive attributes such as race, sex, age and native country.
Combinations of these attributes may identify a person in a small
group. A model trained on this data may also learn unfair patterns
related to sex or race.