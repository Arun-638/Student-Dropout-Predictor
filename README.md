# Student Dropout Prediction

A machine learning pipeline that predicts whether a student will
**Continue** or **Dropout**, built on a 50,000-row academic dataset
(`Machine_Learning_Task_Database.csv`).

## Overview

The raw dataset contains 41 columns, many of which are either pure
noise (`Random_Code`, `Lucky_Number`, `Random_Flag`, etc.) or
plausible-looking but non-predictive fields (`Favourite_Color`,
`City`, `Device`, `Music_Preference`, `Gender`, `Income`, ...).
Instead of using every available column, each feature was evaluated
against the target — correlation for numeric columns, group-wise
dropout rate for categorical ones — and only the columns showing
real signal were kept.

## Pipeline

1. **Feature selection** — drop 27 non-predictive columns
   (`DROP_FEATURES`), keep 13 (`FEATURES`) based on evidence, not
   guesswork
2. **Data cleaning** — strip `%` from `Attendance`, clean and encode
   `Academic_Warning`, fill missing values (mean for numeric, mode
   for categorical)
3. **Train/test split** — 80/20
4. **Class imbalance handling** — target is ~64% Dropout / 36%
   Continue, addressed via `scale_pos_weight` (XGBoost) and
   `class_weight='balanced'` (Random Forest)
5. **Model training** — XGBoost and Random Forest trained and
   compared
6. **Evaluation** — accuracy, classification report, confusion
   matrix (not accuracy alone, given the class imbalance)
7. **Inference** — predict on new/unseen single-student records

## Features used

```
Attendance, Previous_Score, Absences, Assignments_Completed,
Stress_Level, Previous_Failures, Parent_Support, Financial_Stress,
Library_Visits, Mock_Test_Score, Mentor_Meetings,
Assignment_Submission_Rate, Academic_Warning
```

## Results

| Model | Accuracy | Continue recall | Dropout recall |
|---|---|---|---|
| XGBoost (class-weighted) | 0.618 | 0.61 | 0.62 |
| Random Forest (class-weighted) | 0.598 | 0.60 | 0.60 |

Both models converge on similar performance, which is expected: the
strongest individual feature correlates with the target at only
|r| ≈ 0.12, so ~60–62% accuracy reflects a genuine limit in the
available features rather than a modeling shortfall. Accuracy alone
is not the primary metric here — per-class recall matters more given
the imbalance, and both models were tuned to avoid defaulting to the
majority class.

## Example predictions

An at-risk profile (low attendance, low scores, high stress, past
failures) is correctly predicted as high-risk:
```
Predicted Outcome: Dropout
Dropout Probability: 0.881
```

A strong-performing profile (high attendance, high scores, low
stress, no failures) is correctly predicted as low-risk, confirming
the model responds sensibly to input rather than defaulting to one
class.

## Tech stack

- pandas, numpy
- scikit-learn (`train_test_split`, `LabelEncoder`, metrics)
- XGBoost
- Random Forest (scikit-learn)

## Limitations & future work

Performance is capped by the predictive strength of the available
features. Time-series or trend-based features (attendance trend over
the term, GPA trajectory, LMS engagement, fee payment history) would
likely improve results more than further model tuning on the current
feature set.
