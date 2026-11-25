# Lab 04 – Leakage-Safe Classification and Regression Pipelines

**Name:** Kalkidan Asdesach
**ID:** UGR/5481/16
**Department:** Software Stream, 4th Year
**Course:** ML and Big Data

## Overview
Two experiments on synthetic support-ticket data, built with scikit-learn pipelines so that preprocessing never leaks information from validation or test data:
- **Experiment A (classification):** predict `escalated`. Logistic regression, k-NN and decision tree compared with stratified 5-fold CV (F1), tuned with GridSearchCV, test set evaluated once.
- **Experiment B (regression):** predict `resolution_time_hours`. Linear regression, ridge and decision tree compared with 5-fold CV (MAE), tuned, test set evaluated once.

## Results
| Experiment | Selected model | CV score | Test score |
|---|---|---|---|
| A – classification | Decision tree (max_depth=6, min_samples_leaf=20) | F1 0.435 ± 0.102 | F1 0.333 (precision 0.50, recall 0.25) |
| B – regression | Ridge (alpha=0.01) | MAE 2.034 ± 0.096 h | MAE 2.14 h, RMSE 3.27 h |

## Files
- `lab_pipelines.ipynb` – full notebook with executed outputs and the written report
- `data/classification_escalation.csv`, `data/resolution_time_regression.csv` – datasets

## How to run
1. `python -m pip install pandas numpy matplotlib scikit-learn ipykernel`
2. Open `lab_pipelines.ipynb` in VS Code (keep the `data` folder next to it), select a Python kernel and run all cells.