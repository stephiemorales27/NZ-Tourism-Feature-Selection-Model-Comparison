# NZ-Tourism-Feature-Selection-Model-Comparison
An independently developed regression case study predicting total visitor spend in New Zealand's tourism industry, covering the full applied machine learning workflow from business understanding to model optimization and evaluation.

## Overview

This project was completed as Task 1 for the "Implementación de IA en Caso de Negocio Real" (AI Implementation in a Real Business Case) course. Unlike guided course exercises, the dataset for this project was self-selected — I chose a New Zealand tourism dataset and independently designed and implemented the full pipeline in Databricks.

## Note on Code Attribution

The dataset and case topic were self-selected as part of a course assignment that required choosing a real-world dataset and business problem. The full pipeline — business framing, EDA, data cleaning, feature engineering, feature selection, modeling, and evaluation — was independently designed and implemented by me.

## Dataset

This project uses the [NZ Tourism Dataset 2025](https://www.kaggle.com/datasets/digitalashish/nz-tourism-dataset-2024/data) from Kaggle, originally sourced from Stats NZ / data.govt.nz tourism forecast data. The dataset has 644 rows, of which 314 have a value for the target variable. The raw dataset is not included in this repository — download it directly from Kaggle to reproduce the analysis.

## Business Problem

New Zealand's tourism industry is a major contributor to the national economy. This project builds a regression model to predict Total Visitor Spend using tourism metrics (visitor arrivals by purpose, length of stay, country of origin), and identifies which features are most predictive — supporting use cases like marketing budget allocation, infrastructure planning, and economic forecasting.

## Key Highlights

* **Identified and removed target leakage:** `Spend Per Day` is a direct mathematical component of the target variable (`Total Visitor Spend = Spend Per Day × Total Visitor Days`), so it and other target-derived columns were excluded from modeling.
* **Feature selection via method consensus:** Combined Lasso regression, Random Forest feature importance, and Recursive Feature Elimination (RFE), retaining only features selected by at least 2 of the 3 methods — reducing the feature space from 31 to 9 features (a 71% reduction).
* **Temporal validation:** Used a time-based train/test split (rather than random) to simulate realistic forecasting conditions, training on years up to 2018 (242 rows) and testing on later years (72 rows).
* **Model comparison:** Compared a Linear Regression baseline (31 features) against Random Forest and Gradient Boosting models trained on the 9 selected features. The baseline had the best test performance (R² = 0.994); Gradient Boosting reached R² ≈ 0.95 and Random Forest R² ≈ 0.94 with far fewer features, at the cost of a small drop in accuracy.
* **Experiment tracking:** Logged models, parameters, and metrics using MLflow for reproducibility.

## Results (test set, years after 2018)

| Model | Features | R² | RMSE | MAE |
| --- | --- | --- | --- | --- |
| Linear Regression (baseline) | 31 | 0.994 | ~261M | ~176M |
| Random Forest | 9 | 0.942 | ~826M | ~377M |
| Gradient Boosting | 9 | 0.950 | ~771M | ~379M |

## Limitations

Small dataset (314 rows with target) and a single temporal split, so results may change with a different cutoff year.

## Technologies

* Python (Pandas, NumPy, Matplotlib)
* Scikit-learn (Pipelines, LinearRegression, RandomForestRegressor, GradientBoostingRegressor, LassoCV, RFE)
* Databricks
* MLflow

## Files

* `NZ_Tourism_Feature_Selection_Analysis.ipynb` — full notebook with outputs

## Author

Stephanie Morales
[LinkedIn Profile](https://linkedin.com/in/smoralesvillalobos)

