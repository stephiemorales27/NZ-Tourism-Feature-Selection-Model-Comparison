# NZ-Tourism-Feature-Selection-Model-Comparison
An independently developed regression case study predicting total visitor spend in New Zealand's tourism industry, covering the full applied machine learning workflow from business understanding to model optimization and evaluation.

## Overview

This project was completed as Task 1 for the "Implementación de IA en Caso de Negocio Real" (AI Implementation in a Real Business Case) course. Unlike guided course exercises, the dataset for this project was self-selected — I chose a New Zealand tourism dataset and independently designed and implemented the full pipeline in Databricks.

## Note on Code Attribution

The dataset and case topic were self-selected as part of a course assignment that required choosing a real-world dataset and business problem. The full pipeline — business framing, EDA, data cleaning, feature engineering, feature selection, modeling, and evaluation — was independently designed and implemented by me.

## Dataset

This project uses the [NZ Tourism Dataset 2025](https://www.kaggle.com/datasets/digitalashish/nz-tourism-dataset-2024/data) from Kaggle, originally sourced from Stats NZ / data.govt.nz tourism forecast data. The raw dataset is not included in this repository — download it directly from Kaggle to reproduce the analysis.

## Business Problem

New Zealand's tourism industry is a major contributor to the national economy. This project builds a regression model to predict **Total Visitor Spend** using tourism metrics (visitor arrivals by purpose, length of stay, country of origin), and identifies which features are most predictive — supporting use cases like marketing budget allocation, infrastructure planning, and economic forecasting.

## Key Highlights

- **Identified and removed target leakage**: `Spend Per Day` was found to be a direct mathematical component of the target variable (`Total Visitor Spend = Spend Per Day × Total Visitor Days`) and was excluded from modeling.
- **Feature selection via method consensus**: Combined Lasso regression, Random Forest feature importance, and Recursive Feature Elimination (RFE), retaining only features selected by at least 2 of the 3 methods — reducing the feature space from 31 to 10 features (a 68% reduction).
- **Temporal validation**: Used a time-based train/test split (rather than random) to simulate realistic forecasting conditions, training on earlier years and testing on more recent ones.
- **Model comparison**: Compared a Linear Regression baseline against optimized Random Forest and Gradient Boosting models, achieving R² > 0.95 with significantly fewer, less leakage-prone features.
- **Experiment tracking**: Logged all models, parameters, and metrics using MLflow for reproducibility.

## Technologies

- Python (Pandas, NumPy, Matplotlib)
- Scikit-learn (Pipelines, LinearRegression, RandomForestRegressor, GradientBoostingRegressor, LassoCV, RFE)
- Databricks
- MLflow

## Author

Stephanie Morales
[linkedin.com/in/smoralesvillalobos](https://linkedin.com/in/smoralesvillalobos)
