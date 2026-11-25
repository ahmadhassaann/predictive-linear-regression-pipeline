# Predictive Linear Regression Model Pipeline

A reproducible Python machine-learning pipeline that predicts California median house values using Ridge Regression.

## Overview

The project downloads the California Housing dataset from Scikit-Learn, engineers a longitude-region feature, preprocesses numerical and categorical columns, trains a Ridge Regression model, evaluates held-out predictions, saves the fitted pipeline, and exports a diagnostic plot.

## Project Structure

```text
Predictive Linear Regression Model Pipeline/
├── linear_regression.py
├── requirements.txt
├── README.md
├── models/
│   └── ridge_regression_model.pkl
└── results/
    └── actual_vs_predicted.png
```

## Dataset

The project uses the California Housing dataset bundled with Scikit-Learn.

| Property | Value |
|---|---:|
| Samples | 20,640 |
| Original features | 8 |
| Engineered features | 1 |
| Target | `MedHouseVal` |
| Training/test split | 80% / 20% |
| Split random seed | `42` |

The first run may download the dataset if it is not already present in Scikit-Learn's local data cache.

## Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Joblib

## Installation

From a terminal in the project directory:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python linear_regression.py
```

The script automatically creates the `models` and `results` directories when they do not exist.

## Model Evaluation

The following results use a fixed `random_state=42`:

| Metric | Result |
|---|---:|
| R-squared (`R2`) | `0.5781` |
| Root mean squared error (`RMSE`) | `0.7435` |
| Training samples | `16,512` |
| Testing samples | `4,128` |

The target unit is the median house value in units of USD 100,000.

## Generated Outputs

- [Trained Ridge pipeline](models/ridge_regression_model.pkl)
- [Actual-versus-predicted plot](results/actual_vs_predicted.png)
- [Complete PDF report](Report.pdf)

The saved model is a complete Scikit-Learn pipeline. It includes preprocessing and Ridge Regression, so it expects a pandas DataFrame containing the original housing features.

### Loading the Model

```python
import joblib

model = joblib.load("models/ridge_regression_model.pkl")
predictions = model.predict(new_housing_data)
```

Your `new_housing_data` must include the original feature columns and `longitude_region`. The fitted pipeline does not recreate longitude-region categories automatically during inference.

## Pipeline Workflow

1. Load and copy the California Housing data.
2. Add the longitude-region feature.
3. Split the data into training and testing sets.
4. Impute and scale numerical features.
5. Impute and one-hot encode categorical features.
6. Fit the Ridge Regression model.
7. Generate test-set predictions.
8. Calculate R-squared and RMSE.
9. Save the fitted pipeline and visualization.

## Method Notes

- Numerical features are mean-imputed and standardized.
- The categorical feature is most-frequent-imputed and one-hot encoded.
- Unknown prediction categories are ignored by `OneHotEncoder`.
- Ridge uses `alpha=1.0`.

## Limitations

- The current linear model explains about 57.8% of test variance.
- Extreme high-value houses are harder to predict accurately.
- The script does not perform cross-validation or hyperparameter tuning.
- Feature engineering should be moved into a custom transformer for a production inference API.