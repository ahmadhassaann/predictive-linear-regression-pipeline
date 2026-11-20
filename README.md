# Predictive Linear Regression Model Pipeline

A reproducible Python machine-learning pipeline that predicts California median house values with Ridge Regression. The script prepares the data, trains a complete Scikit-Learn pipeline, evaluates predictions, saves the trained model, and produces an actual-versus-predicted plot.

## Features

- Loads the California Housing dataset directly through Scikit-Learn.
- Creates a `longitude_region` feature from longitude (`Western`, `Central`, or `Eastern`).
- Applies mean imputation and standard scaling to numerical features.
- Applies most-frequent imputation and one-hot encoding to categorical features.
- Trains a Ridge Regression model with `alpha=1.0`.
- Reports R-squared and root mean squared error.
- Saves the fitted pipeline with Joblib.
- Exports an actual-versus-predicted scatter plot.

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
- Seaborn
- Joblib

Seaborn is listed in `requirements.txt`, although the current script does not import it.

## Installation

From a terminal in the project directory:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python linear_regression.py
```

The script automatically creates the `models` and `results` directories when they do not exist.

## Model Evaluation

The following results are from an execution using a fixed `random_state=42`:

| Metric | Result |
|---|---:|
| R-squared (`R2`) | `0.5781` |
| Root mean squared error (`RMSE`) | `0.7435` |
| Training samples | `16,512` |
| Testing samples | `4,128` |

The target unit is the median house value in units of USD 100,000.

## Generated Outputs

- Trained model: [ridge_regression_model.pkl](models/ridge_regression_model.pkl)
- Evaluation plot: [actual_vs_predicted.png](results/actual_vs_predicted.png)

The saved model is a complete Scikit-Learn pipeline. It includes preprocessing and Ridge Regression, so it expects a pandas DataFrame containing the original housing features.

### Loading the Model

```python
import joblib

model = joblib.load("models/ridge_regression_model.pkl")
predictions = model.predict(new_housing_data)
```

Your `new_housing_data` must include the original California Housing feature columns. The fitted pipeline creates the engineered longitude region during training, but the model does not recreate it automatically when loading for prediction. Add `longitude_region` to prediction data or wrap feature engineering in a reusable function before using the model for inference.

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

## Author

Ahmad Hassan  
BS Artificial Intelligence  
FAST NUCES Islamabad