
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


# 1. Define project directories
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load California Housing dataset
print("Loading California Housing dataset...")

dataset = fetch_california_housing(as_frame=True)

df = dataset.frame.copy()

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]


# 3. Feature engineering
X["longitude_region"] = pd.cut(
    X["Longitude"],
    bins=3,
    labels=["Western", "Central", "Eastern"]
).astype("object")


# 4. Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# 5. Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


# 6. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 7. Combine preprocessing
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 8. Create Ridge Regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])


# 9. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 10. Train model
print("Training Ridge Regression model...")

model.fit(X_train, y_train)


# 11. Generate predictions
y_pred = model.predict(X_test)


# 12. Evaluate model
r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)


# 13. Display evaluation results
print("\nMODEL EVALUATION RESULTS")
print("-" * 40)

print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

print("-" * 40)


# 14. Save trained model inside models folder
model_path = MODELS_DIR / "ridge_regression_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved at: {model_path}")


# 15. Generate actual vs predicted visualization
plt.figure(figsize=(9, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")

plt.title("Actual vs Predicted - Ridge Regression")

plt.grid(True)
plt.tight_layout()


# 16. Save visualization inside results folder
plot_path = RESULTS_DIR / "actual_vs_predicted.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(f"Visualization saved at: {plot_path}")


# 17. Final execution summary
print("\nPROJECT EXECUTION COMPLETED")
print("-" * 40)

print("Dataset loaded successfully.")
print("Data preprocessing completed.")
print("Ridge Regression trained successfully.")
print("Model evaluation completed.")
print("Trained model saved successfully.")
print("Visualization saved successfully.")

print("-" * 40)
print(f"Project directory: {BASE_DIR}")
