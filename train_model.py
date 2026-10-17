"""Train a Linear Regression model for used car price prediction.

Run:  python train_model.py
Output: model.pkl (pipeline with preprocessing + LinearRegression)

The script generates a realistic synthetic dataset so the project works out of
the box. To use real data, replace `load_data()` with `pd.read_csv("cars.csv")`
containing the same columns (see FEATURES / TARGET below).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_YEAR = 2026
NUMERIC_FEATURES = ["car_age", "kms_driven", "engine_cc", "owners"]
CATEGORICAL_FEATURES = ["fuel_type", "transmission"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price_lakhs"
MODEL_PATH = "model.pkl"


def load_data(n=3000, seed=42):
    """Create a synthetic used-car dataset (prices in lakhs of INR)."""
    rng = np.random.default_rng(seed)
    car_age = rng.integers(0, 16, n)
    kms_driven = np.clip(car_age * rng.normal(11000, 3500, n), 500, 300000).astype(int)
    engine_cc = rng.choice([800, 1000, 1200, 1500, 1800, 2000, 2500], n)
    owners = rng.choice([1, 2, 3, 4], n, p=[0.6, 0.27, 0.10, 0.03])
    fuel_type = rng.choice(["Petrol", "Diesel", "CNG"], n, p=[0.55, 0.38, 0.07])
    transmission = rng.choice(["Manual", "Automatic"], n, p=[0.72, 0.28])

    price = (
        2.0
        + engine_cc * 0.0035
        - car_age * 0.42
        - kms_driven * 0.000007
        - (owners - 1) * 0.35
        + (fuel_type == "Diesel") * 0.9
        - (fuel_type == "CNG") * 0.3
        + (transmission == "Automatic") * 1.2
        + rng.normal(0, 0.5, n)
    )
    price = np.clip(price, 0.4, None).round(2)

    return pd.DataFrame(
        {
            "car_age": car_age,
            "kms_driven": kms_driven,
            "engine_cc": engine_cc,
            "owners": owners,
            "fuel_type": fuel_type,
            "transmission": transmission,
            TARGET: price,
        }
    )


def build_pipeline():
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(f"R2 score : {r2_score(y_test, preds):.3f}")
    print(f"MAE      : {mean_absolute_error(y_test, preds):.3f} lakhs")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
