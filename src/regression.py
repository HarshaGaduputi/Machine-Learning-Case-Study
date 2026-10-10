"""Compact, reproducible regression model for the interactive application.

The Review 1 notebook remains the full ten-algorithm analysis. This model uses
the subset of source features that can be entered comfortably in a web form.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
NUMERIC = [
    "floor_area", "year_built", "energy_star_rating", "ELEVATION",
    "avg_temp", "cooling_degree_days", "heating_degree_days",
    "precipitation_inches", "snowfall_inches",
]
CATEGORICAL = ["State_Factor", "building_class", "facility_type"]
FEATURES = NUMERIC + CATEGORICAL


def train_regression(data_path: Path) -> dict:
    data = pd.read_csv(data_path)
    data = data.dropna(subset=["site_eui"])
    X = data[FEATURES].copy()
    y = data["site_eui"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
    )
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=120, max_depth=18, min_samples_leaf=3,
            random_state=SEED, n_jobs=2,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "R2": float(r2_score(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "Training Rows": int(len(X_train)),
        "Test Rows": int(len(X_test)),
    }
    defaults = X_train[NUMERIC].median().to_dict()
    categories = {
        column: sorted(X_train[column].dropna().unique().tolist())
        for column in CATEGORICAL
    }
    modes = {column: str(X_train[column].mode().iloc[0]) for column in CATEGORICAL}
    return {
        "model": model, "features": FEATURES, "numeric": NUMERIC,
        "categorical": CATEGORICAL, "defaults": defaults,
        "categories": categories, "modes": modes, "metrics": metrics,
    }


def save_regression(result: dict, root: Path) -> None:
    root = Path(root)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    joblib.dump(result, root / "models" / "regression_gui_model.joblib")
    pd.DataFrame([result["metrics"]]).to_csv(
        root / "results" / "regression_gui_metrics.csv", index=False,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = train_regression(root / "data" / "regression" / "train.csv")
    save_regression(result, root)
    print(result["metrics"])
