"""Fail fast if a deployment bundle cannot serve the three example requests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.classification import prepare_data
from src.clustering import POLLUTANTS, WEATHER, engineer_features
from src.prediction_policy import classification_issues, clustering_issues, regression_issues
from src.release_validation import AIR_TIME_FIELDS, WATER_DATE_FEATURES, fingerprint


def check(root: Path) -> dict:
    root = Path(root)
    evidence = json.loads((root / "results/release_validation.json").read_text(encoding="utf-8"))
    for task, name, key in (
        ("regression", "regression_gui_model.joblib", "model_id"),
        ("classification", "classification_model.joblib", "base_model_id"),
        ("classification", "classification_calibrated.joblib", "model_id"),
        ("clustering", "clustering_model.joblib", "model_id"),
    ):
        if fingerprint(root / "models" / name) != evidence[task][key]:
            raise AssertionError(f"Stale release evidence for {name}")

    building = joblib.load(root / "models/regression_gui_model.joblib")
    water = joblib.load(root / "models/classification_model.joblib")
    calibrated = joblib.load(root / "models/classification_calibrated.joblib")
    air = joblib.load(root / "models/clustering_model.joblib")

    building_row = pd.read_csv(root / "data/regression/train.csv", usecols=building["features"], nrows=1)
    building_values = building_row.iloc[0].to_dict()
    assert not regression_issues(building_values, evidence["regression"], building["numeric"])
    building_prediction = float(building["model"].predict(building_row)[0])
    assert np.isfinite(building_prediction)

    x, _, _ = prepare_data(root / "data/classification/water_quality_classification_dataset.csv")
    measured = [field for field in water["features"] if field not in WATER_DATE_FEATURES]
    allowed = x[measured].notna().sum(axis=1) >= evidence["classification"]["measured_fields_required"]
    water_row = x.loc[allowed, water["features"]].head(1)
    assert len(water_row) == 1
    assert not classification_issues(water_row.iloc[0].to_dict(), evidence["classification"], measured)
    probabilities = calibrated.predict_proba(water_row)[0]
    assert np.isfinite(probabilities).all() and np.isclose(probabilities.sum(), 1)

    fields = POLLUTANTS + WEATHER + AIR_TIME_FIELDS
    path = sorted((root / "data/clustering").glob("PRSA_Data_*_20130301-20170228.csv"))[0]
    air_row = pd.read_csv(path, usecols=fields).dropna().head(1)
    scaled = air["preprocessing"].transform(engineer_features(air_row))
    distance = float(air["kmeans"].transform(scaled).min(axis=1)[0])
    assert not clustering_issues(air_row.iloc[0].to_dict(), evidence["clustering"], distance)
    cluster = int(air["kmeans"].predict(scaled)[0])
    assert 0 <= cluster < air["kmeans"].n_clusters
    return {
        "bundle": "ready for research-demo hosting",
        "building_estimate_finite": True,
        "water_probabilities_sum_to_one": True,
        "air_cluster_in_range": True,
        "model_fingerprints_match_evidence": True,
    }


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
