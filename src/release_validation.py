"""Build reproducible, internal evidence for the interactive prediction screens.

Calibration and gate selection use records never used to fit the saved water
classifier. The original held-out partition is evaluated only after selection.
No artifact here is evidence of performance on a new source or population.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from src.classification import SEED as WATER_SEED, prepare_data
from src.clustering import POLLUTANTS, WEATHER, engineer_features
from src.regression import SEED as BUILDING_SEED

WATER_DATE_FEATURES = {"sample_year", "sample_month", "sample_dayofyear", "month_sin", "month_cos"}
AIR_TIME_FIELDS = ["month", "hour", "wd"]


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def percentile_bounds(frame: pd.DataFrame, fields: list[str]) -> dict:
    result = {}
    for name in fields:
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        result[name] = {
            "min": float(values.min()), "p01": float(values.quantile(.01)),
            "p99": float(values.quantile(.99)), "max": float(values.max()),
        }
    return result


def regression_evidence(root: Path, model: dict) -> dict:
    data = pd.read_csv(root / "data/regression/train.csv").dropna(subset=["site_eui"])
    x_train, x_test, _, y_test = train_test_split(
        data[model["features"]], data["site_eui"], test_size=.2,
        random_state=BUILDING_SEED,
    )
    x_cal, x_eval, y_cal, y_eval = train_test_split(
        x_test, y_test, test_size=.5, random_state=2202,
    )
    numeric = model["numeric"]
    count_train = x_train[numeric].notna().sum(axis=1)
    minimum = int(count_train.quantile(.05))
    cal_ok = x_cal[numeric].notna().sum(axis=1) >= minimum
    eval_ok = x_eval[numeric].notna().sum(axis=1) >= minimum
    cal_error = np.abs(y_cal[cal_ok].to_numpy() - model["model"].predict(x_cal[cal_ok]))
    radius = float(np.quantile(cal_error, np.ceil((len(cal_error) + 1) * .90) / len(cal_error), method="higher"))
    eval_truth = y_eval[eval_ok].to_numpy()
    eval_pred = model["model"].predict(x_eval[eval_ok])
    covered = (eval_truth >= np.maximum(0, eval_pred - radius)) & (eval_truth <= eval_pred + radius)
    group_rows = []
    for group, indices in x_eval[eval_ok].groupby("building_class").groups.items():
        positions = x_eval[eval_ok].index.get_indexer(indices)
        group_rows.append({
            "group": str(group), "rows": int(len(positions)),
            "mae": float(np.abs(eval_truth[positions] - eval_pred[positions]).mean()),
        })
    return {
        "model_id": fingerprint(root / "models/regression_gui_model.joblib"),
        "numeric_inputs_required": minimum,
        "training_rows_meeting_gate": int((count_train >= minimum).sum()),
        "training_rows": int(len(count_train)),
        "interval": {
            "nominal_coverage": .90, "radius": radius,
            "calibration_rows": int(cal_ok.sum()), "evaluation_rows": int(eval_ok.sum()),
            "observed_coverage": float(covered.mean()),
            "evaluation_mae": float(np.abs(eval_truth - eval_pred).mean()),
        },
        "group_mae": group_rows,
        "bounds": percentile_bounds(x_train, numeric),
    }


def class_metrics(pred: np.ndarray, truth: np.ndarray, names: list[str]) -> dict:
    return {
        "rows": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "recall": {name: float(recall_score(truth == name, pred == name, zero_division=0)) for name in names},
    }


def classification_evidence(root: Path, model: dict) -> dict:
    x, y, _ = prepare_data(root / "data/classification/water_quality_classification_dataset.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, random_state=WATER_SEED, stratify=y,
    )
    _, x_unused, _, y_unused = train_test_split(
        x_train, y_train, train_size=12000,
        random_state=WATER_SEED, stratify=y_train,
    )
    x_cal, x_gate, y_cal, y_gate = train_test_split(
        x_unused, y_unused, train_size=12000, random_state=2202, stratify=y_unused,
    )
    features = model["features"]
    names = model["class_names"]
    measured = [name for name in features if name not in WATER_DATE_FEATURES]
    calibrated = CalibratedClassifierCV(FrozenEstimator(model["model"]), method="sigmoid")
    calibration_codes = pd.Categorical(y_cal, categories=names).codes
    if (calibration_codes < 0).any():
        raise ValueError("Water calibration labels disagree with saved classes")
    calibrated.fit(x_cal[features], calibration_codes)
    joblib.dump(calibrated, root / "models/classification_calibrated.joblib")

    gate_counts = x_gate[measured].notna().sum(axis=1).to_numpy()
    gate_pred = np.asarray(names)[calibrated.predict(x_gate[features])]
    gate_truth = y_gate.to_numpy()
    selected = None
    # Release targets are explicit; eligibility is selected only on unused
    # training records, then reported on the untouched held-out partition.
    for threshold in range(1, len(measured) + 1):
        keep = gate_counts >= threshold
        if keep.sum() < 1000:
            continue
        metric = class_metrics(gate_pred[keep], gate_truth[keep], names)
        if metric["accuracy"] >= .90 and metric["recall"]["Poor"] >= .83:
            selected = threshold
            break
    if selected is None:
        raise RuntimeError("No water input gate met the specified validation targets")

    test_counts = x_test[measured].notna().sum(axis=1).to_numpy()
    test_probabilities = calibrated.predict_proba(x_test[features])
    test_pred = np.asarray(names)[np.argmax(test_probabilities, axis=1)]
    test_truth = y_test.to_numpy()
    test_ok = test_counts >= selected
    truth_codes = pd.Categorical(y_test, categories=names).codes
    one_hot = np.eye(len(names))[truth_codes]
    raw_probabilities = model["model"].predict_proba(x_test[features])
    confidence = test_probabilities.max(axis=1)
    correct = test_pred == test_truth
    bins = np.minimum((confidence * 10).astype(int), 9)
    reliability = pd.DataFrame([
        {
            "Confidence band": f"{index * 10}–{(index + 1) * 10}%",
            "Samples": int((bins == index).sum()),
            "Mean model confidence": float(confidence[bins == index].mean()),
            "Observed accuracy": float(correct[bins == index].mean()),
        }
        for index in range(10) if (bins == index).any()
    ])
    reliability.to_csv(root / "results/classification_reliability.csv", index=False)
    early = x["sample_year"].between(2010, 2019)
    later = x["sample_year"].between(2020, 2024)
    x_early, _, y_early, _ = train_test_split(
        x.loc[early], y.loc[early], train_size=12000,
        random_state=2202, stratify=y.loc[early],
    )
    temporal_model = clone(model["model"])
    temporal_model.fit(x_early[features], pd.Categorical(y_early, categories=names).codes)
    temporal_pred = np.asarray(names)[temporal_model.predict(x.loc[later, features])]
    temporal_metrics = class_metrics(temporal_pred, y.loc[later].to_numpy(), names)
    return {
        "model_id": fingerprint(root / "models/classification_calibrated.joblib"),
        "base_model_id": fingerprint(root / "models/classification_model.joblib"),
        "measured_fields_required": selected,
        "measured_fields_total": len(measured),
        "gate_targets": {"accuracy": .90, "poor_recall": .83},
        "calibration_rows": int(len(x_cal)),
        "gate_validation": class_metrics(gate_pred[gate_counts >= selected], gate_truth[gate_counts >= selected], names),
        "gate_validation_total": int(len(x_gate)),
        "held_out": class_metrics(test_pred, test_truth, names),
        "held_out_eligible": class_metrics(test_pred[test_ok], test_truth[test_ok], names),
        "held_out_eligible_rows": int(test_ok.sum()),
        "held_out_rows": int(len(test_ok)),
        "held_out_multiclass_brier": float(np.sum((test_probabilities - one_hot) ** 2, axis=1).mean()),
        "base_held_out_multiclass_brier": float(np.sum((raw_probabilities - one_hot) ** 2, axis=1).mean()),
        "temporal_stress": {
            "training_years": "2010–2019", "evaluation_years": "2020–2024",
            "training_rows": int(len(x_early)), **temporal_metrics,
            "note": "Separate model refit with the saved feature list; same source file, not external validation.",
        },
        "bounds": percentile_bounds(x_train, measured),
        "sample_year_bounds": percentile_bounds(x_train, ["sample_year"])["sample_year"],
        "provenance": "Instructor-supplied file; original measurement units and label rules are undocumented.",
    }


def clustering_evidence(root: Path, model: dict) -> dict:
    rows = []
    fields = POLLUTANTS + WEATHER + AIR_TIME_FIELDS
    for path in sorted((root / "data/clustering").glob("PRSA_Data_*_20130301-20170228.csv")):
        frame = pd.read_csv(path, usecols=fields).dropna()
        rows.append(frame.sample(n=min(250, len(frame)), random_state=42))
    if len(rows) != 12:
        raise FileNotFoundError("All 12 Beijing stations are needed for release evidence")
    complete = pd.concat(rows, ignore_index=True)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        scaled = model["preprocessing"].transform(engineer_features(frame))
        return model["kmeans"].predict(scaled)

    baseline = predict(complete)
    comparisons = []
    masked_distances = []
    # Compare partially supplied records with each same record's complete-data
    # assignment. This tests assignment stability, not ground-truth accuracy.
    for seed in range(8):
        partial = complete.copy()
        rng = np.random.default_rng(seed)
        for names, keep_count in ((POLLUTANTS, 4), (WEATHER, 3), (AIR_TIME_FIELDS, 2)):
            keep = np.argsort(rng.random((len(partial), len(names))), axis=1)[:, :keep_count]
            for index, name in enumerate(names):
                partial.loc[~np.any(keep == index, axis=1), name] = np.nan
        partial_scaled = model["preprocessing"].transform(engineer_features(partial))
        comparisons.append(float((model["kmeans"].predict(partial_scaled) == baseline).mean()))
        masked_distances.append(model["kmeans"].transform(partial_scaled).min(axis=1))
    scaled = model["preprocessing"].transform(engineer_features(complete))
    distances = model["kmeans"].transform(scaled).min(axis=1)
    return {
        "model_id": fingerprint(root / "models/clustering_model.joblib"),
        "minimum_pollutants": 4,
        "minimum_weather": 3,
        "minimum_time_fields": 2,
        "masked_assignment_agreement": float(np.mean(comparisons)),
        "mask_trials": len(comparisons),
        "reference_rows": int(len(complete)),
        "distance_p99_complete": float(np.quantile(distances, .99)),
        "distance_p99_masked": float(np.quantile(np.concatenate(masked_distances), .99)),
        "bounds": percentile_bounds(complete, POLLUTANTS + WEATHER),
    }


def build(root: Path) -> dict:
    root = Path(root)
    building = joblib.load(root / "models/regression_gui_model.joblib")
    water = joblib.load(root / "models/classification_model.joblib")
    air = joblib.load(root / "models/clustering_model.joblib")
    evidence = {
        "status": "Internal study validation only; no external population has been evaluated.",
        "regression": regression_evidence(root, building),
        "classification": classification_evidence(root, water),
        "clustering": clustering_evidence(root, air),
    }
    output = root / "results/release_validation.json"
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    built = build(Path(__file__).resolve().parents[1])
    print(json.dumps({
        "water_fields_required": built["classification"]["measured_fields_required"],
        "water_eligible_accuracy": built["classification"]["held_out_eligible"]["accuracy"],
        "regression_interval_coverage": built["regression"]["interval"]["observed_coverage"],
        "masked_cluster_agreement": built["clustering"]["masked_assignment_agreement"],
    }, indent=2))
