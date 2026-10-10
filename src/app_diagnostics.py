"""Export honest held-out results for the prediction screen charts.

The GUI models are loaded from disk. The deterministic test partitions are
reconstructed from the original supplied files; no models are refitted here.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.classification import SEED as CLASSIFICATION_SEED, prepare_data
from src.regression import SEED as REGRESSION_SEED


def save_diagnostics(root: Path) -> None:
    root = Path(root)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    regression = joblib.load(root / "models" / "regression_gui_model.joblib")
    buildings = pd.read_csv(root / "data" / "regression" / "train.csv")
    buildings = buildings.dropna(subset=["site_eui"])
    _, X_test, _, y_test = train_test_split(
        buildings[regression["features"]], buildings["site_eui"],
        test_size=0.2, random_state=REGRESSION_SEED,
    )
    predicted = regression["model"].predict(X_test)
    pd.DataFrame({
        "Actual Site EUI": y_test.to_numpy(),
        "Predicted Site EUI": predicted,
    }).to_csv(results / "regression_gui_test_predictions.csv", index=False)
    regression_test_rows = len(y_test)
    assert len(y_test) == regression["metrics"]["Test Rows"]

    classification = joblib.load(root / "models" / "classification_model.joblib")
    X, y, _ = prepare_data(root / "data" / "classification" / "water_quality_classification_dataset.csv")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CLASSIFICATION_SEED, stratify=y,
    )
    class_names = classification["class_names"]
    predicted_codes = classification["model"].predict(X_test[classification["features"]])
    true_codes = pd.Categorical(y_test, categories=class_names).codes
    if (true_codes < 0).any():
        raise ValueError("Held-out labels do not match the saved model classes")
    matrix = confusion_matrix(true_codes, predicted_codes, labels=np.arange(len(class_names)))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        results / "classification_gui_confusion.csv", index_label="Actual class",
    )
    print(f"Saved {regression_test_rows:,} regression and {len(X_test):,} classification held-out predictions")


if __name__ == "__main__":
    save_diagnostics(Path(__file__).resolve().parents[1])
