"""Water-quality classification for the consolidated Review 2 comparison."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "Water_Quality_Class"
REVIEW2_ALGORITHMS = (
    "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging",
    "MLP Neural Network",
)


def prepare_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Clean the same labeled source used for Review 1 Part A."""
    raw = pd.read_csv(csv_path)
    audit = {
        "raw_rows": len(raw),
        "missing_targets": int(raw[TARGET].isna().sum()),
    }
    raw = raw.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    audit["labeled_unique_rows"] = len(raw)
    dates = pd.to_datetime(raw["Sample Date"], errors="coerce")
    raw["sample_year"] = dates.dt.year
    raw["sample_month"] = dates.dt.month
    raw["sample_dayofyear"] = dates.dt.dayofyear
    raw["month_sin"] = np.sin(2 * np.pi * raw["sample_month"] / 12)
    raw["month_cos"] = np.cos(2 * np.pi * raw["sample_month"] / 12)
    X = raw.drop(columns=[TARGET, "Sample Date", "Sample Time"]).select_dtypes(include="number")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, raw[TARGET], audit


def make_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", classifier),
    ])


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "Weighted F1": float(f1_score(y_test, pred, average="weighted")),
        "ROC-AUC": float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")),
    }
    return metrics, confusion_matrix(y_test, pred)


def run_classification(csv_path: Path, train_limit: int = 12000) -> dict:
    X, y, audit = prepare_data(csv_path)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )
    # The same stratified training sample is used for all ten algorithms.
    # It keeps SVC and five-fold validation practical on a laptop.
    if len(X_train_full) > train_limit:
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full, train_size=train_limit,
            random_state=SEED, stratify=y_train_full,
        )
    else:
        X_train, y_train = X_train_full, y_train_full

    # Missingness threshold is determined on training data only.
    feature_columns = X_train.columns[X_train.isna().mean() <= 0.8].tolist()
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]
    encoder = LabelEncoder().fit(y_train)
    y_train_code = encoder.transform(y_train)
    y_test_code = encoder.transform(y_test)

    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, n_jobs=2),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Support Vector Machine": SVC(kernel="rbf", C=1, probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=2, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=80, learning_rate=0.5, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, random_state=SEED),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=12, random_state=SEED),
            n_estimators=60, n_jobs=2, random_state=SEED,
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", early_stopping=True,
            max_iter=120, random_state=SEED,
        ),
    }
    models, rows, matrices = {}, [], {}
    for name, classifier in algorithms.items():
        model = make_pipeline(classifier)
        model.fit(X_train, y_train_code)
        scores, matrix = evaluate_model(model, X_test, y_test_code)
        rows.append({"Algorithm": name, **scores})
        models[name] = model
        matrices[name] = matrix

    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    tree_search = GridSearchCV(
        make_pipeline(DecisionTreeClassifier(random_state=SEED)),
        {"model__max_depth": [None, 8, 12, 20], "model__min_samples_leaf": [1, 5]},
        scoring="f1_weighted", cv=cv3, n_jobs=1,
    ).fit(X_train, y_train_code)
    baseline_cv = cross_val_score(
        make_pipeline(DecisionTreeClassifier(random_state=SEED)),
        X_train, y_train_code, scoring="f1_weighted", cv=cv3, n_jobs=1,
    ).mean()
    tuned_tree = tree_search.best_estimator_
    tuned_scores, tuned_matrix = evaluate_model(tuned_tree, X_test, y_test_code)
    # Copy before replacing the table row; otherwise the update below mutates
    # the same dictionary and makes the before/after test scores identical.
    baseline_scores = next(row for row in rows if row["Algorithm"] == "Decision Tree").copy()
    for row in rows:
        if row["Algorithm"] == "Decision Tree":
            row.update(tuned_scores)
    models["Decision Tree"] = tuned_tree
    matrices["Decision Tree"] = tuned_matrix
    comparison = pd.DataFrame(rows).sort_values("Weighted F1", ascending=False).reset_index(drop=True)

    # Select only among the five Review 2 additions, using training CV rather
    # than the held-out test table. This keeps the test set descriptive.
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name in REVIEW2_ALGORITHMS:
        scores = cross_val_score(models[name], X_train, y_train_code,
                                 scoring="f1_weighted", cv=cv5, n_jobs=1)
        cv_rows.append({"Algorithm": name, "Mean 5-Fold Weighted F1": scores.mean(),
                        "Std 5-Fold Weighted F1": scores.std()})
    cv_table = pd.DataFrame(cv_rows).sort_values("Mean 5-Fold Weighted F1", ascending=False)
    final_name = str(cv_table.iloc[0]["Algorithm"])

    return {
        "audit": audit, "feature_columns": feature_columns,
        "class_names": encoder.classes_.tolist(),
        "training_rows": len(X_train), "test_rows": len(X_test),
        "comparison": comparison, "cv_table": cv_table,
        "tuning": pd.DataFrame([{
            "Algorithm": "Decision Tree",
            "Baseline 3-Fold F1": baseline_cv,
            "Tuned 3-Fold F1": tree_search.best_score_,
            "Improvement": tree_search.best_score_ - baseline_cv,
            "Baseline Test F1": baseline_scores["Weighted F1"],
            "Tuned Test F1": tuned_scores["Weighted F1"],
            "Best Parameters": str(tree_search.best_params_),
        }]),
        "confusion_matrices": matrices, "final_name": final_name,
        "final_model": models[final_name], "models": models,
        "X_test": X_test, "y_test_code": y_test_code,
    }


def save_outputs(result: dict, directory: Path) -> None:
    directory = Path(directory)
    (directory / "models").mkdir(parents=True, exist_ok=True)
    (directory / "results").mkdir(parents=True, exist_ok=True)
    result["comparison"].to_csv(directory / "results" / "classification_comparison.csv", index=False)
    result["cv_table"].to_csv(directory / "results" / "classification_cv.csv", index=False)
    result["tuning"].to_csv(directory / "results" / "classification_tuning.csv", index=False)
    joblib.dump({
        "model": result["final_model"],
        "name": result["final_name"],
        "features": result["feature_columns"],
        "class_names": result["class_names"],
    }, directory / "models" / "classification_model.joblib")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = run_classification(
        project_root / "data/classification/water_quality_classification_dataset.csv"
    )
    save_outputs(output, project_root)
    print(output["comparison"].round(4).to_string(index=False))
    print("\nReview 2 model selection (training-only CV):")
    print(output["cv_table"].round(4).to_string(index=False))
    print(f"\nSelected model: {output['final_name']}")
