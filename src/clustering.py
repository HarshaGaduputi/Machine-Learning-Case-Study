"""Beijing air-quality clustering with reproducible shared preprocessing.

No target label is used during fitting. The station name and calendar day are
retained for interpretation only, not passed to either clustering algorithm.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
COMPASS = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}
FEATURES = (
    [f"{name}_log" for name in POLLUTANTS]
    + WEATHER
    + ["hour_sin", "hour_cos", "month_sin", "month_cos", "wd_sin", "wd_cos"]
)


def load_stations(directory: Path) -> pd.DataFrame:
    """Load the supplied 12 station CSVs and verify their shared schema."""
    files = sorted(Path(directory).glob("PRSA_Data_*_20130301-20170228.csv"))
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 Beijing station CSV files in {directory}; found {len(files)}")
    frames = []
    reference = None
    for path in files:
        frame = pd.read_csv(path)
        if reference is None:
            reference = list(frame.columns)
        elif list(frame.columns) != reference:
            raise ValueError(f"Station schema differs: {path.name}")
        station = path.name.removeprefix("PRSA_Data_").removesuffix("_20130301-20170228.csv")
        if not (frame["station"] == station).all():
            raise ValueError(f"Station names disagree with filename: {path.name}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["station", "year", "month", "day", "hour"]
    if combined.duplicated(keys).any():
        raise ValueError("Duplicate station timestamps found")
    return combined


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transform raw readings; leave missing values for the fitted imputer."""
    required = POLLUTANTS + WEATHER + ["hour", "month", "wd"]
    absent = set(required) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing inputs: {sorted(absent)}")
    features = pd.DataFrame(index=frame.index)
    for name in POLLUTANTS:
        values = pd.to_numeric(frame[name], errors="coerce")
        # Negative concentrations are physically impossible; treat them as missing.
        features[f"{name}_log"] = np.log1p(values.where(values >= 0))
    for name in WEATHER:
        values = pd.to_numeric(frame[name], errors="coerce")
        if name in ("RAIN", "WSPM"):
            values = values.where(values >= 0)
        features[name] = values
    hour = pd.to_numeric(frame["hour"], errors="coerce")
    month = pd.to_numeric(frame["month"], errors="coerce")
    if not hour.dropna().between(0, 23).all() or not month.dropna().between(1, 12).all():
        raise ValueError("Hour must be 0-23 and month must be 1-12")
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    angle = np.deg2rad(frame["wd"].map(COMPASS))
    features["wd_sin"] = np.sin(angle)
    features["wd_cos"] = np.cos(angle)
    return features[FEATURES]


def sample_indices(n_rows: int, size: int, seed: int = SEED) -> np.ndarray:
    """Choose a stable sample without replacement, sorted for clear row alignment."""
    return np.sort(np.random.default_rng(seed).choice(n_rows, min(size, n_rows), replace=False))


def evaluate(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        raise ValueError("At least two clusters are needed for evaluation")
    return {
        "Silhouette Score": float(silhouette_score(X, labels)),
        "Davies-Bouldin Index": float(davies_bouldin_score(X, labels)),
        "Calinski-Harabasz Index": float(calinski_harabasz_score(X, labels)),
    }


def run_clustering(
    raw: pd.DataFrame,
    *,
    selection_size: int = 12000,
    silhouette_size: int = 1500,
    comparison_size: int = 3000,
    k_values: range = range(2, 8),
) -> dict:
    """Fit both methods, using one common sample for their final metrics."""
    features = engineer_features(raw)
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X = preprocessing.fit_transform(features)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values remain after preprocessing")

    select_idx = sample_indices(len(X), selection_size)
    silhouette_idx = sample_indices(len(select_idx), silhouette_size, SEED + 1)
    X_select = X[select_idx]
    X_sil = X_select[silhouette_idx]
    elbow_rows = []
    for k in k_values:
        candidate = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_select)
        labels = candidate.predict(X_sil)
        elbow_rows.append({
            "K": k,
            "Inertia": float(candidate.inertia_),
            "Silhouette Score": float(silhouette_score(X_sil, labels)),
        })
    elbow = pd.DataFrame(elbow_rows)
    best_k = int(elbow.loc[elbow["Silhouette Score"].idxmax(), "K"])

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    all_labels = kmeans.labels_
    common_idx = sample_indices(len(X), comparison_size, SEED + 2)
    X_common = X[common_idx]
    kmeans_common = all_labels[common_idx]

    # Linkage comparison uses exactly the common feature matrix and cluster count.
    linkage_rows = []
    candidates = {}
    for method in ("ward", "complete", "average"):
        labels = AgglomerativeClustering(n_clusters=best_k, linkage=method).fit_predict(X_common)
        candidates[method] = labels
        linkage_rows.append({"Linkage": method, **evaluate(X_common, labels)})
    linkage_table = pd.DataFrame(linkage_rows)
    # Ward is used for the final hierarchical model because it minimizes the
    # increase in within-cluster variance; other strategies remain visible.
    hierarchy_labels = candidates["ward"]
    comparison = pd.DataFrame([
        {"Algorithm": "K-Means", "Clusters": best_k, "Evaluation Rows": len(common_idx),
         **evaluate(X_common, kmeans_common)},
        {"Algorithm": "Agglomerative (Ward)", "Clusters": best_k,
         "Evaluation Rows": len(common_idx), **evaluate(X_common, hierarchy_labels)},
    ])

    pca = PCA(n_components=2, random_state=SEED)
    pca_points = pca.fit_transform(X_common)
    dendrogram_idx = sample_indices(len(X_common), min(1000, len(X_common)), SEED + 3)
    dendrogram_linkage = linkage(X_common[dendrogram_idx], method="ward")
    tsne_idx = sample_indices(len(X_common), min(1200, len(X_common)), SEED + 4)
    tsne_input = PCA(n_components=min(10, X.shape[1]), random_state=SEED).fit_transform(X_common[tsne_idx])
    tsne_points = TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto",
        random_state=SEED,
    ).fit_transform(tsne_input)

    profile = raw.assign(Cluster=all_labels).groupby("Cluster")[POLLUTANTS + WEATHER].mean()
    profile.insert(0, "Observations", pd.Series(all_labels).value_counts().sort_index())
    return {
        "features": features, "preprocessing": preprocessing, "X": X,
        "elbow": elbow, "best_k": best_k, "kmeans": kmeans,
        "all_labels": all_labels, "common_indices": common_idx,
        "kmeans_common": kmeans_common, "hierarchy_common": hierarchy_labels,
        "linkage_table": linkage_table, "comparison": comparison,
        "pca": pca, "pca_points": pca_points,
        "dendrogram_linkage": dendrogram_linkage,
        "tsne_indices": tsne_idx, "tsne_points": tsne_points,
        "profile": profile,
        "raw_medians": raw[POLLUTANTS + WEATHER].median().to_dict(),
    }


def save_outputs(result: dict, directory: Path) -> None:
    directory = Path(directory)
    (directory / "models").mkdir(parents=True, exist_ok=True)
    (directory / "results").mkdir(parents=True, exist_ok=True)
    result["elbow"].to_csv(directory / "results" / "clustering_elbow.csv", index=False)
    result["linkage_table"].to_csv(directory / "results" / "clustering_linkages.csv", index=False)
    result["comparison"].to_csv(directory / "results" / "clustering_comparison.csv", index=False)
    result["profile"].to_csv(directory / "results" / "clustering_profiles.csv")
    pd.DataFrame({
        "PC1": result["pca_points"][:, 0],
        "PC2": result["pca_points"][:, 1],
        "K-Means Cluster": result["kmeans_common"],
        "Agglomerative Cluster": result["hierarchy_common"],
    }).to_csv(directory / "results" / "clustering_pca.csv", index=False)
    joblib.dump({
        "preprocessing": result["preprocessing"],
        "kmeans": result["kmeans"],
        "features": FEATURES,
        "profile": result["profile"],
        "raw_medians": result["raw_medians"],
    }, directory / "models" / "clustering_model.joblib")
