"""Read-only checks for a candidate research-demo bundle.

This checks packaging consistency before the separate model smoke check. It
does not grant dataset publication rights or authenticate files to an owner:
the model IDs and evidence live in the same repository.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "app/streamlit_app.py", "app/dashboard.py", ".streamlit/config.toml",
    "requirements.txt",
    "data/regression/train.csv",
    "data/classification/water_quality_classification_dataset.csv",
    "models/regression_gui_model.joblib", "models/classification_model.joblib",
    "models/classification_calibrated.joblib", "models/clustering_model.joblib",
    "results/release_validation.json", "results/regression_gui_metrics.csv",
    "results/regression_gui_test_predictions.csv",
    "results/classification_comparison.csv", "results/classification_gui_confusion.csv",
    "results/classification_reliability.csv", "results/clustering_comparison.csv",
    "results/clustering_elbow.csv", "results/clustering_pca.csv",
    "results/clustering_linkages.csv", "results/clustering_profiles.csv",
    "notebooks/regression_review1.ipynb",
    "notebooks/classification_review1_part_a.ipynb",
    "notebooks/clustering.ipynb", "notebooks/classification_review2.ipynb",
)
MODEL_EVIDENCE = (
    ("models/regression_gui_model.joblib", "regression", "model_id"),
    ("models/classification_model.joblib", "classification", "base_model_id"),
    ("models/classification_calibrated.joblib", "classification", "model_id"),
    ("models/clustering_model.joblib", "clustering", "model_id"),
)


def regular_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Not a regular file: {relative}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty required file: {relative}")
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check(root: Path, *, assets_only: bool = False) -> dict:
    root = root.resolve()
    for relative in REQUIRED:
        regular_file(root, relative)
    stations = sorted((root / "data/clustering").glob("PRSA_Data_*_20130301-20170228.csv"))
    if len(stations) != 12:
        raise ValueError(f"Expected 12 Beijing station files; found {len(stations)}")
    for path in stations:
        regular_file(root, str(path.relative_to(root)))

    evidence = json.loads(regular_file(root, "results/release_validation.json").read_text(encoding="utf-8"))
    model_sha256 = {}
    for relative, task, key in MODEL_EVIDENCE:
        full_hash = digest(root / relative)
        if full_hash[:12] != evidence[task][key]:
            raise ValueError(f"Model and release evidence disagree: {relative}")
        model_sha256[relative] = full_hash

    notebook_outputs = {}
    for relative in (name for name in REQUIRED if name.endswith(".ipynb")):
        notebook = json.loads((root / relative).read_text(encoding="utf-8"))
        count = sum(bool(cell.get("outputs")) for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "code")
        if count == 0:
            raise ValueError(f"Notebook has no saved outputs: {relative}")
        notebook_outputs[relative] = count

    packages = {}
    if not assets_only:
        if sys.version_info[:2] != (3, 13):
            raise ValueError("Release runtime must use Python 3.13")
        requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        for line in requirements:
            if not line or line.startswith("#"):
                continue
            name, expected = line.split("==", 1)
            installed = metadata.version(name)
            if installed != expected:
                raise ValueError(f"{name} {installed} installed; {expected} required")
            packages[name] = installed

    return {
        "asset_checks": "passed",
        "runtime_checks": "skipped" if assets_only else "passed",
        "model_sha256": model_sha256,
        "notebook_output_cells": notebook_outputs,
        "packages": packages,
        "release_status": "HOLD",
        "publication_permission": "HOLD: building and water data owner permission unresolved",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true",
                        help="Check files and notebook outputs without a release runtime")
    parser.add_argument("--release-gate", action="store_true",
                        help="Fail while the dataset publication hold remains unresolved")
    args = parser.parse_args()
    result = check(ROOT, assets_only=args.assets_only)
    print(json.dumps(result, indent=2))
    if args.release_gate:
        parser.exit(1, "Release blocked: building and water publication permission is on hold.\n")


if __name__ == "__main__":
    main()
