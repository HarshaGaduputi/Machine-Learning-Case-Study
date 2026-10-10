# Machine Learning Case Study · Review 2

This project has three assigned tracks and a working interactive GUI for **regression, classification, and clustering**. The Review 2 notebooks run the full water-quality classification comparison and the full Beijing air-quality clustering analysis. The supplied Review 1 regression and classification notebooks are retained in `notebooks/`, with local dataset paths repaired.

## Problem statements and data

| Track | Supplied data | Problem |
|---|---|---|
| Regression (Review 1) | `data/regression/train.csv`, 75,757 building records | Predict Site Energy Use Intensity (`site_eui`). |
| Classification (Reviews 1–2) | `data/classification/water_quality_classification_dataset.csv`, 80,000 raw records | Predict the labeled `Good`, `Moderate`, or `Poor` water-quality class. Rows with missing class labels are excluded. |
| Clustering (Review 2) | 12 Beijing station CSVs in `data/clustering/`, 420,768 hourly readings | Find recurring pollution and weather patterns without class labels. |

The three tracks use separate instructor-supplied datasets. The water-quality dataset has extensive missing predictor values, so Review 2 selects columns with at most 80% missingness **on the training split only**, then fits a median imputer with missing indicators and a scaler inside every model pipeline. Review 2 uses the same stratified 12,000-row training sample and the same complete 20% test partition for all ten classifiers. The smaller training sample makes SVC and cross-validation practical on a laptop. It should be considered when interpreting the scores.

For clustering, six pollutant measurements are log transformed; hour, month, and wind direction are cyclic features. Missing values are median-imputed and features are standardized. K-Means is fitted on all 420,768 rows. Ward agglomerative clustering is fitted on a reproducible 3,000-row sample due to its pairwise memory and runtime cost. **Both algorithms' final metrics use those same 3,000 observations.** No ground-truth quality labels are used during clustering.

## Review 2 results

### Classification · consolidated ten-algorithm comparison

Run `notebooks/classification_review2.ipynb` for the full table of Accuracy, weighted Precision, weighted Recall, weighted F1, and weighted one-vs-rest ROC-AUC, confusion matrices for all ten algorithms, the final ROC curves, and the tuning comparison. The notebook saves machine-readable tables under `results/` and the chosen model under `models/`.

| Leading algorithm | Accuracy | Weighted F1 | Weighted OvR ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 0.8855 | 0.8837 | 0.9680 |
| Bagging | 0.8788 | 0.8778 | 0.9607 |
| Gradient Boosting | 0.8770 | 0.8750 | 0.9603 |

Random Forest is selected by five-fold training cross-validation (weighted F1 `0.8814 ± 0.0042`, versus Bagging's `0.8763 ± 0.0075`). Decision Tree grid search improved three-fold training-CV weighted F1 from `0.8319` to `0.8372`; its held-out F1 was unchanged at `0.8449`. These results describe this fixed split and sampling plan.

### Clustering · common-sample comparison

| Algorithm | Clusters | Evaluation rows | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---:|---:|---:|---:|---:|
| K-Means | 3 | 3,000 | 0.2069 | 1.5907 | 743.27 |
| Agglomerative (Ward) | 3 | 3,000 | 0.1710 | 1.6598 | 575.60 |

The elbow and candidate silhouette curves select `K=3`. The clustering notebook also shows a dendrogram, a comparison of Ward/complete/average linkage, separate two-component PCA plots for both algorithms, a t-SNE plot for K-Means, and profiles expressed in original measurement units. The cluster IDs are arbitrary and are **not** official air-quality or health categories.

### Review 1 material

`notebooks/regression_review1.ipynb` and `notebooks/classification_review1_part_a.ipynb` are the provided Review 1 work. Their saved outputs remain available. They were not refitted as part of this Review 2 run; Review 2 classification is a fresh common-split comparison covering Parts A and B.

For the GUI, `src/regression.py` fits a separate Random Forest on 12 practical building and weather inputs from the same regression dataset. Its held-out test performance is **R² 0.4969, RMSE 41.51, MAE 20.94** on 15,152 buildings. It does not replace the ten-algorithm Review 1 notebook.

## Setup

Use Python 3.11–3.13. The Review 2 notebooks and GUI were verified with Python 3.13, scikit-learn 1.7.2, pandas 3.0.6, Streamlit 1.51.0, and the version ranges in `requirements.txt`.

```bash
cd review2_case_study
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open and **Run All** for these notebooks in order:

1. `notebooks/clustering.ipynb`
2. `notebooks/classification_review2.ipynb`

Both Review 2 notebooks have already been executed from a clean kernel and saved with visible outputs. Running them again refreshes `results/` and `models/`. To rebuild the regression model used by the GUI, run:

```bash
python -m src.regression
```

Then start the GUI:

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit. The GUI has three prediction tabs:

1. **Regression:** enter building and annual weather details to predict Site EUI.
2. **Classification:** enter a sample date and available water measurements to predict `Good`, `Moderate`, or `Poor`; blank measurements are imputed by the fitted model.
3. **Clustering:** enter an hourly pollution and weather reading to receive a K-Means cluster assignment and its average measurements.

Additional tabs display the clustering comparison, elbow curve, PCA scatterplot, and linkage comparison. No public deployment is included.

## Repository structure

```text
data/           Original supplied CSVs by track
notebooks/      Review 1 notebooks and fully executed Review 2 notebooks
src/            Shared, modular feature engineering, fitting, evaluation, and exports
models/         Fitted models used by the GUI
results/        Tables and PCA points exported by the notebooks
app/            Streamlit GUI
requirements.txt
```

The `src/build_notebooks.py` generator recreates the clean Review 2 notebook source. Running it replaces notebook outputs, so rerun both notebooks afterward.

## Scope and limitations

- The clustering application is an exploratory assignment tool. It does not predict a ground-truth air-quality label or provide health guidance.
- Hierarchical clusters are available only for the 3,000-row analysis sample; the GUI predicts K-Means clusters for new readings.
- The classification scores are measured on one fixed held-out split, with 12,000 labeled training rows used for each algorithm. They are not external validation scores.
- The regression GUI model uses 12 inputs so the form remains practical; its score is measured separately from the ten-model Review 1 comparison.
- The source folder was a file export without Git metadata or a remote URL. This project can be committed and pushed to the team's GitHub repository; a public URL and earlier commit history cannot be reconstructed from the export.

## AI assistance disclosure

OpenAI Codex assisted with code review, refactoring, notebook and GUI scaffolding, execution checks, and README preparation. The numeric results are generated by running the code on the supplied datasets. The team should independently review and write its own final scientific interpretation and presentation narrative for the viva.
