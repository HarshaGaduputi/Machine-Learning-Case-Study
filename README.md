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

Use Python 3.13. The Review 2 notebooks and original GUI run were verified with Python 3.13.9; the saved models use scikit-learn 1.7.2. `requirements.txt` contains the proposed app runtime; `requirements-dev.txt` adds the notebook and plotting tools. The Streamlit 1.65.0 upgrade is **not yet validated in a fresh environment**; follow [DEPLOYMENT.md](DEPLOYMENT.md) before any release.

```bash
cd review2_case_study
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m jupyter lab
```

Open and **Run All** for these notebooks in order:

1. `notebooks/clustering.ipynb`
2. `notebooks/classification_review2.ipynb`

Both Review 2 notebooks have already been executed from a clean kernel and saved with visible outputs. Running them again refreshes `results/` and `models/`. To rebuild the regression model used by the GUI, run:

```bash
python -m src.regression
python -m src.app_diagnostics
python -m src.release_validation
python -m src.smoke_check
```

The second command rebuilds the GUI's held-out regression and classification charts from the saved models and original datasets. The third command calibrates the saved water classifier on previously unused training-partition records, selects an input-completeness gate on a separate subset, checks the original held-out partition, builds a regression error interval, checks cluster assignment stability under masked inputs, and saves `results/release_validation.json`. The final command verifies the model fingerprints and runs one real source example through each task. Run both diagnostic commands and the smoke check after refitting any affected model. The clustering notebook saves the PCA projection used to place a new reading on the historical scatterplot.

Then start the GUI:

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit. The GUI has three prediction views and one study-results view. The three labels at the top link directly to their prediction views and stay in sync with the task switcher. Inputs start blank instead of silently displaying dataset medians. Each predictor has a switch that starts the form with an actual row from its supplied dataset; you can edit that row before predicting. Unsupported inputs receive an explanation instead of a prediction. Accepted results include their model version, entered inputs, imputed fields, and limitations in a downloadable JSON record. The app does not save submitted input records server-side.

1. **Regression:** enter floor area, facility use, and at least eight of nine numeric building/weather fields. Building class is derived from source combinations. The output is Site EUI in kBtu/ft²/year with an empirical 90% internal error range and held-out charts. The interval radius is 46.35; 91.2% of 7,461 eligible held-out buildings were inside it. Input values beyond observed source minima/maxima are withheld.
2. **Classification:** enter a sample date and at least 12 of 24 measured water fields. The gate was chosen on previously unused training-partition records, where accuracy met 90% and Poor-label recall met 83%. On 4,992 eligible records in the original random held-out partition, accuracy was 90.2% and Poor recall 83.4%. Probabilities are calibrated using a separate 12,000-record subset and displayed alongside the held-out confusion matrix and reliability plot. A separate same-file time-split stress test trained on 12,000 records from 2010–2019 and evaluated on 15,070 records from 2020–2024: accuracy was **85.2% and Poor-label recall 64.6%**. The supplied water file does not document measurement units or label rules. Input guidance ranges are observed data quantiles, not safety limits.
3. **Clustering:** enter at least four measured pollutants, three weather readings, and two time/wind fields. Units follow the [UCI Beijing dataset](https://archive.ics.uci.edu/dataset/501/beijing). With that mask pattern, assignment matched the same complete historical reading in 91.9% of trials on 3,000 station records; this checks stability, not accuracy. Inputs outside observed ranges or beyond the historical 99th-percentile distance to a cluster centre are withheld. The result shows the K-Means cluster, the new reading projected among real historical observations, and entered pollutant readings compared with cluster averages.

The new input's true building EUI or water-quality class is unknown, so it is never shown as a measured outcome in the test charts. The clustering projection is a display of the entered reading; it is not a known air-quality label. Every historical plotted value comes from the supplied datasets and fitted models.

The **Study results** view contains the internal reliability checks, regression test metrics, consolidated ten-classifier table, clustering comparison, elbow curve, PCA scatterplot, linkage comparison, and method limits. No public deployment is included.

### Public-release status

This is a polished **research demonstration**, not a validated operational service. **Publication of the building and water datasets remains on hold pending data-owner permission.** Do not push this repository or upload its bundle to a public service while that decision is open. Public deployment also requires fresh-environment dependency and smoke checks. Before presenting it as a real-world predictor, obtain and document the water file's original source, measurement units, label construction rules, intended water use, and redistribution rights; verify the building dataset's public redistribution terms; and test each model on new representative data from the intended users. A random held-out split and masked-input stability check do not establish transport to a new site, year, or sampling method. The water time-split stress test also shows materially worse Poor-label recall. Do not market its class or probability as a drinking-water safety decision. The air clusters are patterns, not official AQI categories.

The deployment preparation and the exact remaining checks are in [DEPLOYMENT.md](DEPLOYMENT.md). Streamlit Community Cloud is the direct fit for this Python/WebSocket app; Python 3.13 must be selected explicitly. Keep model files from this trusted build under deployment control: `joblib.load` is unsafe for untrusted files. The [Beijing source](https://archive.ics.uci.edu/dataset/501/beijing) is CC BY 4.0 and requires attribution.

### Interface references

The form structure was checked against [ENERGY STAR's description of EUI](https://www.energystar.gov/buildings/benchmark/understand-metrics/what-eui), [USGS water-quality parameter conventions](https://water.usgs.gov/nwc/NWC/water_quality/tables/constituent.html), the [UCI Beijing dataset's field definitions and units](https://archive.ics.uci.edu/dataset/501/beijing), and [Streamlit's form behavior](https://docs.streamlit.io/develop/concepts/architecture/forms). These references guide labels and interaction only. Predictions use the local supplied datasets and saved models, not values from those websites.

## Repository structure

```text
data/           Original supplied CSVs by track
notebooks/      Review 1 notebooks and fully executed Review 2 notebooks
src/            Shared, modular feature engineering, fitting, evaluation, and exports
models/         Fitted models used by the GUI
results/        Tables, PCA points, and internal release evidence
app/            Streamlit GUI
requirements.txt
requirements-dev.txt
```

The `src/build_notebooks.py` generator writes **unexecuted drafts** only when given `--output-dir`; it refuses to replace existing notebooks unless `--force` is explicitly provided. Keep the executed review notebooks intact.

## Scope and limitations

- The clustering application is an exploratory assignment tool. It does not predict a ground-truth air-quality label or provide health guidance.
- Hierarchical clusters are available only for the 3,000-row analysis sample; the GUI predicts K-Means clusters for new readings.
- The classification scores are measured on one fixed held-out split, with 12,000 labeled training rows used for each algorithm. They are not external validation scores.
- The regression GUI model uses 12 inputs so the form remains practical; its score is measured separately from the ten-model Review 1 comparison.
- The source folder was a file export without Git metadata or a remote URL. This project can be committed and pushed to the team's GitHub repository; a public URL and earlier commit history cannot be reconstructed from the export.

## AI assistance disclosure

OpenAI Codex assisted with code review, refactoring, notebook and GUI scaffolding, execution checks, and README preparation. The numeric results are generated by running the code on the supplied datasets. The team should independently review and write its own final scientific interpretation and presentation narrative for the viva.
