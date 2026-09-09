# Urban Energy Consumption & Efficiency Analysis

An end-to-end machine learning case study on building energy efficiency, built on a single unified real-world dataset. The project predicts a building's Site Energy Use Intensity (EUI) through regression, classifies buildings into efficiency bands, and is being extended to cluster buildings by consumption pattern.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Setup Instructions](#5-setup-instructions)
6. [Usage](#6-usage)
7. [Team and Contribution Timeline](#7-team-and-contribution-timeline)
8. [Known Limitations](#8-known-limitations)
9. [Roadmap](#9-roadmap)


---

## 1. Project Overview

Buildings account for a large share of global energy consumption, and Site Energy Use Intensity (EUI) — energy used per square foot of floor area — is one of the standard metrics real estate owners, city planners, and sustainability programs use to compare buildings and prioritize retrofits. Two buildings can look similar on paper and still perform very differently once age, occupancy, climate exposure, and construction type are accounted for, which is what makes EUI a genuinely hard number to predict from static characteristics alone.

This project works with a large real-world dataset of roughly 76,000 U.S. buildings, spanning multiple states and several years, that combines structural attributes (floor area, age, facility type, building class, ENERGY STAR rating) with local weather conditions (temperature, degree-days, wind, precipitation) for each building's observed year. The aim is to understand what actually drives energy performance in this data, produce a model that predicts EUI accurately enough to be useful, and translate that into a simple, actionable efficiency label. A further extension — grouping buildings into natural consumption segments rather than a single predicted number — is intended to make the analysis more directly useful for identifying which buildings are worth prioritizing for intervention.

## 2. Dataset

| Property | Detail |
|---|---|
| Source | WiDS Datathon 2022 (Kaggle) — https://www.kaggle.com/competitions/widsdatathon2022/data |
| File used | `train.csv` |
| Size | 75,757 rows, 64 columns |
| Target variable | `site_eui` (Site Energy Use Intensity) |
| Target statistics | Mean 82.58, standard deviation 58.26, minimum 1.00, maximum 997.87, skew 4.74 |

### Feature groups

- **Building characteristics:** `floor_area`, `year_built`, `facility_type`, `building_class`, `energy_star_rating`, `State_Factor`
- **Weather data:** monthly minimum/average/maximum temperatures, heating and cooling degree days, wind speed and direction, precipitation, days with fog

### Data quality issues and handling

- Target variable is heavily right-skewed (skew 4.74); a `log1p` transform (`log_site_eui`) is used as the modeling target, with predictions back-transformed to the original scale for evaluation.
- Columns with more than 60% missing values are dropped entirely (e.g. `direction_max_wind_speed`, `days_with_fog`, `direction_peak_wind_speed`, `max_wind_speed`).
- Remaining missing values: numeric columns imputed with the training-set median, categorical columns imputed with the training-set mode. Imputers are fit on the training split only and applied to the test split to avoid leakage.
- Outliers in `site_eui` are capped using the 1.5x IQR rule prior to the log transform.

## 3. Methodology

### 3.1 Exploratory Data Analysis

- Distribution plot of `site_eui` and its log-transformed version, to justify the log transform
- Correlation heatmap of numeric features against the target
- Distribution plots across all numeric features
- Categorical breakdowns: count plot of `facility_type` (top 20) and `building_class`
- Boxplot of `site_eui` by `facility_type` (top 15)
- Scatter plots: `floor_area` vs. `site_eui`, `energy_star_rating` vs. `site_eui`, `cooling_degree_days` vs. `site_eui`
- Missing-value heatmap across all features

### 3.2 Preprocessing and Feature Engineering

- Dropped `id`, `Year_Factor`, and all columns exceeding 60% missingness
- Removed exact duplicate rows
- Capped `site_eui` outliers using the 1.5x IQR rule, then applied `log1p`
- Engineered features:
  - `age_of_building` = 2022 − `year_built`
  - `floor_area_log` = `log1p(floor_area)`
  - `hdd_cdd_ratio` = `heating_degree_days / (cooling_degree_days + 1)`
  - `energy_star_missing_flag` — binary indicator for missing `energy_star_rating`
- Train/test split: 80:20, `random_state=42`, on the continuous log-target
- A **second, separately stratified** 80:20 split was created for the classification track (stratified on the binarized target), since the continuous regression target cannot be stratified. Imputers, encoders, and scalers for this split are fit independently on its own training partition.
- Encoding: ordinal encoding for low-cardinality categoricals; target encoding (`TargetEncoder`, fit on the log target) for `facility_type`
- Scaling: `StandardScaler`, fit on the training partition only

### 3.3 Regression Track (10 algorithms)

All models are evaluated on `R²`, `RMSE`, and `MAE`, computed after back-transforming predictions from log scale to the original EUI scale.

| # | Algorithm | Configuration |
|---|---|---|
| 1 | Linear Regression | Default (OLS baseline) |
| 2 | Ridge Regression | `GridSearchCV` over `alpha ∈ {0.01, 0.1, 1, 10, 100}`, 5-fold CV |
| 3 | Lasso Regression | `GridSearchCV` over `alpha ∈ {0.001, 0.01, 0.1, 1}`, 5-fold CV |
| 4 | ElasticNet | `GridSearchCV` over `alpha ∈ {0.01, 0.1, 1}`, `l1_ratio ∈ {0.2, 0.5, 0.8}`, 3-fold CV |
| 5 | Polynomial Regression | Degree 1 vs. degree 2 comparison via `PolynomialFeatures` + `LinearRegression` pipeline |
| 6 | Decision Tree Regressor | `GridSearchCV` over `max_depth ∈ {3, 5, 7, 10, None}`, 5-fold CV |
| 7 | Random Forest Regressor | `n_estimators=100`, `max_depth=10` |
| 8 | Gradient Boosting Regressor | `n_estimators=100`, `learning_rate=0.1` |
| 9 | Support Vector Regressor | RBF kernel, `GridSearchCV` over `C ∈ {1, 10}`, `epsilon=0.1`, `max_iter=1000` (capped for tractability on 75k rows), 3-fold CV |
| 10 | KNN Regressor | `GridSearchCV` over `n_neighbors ∈ {5, 10, 15}`, 3-fold CV |

Additional analysis: 5-fold cross-validated R² for the top 2 models by test R²; a before/after hyperparameter tuning comparison (default vs. `GridSearchCV`-tuned, with delta R²) for Ridge, Lasso, ElasticNet, Decision Tree, and KNN; residual and feature-importance visualizations for the best-performing model.

### 3.4 Classification Track (5 algorithms)

`site_eui` is binarized at the dataset median into two classes (0 = below median / Low EUI, 1 = above median / High EUI). All models are evaluated on Accuracy, Precision, Recall, Weighted F1, and ROC-AUC.

| # | Algorithm | Configuration |
|---|---|---|
| 1 | Logistic Regression | Default baseline |
| 2 | KNN Classifier | Tuned over a range of `k` values |
| 3 | Naive Bayes | Gaussian Naive Bayes, default |
| 4 | Decision Tree Classifier | Tuned over `max_depth` |
| 5 | Support Vector Classifier | RBF kernel |

## 4. Results

### 4.1 Regression — Test Set Performance

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | Random Forest | 0.5029 | 25.66 | 17.48 |
| 2 | Gradient Boosting | 0.4834 | 26.16 | 17.85 |
| 3 | Decision Tree | 0.4718 | 26.45 | 17.96 |
| 4 | KNN Regressor | 0.4449 | 27.12 | 18.84 |
| 5 | ElasticNet | 0.4046 | 28.09 | 19.90 |
| 6 | Lasso Regression | 0.4026 | 28.13 | 19.92 |
| 7 | Ridge Regression | 0.4012 | 28.17 | 19.92 |
| 8 | Linear Regression | 0.4005 | 28.18 | 19.93 |
| 9 | SVR | -2.0062 | 63.11 | 54.07 |
| 10 | Polynomial Regression (degree 2) | approx. -5.97 × 10¹⁶ | approx. 8.89 × 10⁹ | approx. 7.23 × 10⁷ |

5-fold cross-validated R² on the top 2 models by test performance:

| Model | CV R² (mean) | CV R² (std) |
|---|---|---|
| Random Forest | 0.5566 | 0.0105 |
| Gradient Boosting | 0.5278 | 0.0102 |

**Best model: Random Forest**, with test R² of 0.5029 and cross-validated R² of 0.5566 ± 0.0105. Ensemble tree methods (Random Forest, Gradient Boosting, Decision Tree) consistently outperform linear models, indicating the underlying relationships between building/weather features and EUI are non-linear. SVR underperforms because its iteration cap (`max_iter=1000`, required for tractability on 75,000 rows) prevents convergence. Degree-2 Polynomial Regression fails catastrophically: expanding 64 features to over 2,000 polynomial terms causes severe multicollinearity and numerical instability, producing a nonsensical negative R².

### 4.2 Classification — Test Set Performance

| Rank | Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | KNN | 0.7645 | 0.7648 | 0.7645 | 0.7645 | 0.8493 |
| 2 | Decision Tree | 0.7637 | 0.7640 | 0.7637 | 0.7637 | 0.8501 |
| 3 | Logistic Regression | 0.7524 | 0.7583 | 0.7524 | 0.7510 | 0.8371 |
| 4 | SVC | 0.6847 | 0.7003 | 0.6847 | 0.6784 | 0.7603 |
| 5 | Naive Bayes | 0.5924 | 0.6539 | 0.5924 | 0.5472 | 0.6816 |

**Best model: KNN** on accuracy (76.45%); **Decision Tree** narrowly leads on ROC-AUC (0.8501). The two are close enough to be considered comparable, and either is a reasonable pick depending on whether the priority is a fixed-threshold decision (favor accuracy) or ranking confidence across thresholds (favor ROC-AUC). Naive Bayes performs noticeably worse than the other four, likely because its conditional-independence assumption is violated by the correlated weather features (e.g. heating and cooling degree days, monthly temperatures).

## 5. Setup Instructions

### 5.1 Prerequisites

- Python 3.10 or later
- `train.csv` from the WiDS Datathon 2022 Kaggle competition, placed in the same working directory as the notebook (or update the read path in the "Load dataset" cell if you place it elsewhere)

### 5.2 Installation

```bash
git clone https://github.com/<your-username>/urban-energy-efficiency-analysis.git
cd urban-energy-efficiency-analysis

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## 6. Usage

1. Place `train.csv` in the same directory as the notebook.
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open the notebook and run all cells in order from top to bottom. Later cells — comparative evaluation tables, cross-validation, hyperparameter tuning summaries, and visualizations — depend on model objects fitted earlier in the same run.

## 7. Team and Contribution Timeline

The regression and classification work was divided across three contributors, with each stage building on the previous one's committed work.

| Stage | Contributor | Scope |
|---|---|---|
| 1 | Person 1 | Dataset loading, EDA, data cleaning, feature engineering, train/test split, and the first five regression algorithms (Linear, Ridge, Lasso, ElasticNet, Polynomial) |
| 2 | Person 2 | The remaining five regression algorithms (Decision Tree, Random Forest, Gradient Boosting, SVR, KNN), comparative evaluation, cross-validation, hyperparameter tuning, and result visualizations |
| 3 | Person 3 | The full classification track (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVC) and its evaluation summary |

Submission deadline: September 17.

## 8. Known Limitations

- **Polynomial Regression (degree 2)** is numerically unstable on this dataset (feature count explodes from 64 to over 2,000 terms) and its result should be treated as a negative finding rather than a usable model.
- **SVR** is capped at 1,000 iterations for tractability on 75,000 rows, which prevents full convergence and understates its achievable performance.
- The best regression R² achieved (0.50) leaves roughly half the variance in `site_eui` unexplained; there is meaningful room for improvement through richer feature engineering (e.g. interaction terms, non-linear transforms of weather variables) or more extensive hyperparameter search.
- The classification target is a simple median split rather than a domain-defined efficiency standard (e.g. ENERGY STAR bands), so "Low/High EUI" reflects the dataset's own distribution rather than an external benchmark.

## 9. Roadmap

A high-level view of how the project actually progressed, from raw data to final evaluation.

```
                              Urban Energy Efficiency Analysis
                                            │
        ┌───────────────┬───────────────────┼───────────────────┬───────────────┐
        │               │                   │                   │               │
  Dataset Load       Preprocessing     Regression Track    Classification    Evaluation
    & EDA           & Feature Eng.      (10 algorithms)         Track       & Comparison
        │               │                   │                   │               │
   Shape, dtypes    Outlier capping    Linear · Ridge      Median split    Best regressor:
   & missing-value    (IQR) + log      Lasso · ElasticNet   into Low/High   Random Forest
   audit              transform        Polynomial              EUI          (R² 0.50)
        │               │                   │                   │               │
   Distribution     age_of_building,   Decision Tree ·      Logistic Reg. ·  Best classifier:
   plots, corr.     floor_area_log,    Random Forest ·      KNN · Naive      KNN (76.5% acc.)
   heatmap,         hdd_cdd_ratio,     Gradient Boosting     Bayes · SVC
   category &       star_missing_flag       │                   │
   scatter plots         │            SVR · KNN            Accuracy,
        │            Train/test          │                Precision,
   Missing-value     split (80:20)   Cross-validation      Recall, F1,
   heatmap          + stratified      + GridSearchCV        ROC-AUC
                     split for clf.    tuning
```

