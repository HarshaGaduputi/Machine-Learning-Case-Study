"""Interactive predictions for all three case-study tracks."""

from datetime import date
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clustering import COMPASS, POLLUTANTS, WEATHER, engineer_features  # noqa: E402

st.set_page_config(page_title="Machine Learning Case Study", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp {background: #f4f7fb; color: #14243a;}
.block-container {max-width: 1160px; padding-top: 2rem;}
h1, h2, h3 {color: #153a59;}
div[data-testid="stMetric"] {background: white; border: 1px solid #dbe5ee;
  border-radius: 12px; padding: 1rem;}
div[data-testid="stMetric"] * {color: #153a59 !important;}
.hero {background: linear-gradient(110deg,#174464,#167e8c); color: white;
  padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.4rem;}
.hero h1, .hero p {color: white; margin: 0;}
.hero p {opacity: .9; margin-top: .4rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(name):
    path = ROOT / "models" / name
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_tables():
    results = ROOT / "results"
    paths = {
        "comparison": results / "clustering_comparison.csv",
        "elbow": results / "clustering_elbow.csv",
        "pca": results / "clustering_pca.csv",
        "linkages": results / "clustering_linkages.csv",
        "regression_test": results / "regression_gui_test_predictions.csv",
        "classification_confusion": results / "classification_gui_confusion.csv",
    }
    return {name: pd.read_csv(path) if path.exists() else None for name, path in paths.items()}


st.markdown("""<div class="hero"><h1>Machine Learning Case Study</h1>
<p>Predict building energy use, classify water quality, and explore Beijing air-quality clusters.</p></div>""", unsafe_allow_html=True)
artifact = load_model("clustering_model.joblib")
regression = load_model("regression_gui_model.joblib")
classification = load_model("classification_model.joblib")
tables = load_tables()
regression_metrics_path = ROOT / "results" / "regression_gui_metrics.csv"
regression_metrics = pd.read_csv(regression_metrics_path) if regression_metrics_path.exists() else None
classification_path = ROOT / "results" / "classification_comparison.csv"
classification_results = pd.read_csv(classification_path) if classification_path.exists() else None
comparison = tables["comparison"]
cols = st.columns(3)
cols[0].metric("Regression test R²", f"{regression_metrics.iloc[0]['R2']:.3f}" if regression_metrics is not None else "—")
cols[1].metric("Classification weighted F1", f"{classification_results.iloc[0]['Weighted F1']:.3f}" if classification_results is not None else "—")
cols[2].metric("K-Means silhouette", f"{comparison.iloc[0]['Silhouette Score']:.3f}" if comparison is not None else "—")

regression_tab, classification_tab, predict_tab, explore_tab, methods_tab = st.tabs([
    "Regression", "Classification", "Clustering", "Clustering results", "Methods and limits"
])

with regression_tab:
    st.subheader("Predict building Site Energy Use Intensity")
    st.caption("This app uses a separately fitted Random Forest with 12 form-friendly predictors from the Review 1 building dataset.")
    if regression is None:
        st.error("Regression model missing. Run `python -m src.regression` from the project folder.")
    else:
        defaults = regression["defaults"]
        categories = regression["categories"]
        modes = regression["modes"]
        with st.form("regression_form"):
            st.markdown("**Building**")
            c1, c2, c3 = st.columns(3)
            values = {}
            with c1:
                values["floor_area"] = st.number_input("Floor area", min_value=1.0, value=float(round(defaults["floor_area"], 1)))
                values["year_built"] = st.number_input("Year built", min_value=1700.0, max_value=2026.0, value=float(round(defaults["year_built"])))
            with c2:
                values["energy_star_rating"] = st.number_input("ENERGY STAR rating", min_value=0.0, max_value=100.0, value=float(round(defaults["energy_star_rating"], 1)))
                values["ELEVATION"] = st.number_input("Elevation", value=float(round(defaults["ELEVATION"], 1)))
            with c3:
                for field, label in [("State_Factor", "State category"), ("building_class", "Building class"), ("facility_type", "Facility type")]:
                    choices = categories[field]
                    values[field] = st.selectbox(label, choices, index=choices.index(modes[field]))
            st.markdown("**Annual weather**")
            w1, w2, w3 = st.columns(3)
            weather_fields = [
                ("avg_temp", "Average temperature"),
                ("cooling_degree_days", "Cooling degree days"),
                ("heating_degree_days", "Heating degree days"),
                ("precipitation_inches", "Precipitation (inches)"),
                ("snowfall_inches", "Snowfall (inches)"),
            ]
            for i, (field, label) in enumerate(weather_fields):
                with (w1, w2, w3)[i % 3]:
                    values[field] = st.number_input(
                        label, value=float(round(defaults[field], 2)),
                        min_value=0.0 if field != "avg_temp" else None,
                    )
            predict_regression = st.form_submit_button("Predict energy use", type="primary")
        if predict_regression:
            row = pd.DataFrame([values], columns=regression["features"])
            prediction = float(regression["model"].predict(row)[0])
            st.success(f"Predicted Site EUI: {prediction:.2f}")
            st.caption("This estimate is not a building audit or efficiency certification.")
            if tables["regression_test"] is not None:
                test = tables["regression_test"]
                st.markdown("**How this estimate compares with held-out buildings**")
                fig = px.histogram(
                    test, x="Actual Site EUI", nbins=60,
                    title="Actual energy use in the held-out test buildings",
                    color_discrete_sequence=["#167e8c"],
                )
                fig.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="#d9463e",
                              annotation_text="Your predicted EUI", annotation_position="top right")
                fig.update_layout(yaxis_title="Number of test buildings", showlegend=False)
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Bars show the {len(test):,} real held-out target values. The dashed line is the model's estimate for your input; its actual EUI is unknown.")

                sample = test.sample(n=min(3000, len(test)), random_state=42)
                fig = px.scatter(
                    sample, x="Actual Site EUI", y="Predicted Site EUI", opacity=0.35,
                    title="Model predictions versus actual energy use on held-out buildings",
                    color_discrete_sequence=["#174464"],
                )
                upper = float(max(sample["Actual Site EUI"].max(), sample["Predicted Site EUI"].max()))
                fig.add_shape(type="line", x0=0, y0=0, x1=upper, y1=upper,
                              line=dict(color="#d9463e", dash="dash"))
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Each dot is a real held-out building ({len(sample):,} sampled for display). The dashed diagonal is perfect prediction; your input has no known actual value to place on this plot.")
        if regression_metrics is not None:
            m = regression_metrics.iloc[0]
            st.info(f"Held-out evaluation: R² {m['R2']:.3f} · RMSE {m['RMSE']:.2f} · MAE {m['MAE']:.2f} · {int(m['Test Rows']):,} test buildings")

with classification_tab:
    st.subheader("Predict a water-quality class")
    st.caption("Enter the measurements you have. Blank fields use the median imputer fitted on training data. The model predicts one of the dataset's Good, Moderate, or Poor labels.")
    if classification is None:
        st.error("Classification model missing. Run the Review 2 classification notebook first.")
    else:
        date_features = {"sample_year", "sample_month", "sample_dayofyear", "month_sin", "month_cos"}
        measurement_fields = [f for f in classification["features"] if f not in date_features]
        primary_fields = [
            "pH", "temperature", "dissolved_oxygen", "electrical_conductance",
            "nitrate_nitrogen", "total_phosphorus", "oxygen_saturation", "Depth",
        ]
        other_fields = [f for f in measurement_fields if f not in primary_fields]
        with st.form("classification_form"):
            sample_date = st.date_input("Sample date", value=date(2015, 10, 16))
            st.markdown("**Common measurements**")
            entered = {}
            input_cols = st.columns(3)
            for i, field in enumerate(primary_fields):
                with input_cols[i % 3]:
                    entered[field] = st.text_input(field, placeholder="Leave blank if unknown")
            with st.expander("Additional measurements"):
                more_cols = st.columns(3)
                for i, field in enumerate(other_fields):
                    with more_cols[i % 3]:
                        entered[field] = st.text_input(field, placeholder="Optional")
            predict_classification = st.form_submit_button("Predict water quality", type="primary")
        if predict_classification:
            values, invalid = {}, []
            for field in measurement_fields:
                raw_value = entered[field].strip()
                if not raw_value:
                    values[field] = np.nan
                else:
                    try:
                        numeric_value = float(raw_value)
                        if not np.isfinite(numeric_value):
                            raise ValueError
                        values[field] = numeric_value
                    except ValueError:
                        invalid.append(field)
            if invalid:
                st.error("Enter valid numbers for: " + ", ".join(invalid))
            elif all(pd.isna(values[field]) for field in measurement_fields):
                st.error("Enter at least one water measurement before predicting.")
            else:
                month = sample_date.month
                values.update({
                    "sample_year": sample_date.year,
                    "sample_month": month,
                    "sample_dayofyear": sample_date.timetuple().tm_yday,
                    "month_sin": np.sin(2 * np.pi * month / 12),
                    "month_cos": np.cos(2 * np.pi * month / 12),
                })
                row = pd.DataFrame([values], columns=classification["features"])
                probabilities = classification["model"].predict_proba(row)[0]
                predicted_class = classification["class_names"][int(np.argmax(probabilities))]
                st.success(f"Predicted class: {predicted_class}")
                st.caption(f"{sum(pd.notna(values[f]) for f in measurement_fields)} measurements supplied; remaining fields were imputed. This is a model estimate, not a laboratory certification.")
                probability_table = pd.DataFrame({
                    "Class": classification["class_names"],
                    "Model probability": probabilities,
                })
                st.dataframe(probability_table.style.format({"Model probability": "{:.1%}"}),
                             width="stretch", hide_index=True)
                fig = px.bar(probability_table, x="Class", y="Model probability",
                             title="Model probabilities for your water sample",
                             color="Class", color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], showlegend=False)
                st.plotly_chart(fig, width="stretch")
                st.caption("These are the fitted model's probabilities for the supplied measurements, not measured water-quality outcomes.")
                if tables["classification_confusion"] is not None:
                    matrix = tables["classification_confusion"].set_index("Actual class")
                    fig = px.imshow(
                        matrix.to_numpy(), x=matrix.columns.tolist(), y=matrix.index.tolist(),
                        text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="Predicted class", y="Actual class", color="Test samples"),
                        title="Held-out water samples: actual versus predicted class",
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption(f"Counts come from {int(matrix.to_numpy().sum()):,} labeled samples in the original held-out test set; your new sample is not included.")
        if classification_results is not None:
            with st.expander("All ten classifier results"):
                st.dataframe(classification_results.round(4), width="stretch", hide_index=True)

with predict_tab:
    st.subheader("Enter one hourly observation")
    st.caption("The prediction is a cluster assignment based on similar measured conditions. It is not an official air-quality category or health advisory.")
    medians = artifact["raw_medians"]
    with st.form("reading"):
        c1, c2, c3 = st.columns(3)
        values = {}
        for i, name in enumerate(POLLUTANTS + WEATHER):
            with (c1, c2, c3)[i % 3]:
                values[name] = st.number_input(
                    name, value=float(round(medians[name], 2)),
                    min_value=0.0 if name in POLLUTANTS + ["RAIN", "WSPM"] else None,
                    step=1.0 if name in POLLUTANTS else 0.1,
                )
        with c1:
            values["month"] = st.selectbox("Month", range(1, 13), index=5)
        with c2:
            values["hour"] = st.selectbox("Hour", range(24), index=12)
        with c3:
            values["wd"] = st.selectbox("Wind direction", list(COMPASS), index=0)
        submitted = st.form_submit_button("Find cluster", type="primary")

    if submitted:
        reading = pd.DataFrame([values])
        feature_row = engineer_features(reading)
        scaled = artifact["preprocessing"].transform(feature_row)
        cluster = int(artifact["kmeans"].predict(scaled)[0])
        distances = artifact["kmeans"].transform(scaled)[0]
        profile = artifact["profile"].loc[cluster]
        st.success(f"This reading belongs to cluster {cluster}.")
        st.caption(f"Distance to this cluster centre in standardized feature space: {distances[cluster]:.2f}. Cluster numbers are arbitrary identifiers.")
        st.markdown("**Typical measurements in this cluster**")
        st.dataframe(profile.to_frame("Cluster average").T.round(2), width="stretch")
        if tables["pca"] is not None and "pca" in artifact:
            points = tables["pca"].copy()
            points["Cluster"] = points["K-Means Cluster"].astype(str)
            location = artifact["pca"].transform(scaled)[0]
            fig = px.scatter(
                points, x="PC1", y="PC2", color="Cluster", opacity=0.4,
                title="Your reading among 3,000 historical Beijing observations",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.add_scatter(x=[location[0]], y=[location[1]], mode="markers+text",
                            marker=dict(symbol="diamond", size=17, color="#d9463e",
                                        line=dict(width=2, color="white")),
                            text=["Your reading"], textposition="top center",
                            name="Your reading")
            fig.update_layout(xaxis_title="Principal component 1", yaxis_title="Principal component 2")
            st.plotly_chart(fig, width="stretch")
            st.caption("Historical points are the real analysis sample. The diamond is your entered reading projected with the same fitted PCA; clustering itself uses all 17 features.")

        ratios = pd.DataFrame({
            "Pollutant": POLLUTANTS,
            "Compared with cluster average": [values[name] / profile[name] for name in POLLUTANTS],
        })
        fig = px.bar(ratios, x="Pollutant", y="Compared with cluster average",
                     title="Your pollutant readings relative to this cluster's averages",
                     color_discrete_sequence=["#167e8c"])
        fig.add_hline(y=1, line_dash="dash", line_color="#d9463e")
        fig.update_layout(yaxis_title="Your reading ÷ cluster average")
        st.plotly_chart(fig, width="stretch")
        st.caption("A ratio of 1 means the entered measurement equals the observed cluster average. This is a comparison with the historical data, not an air-quality or safety threshold.")

with explore_tab:
    if comparison is not None:
        st.subheader("Algorithm comparison")
        st.dataframe(comparison.round(3), width="stretch", hide_index=True)
        st.caption("Both methods are evaluated on the same 3,000 scaled observations. K-Means is fitted on all records; agglomerative clustering is fitted on this sample.")
    if tables["elbow"] is not None:
        elbow = tables["elbow"]
        fig = px.line(elbow, x="K", y="Inertia", markers=True,
                      title="K-Means elbow: within-cluster inertia by K")
        fig.update_layout(xaxis_title="Number of clusters (K)", yaxis_title="Inertia")
        st.plotly_chart(fig, width="stretch")
    if tables["pca"] is not None:
        points = tables["pca"]
        algorithm = st.radio("Colour by", ["K-Means Cluster", "Agglomerative Cluster"], horizontal=True)
        points["Cluster"] = points[algorithm].astype(str)
        fig = px.scatter(points, x="PC1", y="PC2", color="Cluster", opacity=0.55,
                         title=f"Two-component PCA view: {algorithm}",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, width="stretch")
        st.caption("PCA is a 2D projection for display; clustering uses all 17 engineered features.")

with methods_tab:
    st.subheader("Regression")
    st.write("A Random Forest is fitted on 80% of the supplied building dataset using the 12 fields available in the form. Missing numerical values are median-imputed and categorical values are encoded using the training data only. The displayed R², RMSE, and MAE come from the remaining 20% test split.")
    st.subheader("Classification")
    st.write("The selected Random Forest is fitted on a stratified 12,000-row sample of labeled water-quality data. Its imputer handles measurements left blank in the form. Class probabilities are model outputs, and the comparison table reports scores on a common held-out test set.")
    st.subheader("Clustering")
    st.write("Pollutant concentrations are log transformed; hour, month and wind direction are represented cyclically. Missing values are filled with medians learned from the dataset, then all features are standardized before K-Means predicts the nearest cluster centre.")
    if tables["linkages"] is not None:
        st.markdown("**Hierarchical linkage comparison**")
        st.dataframe(tables["linkages"].round(3), width="stretch", hide_index=True)
    st.write("Clustering is unsupervised. No ground-truth class labels were used during fitting. Hierarchical clustering uses a sample because its time and memory costs grow rapidly with the number of observations. Results describe this historical dataset and should not be used as a health recommendation.")
