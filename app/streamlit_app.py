"""Interactive Review 2 dashboard for Beijing air-quality clusters."""

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

st.set_page_config(page_title="Air Quality Clusters", page_icon="🌤️", layout="wide")
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
def load_model():
    path = ROOT / "models" / "clustering_model.joblib"
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
    }
    return {name: pd.read_csv(path) if path.exists() else None for name, path in paths.items()}


st.markdown("""<div class="hero"><h1>Beijing Air Quality Clusters</h1>
<p>Explore environmental patterns across 12 monitoring stations, then assign a new reading to a K-Means cluster.</p></div>""", unsafe_allow_html=True)
artifact = load_model()
tables = load_tables()
if artifact is None:
    st.error("Trained model not found. Run notebooks/clustering.ipynb from top to bottom first.")
    st.stop()

comparison = tables["comparison"]
if comparison is not None:
    cols = st.columns(3)
    cols[0].metric("K-Means clusters", int(comparison.loc[0, "Clusters"]))
    cols[1].metric("Evaluation readings", int(comparison.loc[0, "Evaluation Rows"]))
    cols[2].metric("K-Means silhouette", f"{comparison.loc[0, 'Silhouette Score']:.3f}")

predict_tab, explore_tab, methods_tab = st.tabs(["Predict a cluster", "Explore results", "Method and limits"])

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
    st.subheader("How the assignment is made")
    st.write("Pollutant concentrations are log transformed; hour, month and wind direction are represented cyclically. Missing values are filled with medians learned from the dataset, then all features are standardized before K-Means predicts the nearest cluster centre.")
    if tables["linkages"] is not None:
        st.markdown("**Hierarchical linkage comparison**")
        st.dataframe(tables["linkages"].round(3), width="stretch", hide_index=True)
    st.write("Clustering is unsupervised. No ground-truth class labels were used during fitting. Hierarchical clustering uses a sample because its time and memory costs grow rapidly with the number of observations. Results describe this historical dataset and should not be used as a health recommendation.")
