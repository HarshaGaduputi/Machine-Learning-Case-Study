"""Minimal local interface for the three saved case-study models."""

from __future__ import annotations

from calendar import month_name
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.clustering import COMPASS, POLLUTANTS, WEATHER, engineer_features

ROOT = Path(__file__).resolve().parents[1]
TEAL = "#176d69"
INK = "#18313c"
CORAL = "#cf6249"

DATE_FEATURES = {"sample_year", "sample_month", "sample_dayofyear", "month_sin", "month_cos"}
COMMON_WATER = [
    "pH", "temperature", "dissolved_oxygen", "electrical_conductance",
    "nitrate_nitrogen", "total_phosphorus", "oxygen_saturation", "Depth",
]
WATER_LABELS = {
    "pH": "pH", "temperature": "Water temperature", "dissolved_oxygen": "Dissolved oxygen",
    "electrical_conductance": "Electrical conductance", "nitrate_nitrogen": "Nitrate nitrogen",
    "total_phosphorus": "Total phosphorus", "oxygen_saturation": "Oxygen saturation",
    "Depth": "Sample depth", "total_suspended_solids": "Total suspended solids",
    "total_dissolved_solids": "Total dissolved solids",
    "biochemical_oxygen_demand": "Biochemical oxygen demand",
    "chemical_oxygen_demand": "Chemical oxygen demand",
}
AIR_LABELS = {
    **{name: f"{name} (µg/m³)" for name in POLLUTANTS},
    "TEMP": "Temperature (°C)", "PRES": "Pressure (hPa)",
    "DEWP": "Dew point (°C)", "RAIN": "Rain (mm)",
    "WSPM": "Wind speed (m/s)",
}


@st.cache_resource
def load_model(filename: str):
    path = ROOT / "models" / filename
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame | None]:
    names = {
        "regression_metrics": "regression_gui_metrics.csv",
        "regression_test": "regression_gui_test_predictions.csv",
        "classification_results": "classification_comparison.csv",
        "classification_confusion": "classification_gui_confusion.csv",
        "clustering_comparison": "clustering_comparison.csv",
        "clustering_elbow": "clustering_elbow.csv",
        "clustering_pca": "clustering_pca.csv",
        "clustering_linkages": "clustering_linkages.csv",
    }
    return {
        name: pd.read_csv(ROOT / "results" / filename)
        if (ROOT / "results" / filename).exists() else None
        for name, filename in names.items()
    }


@st.cache_data
def building_reference(features: tuple[str, ...]):
    """Use actual dataset combinations and ranges for the building form."""
    data = pd.read_csv(ROOT / "data" / "regression" / "train.csv", usecols=list(features))
    pairs = data[["facility_type", "building_class"]].dropna().drop_duplicates()
    if pairs.groupby("facility_type")["building_class"].nunique().gt(1).any():
        raise ValueError("One facility type maps to multiple building classes")
    facility_class = pairs.set_index("facility_type")["building_class"].to_dict()
    numeric = data.select_dtypes(include="number")
    bounds = {name: (float(numeric[name].min()), float(numeric[name].max())) for name in numeric}
    return facility_class, data.iloc[0].to_dict(), bounds


@st.cache_data
def water_example() -> dict:
    source = ROOT / "data" / "classification" / "water_quality_classification_dataset.csv"
    return pd.read_csv(source, nrows=1).iloc[0].to_dict()


@st.cache_data
def water_reference_ranges(fields: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """Observed central ranges guide entry without pretending to define safe limits."""
    source = ROOT / "data" / "classification" / "water_quality_classification_dataset.csv"
    data = pd.read_csv(source, usecols=list(fields))
    quantiles = data.quantile([0.01, 0.99])
    return {name: (float(quantiles.at[0.01, name]), float(quantiles.at[0.99, name]))
            for name in fields}


@st.cache_data
def air_example() -> tuple[dict, str]:
    files = sorted((ROOT / "data" / "clustering").glob("PRSA_Data_*_20130301-20170228.csv"))
    if not files:
        raise FileNotFoundError("No Beijing source station files were found")
    source = pd.read_csv(files[0], nrows=100)
    row = source.dropna(subset=POLLUTANTS + WEATHER + ["month", "hour", "wd"]).iloc[0]
    stamp = f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d} {int(row['hour']):02d}:00"
    return row.to_dict(), f"{row['station']} · {stamp}"


def source_number(example: dict, name: str) -> float | None:
    value = example.get(name)
    return float(value) if value is not None and pd.notna(value) else None


def source_text(example: dict, name: str) -> str:
    value = source_number(example, name)
    return f"{value:.12g}" if value is not None else ""


def pretty_name(value: str) -> str:
    return value.replace("_", " ").replace("2to4", "2–4").replace("5plus", "5+")


def draw(fig, *, height: int = 365) -> None:
    fig.update_layout(
        template="plotly_white", height=height,
        font=dict(color=INK, size=13),
        title=dict(font=dict(size=16, color=INK)),
        margin=dict(l=16, r=16, t=60, b=24),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def draw_regression_charts(test: pd.DataFrame, prediction: float) -> None:
    st.markdown("#### Compare with held-out buildings")
    cap = max(float(test["Actual Site EUI"].quantile(0.99)), prediction * 1.05)
    shown = test.loc[test["Actual Site EUI"] <= cap]
    fig = px.histogram(shown, x="Actual Site EUI", nbins=45,
                       title="Actual energy use in the test set",
                       color_discrete_sequence=[TEAL])
    fig.add_vline(x=prediction, line_width=2, line_dash="dash", line_color=CORAL,
                  annotation_text="Your estimate", annotation_position="top right")
    fig.update_layout(showlegend=False, yaxis_title="Buildings")
    draw(fig)
    hidden = len(test) - len(shown)
    st.caption(f"{len(shown):,} measured test values shown; {hidden:,} high values outside this view. "
               "The line is your model estimate, not a measured outcome.")

    sample = test.sample(n=min(3000, len(test)), random_state=42)
    fig = px.scatter(sample, x="Actual Site EUI", y="Predicted Site EUI", opacity=0.36,
                     title="Model predictions versus measured energy use",
                     color_discrete_sequence=[TEAL])
    upper = float(max(sample["Actual Site EUI"].max(), sample["Predicted Site EUI"].max()))
    fig.add_shape(type="line", x0=0, y0=0, x1=upper, y1=upper,
                  line=dict(color=CORAL, dash="dash"))
    draw(fig)
    st.caption(f"{len(sample):,} real held-out buildings sampled for display. The dashed diagonal is perfect prediction. "
               "Your new building has no known actual EUI to plot here.")


def regression_tab(model: dict | None, tables: dict) -> None:
    st.subheader("Building energy · regression")
    st.caption("Estimate annual Site Energy Use Intensity using the saved building model. "
               "Enter measurements you know; unknown optional fields use training-data imputers.")
    if model is None:
        st.error("Building model missing. Run `python -m src.regression`.")
        return

    facility_class, example_row, bounds = building_reference(tuple(model["features"]))
    example_mode = st.toggle("Start from a real building in the supplied dataset", key="reg_example")
    example = example_row if example_mode else {}
    mode = "example" if example_mode else "manual"
    if example_mode:
        st.caption("Source row 1 of `data/regression/train.csv`. Edit any value to create your own input.")

    with st.form(f"building_{mode}"):
        st.markdown("**Building details**")
        a, b = st.columns(2)
        with a:
            floor_area = st.number_input("Floor area (ft²) *", min_value=1.0,
                                         value=source_number(example, "floor_area"), step=100.0,
                                         key=f"reg_{mode}_floor")
            facility_options = sorted(model["categories"]["facility_type"], key=pretty_name)
            default_facility = example.get("facility_type")
            facility = st.selectbox(
                "Facility use *", facility_options,
                index=facility_options.index(default_facility) if default_facility in facility_options else None,
                placeholder="Select a building use", format_func=pretty_name,
                key=f"reg_{mode}_facility",
                help="Building class is inferred from this use type using combinations present in the dataset.",
            )
        with b:
            year_built = st.number_input("Year built", min_value=1700, max_value=date.today().year,
                                         value=int(example["year_built"]) if example_mode and pd.notna(example.get("year_built")) else None,
                                         step=1, key=f"reg_{mode}_year")
            rating = st.number_input("ENERGY STAR rating", min_value=0.0, max_value=100.0,
                                     value=source_number(example, "energy_star_rating"), step=1.0,
                                     key=f"reg_{mode}_rating",
                                     help="Optional. Leave blank if your building has no score.")

        with st.expander("Location and annual weather · optional"):
            st.caption("Use observed annual values when available. The region codes are anonymized in this dataset.")
            c1, c2 = st.columns(2)
            with c1:
                region_options = model["categories"]["State_Factor"]
                default_region = example.get("State_Factor")
                region = st.selectbox(
                    "Dataset region code", region_options,
                    index=region_options.index(default_region) if default_region in region_options else None,
                    placeholder="Unknown", key=f"reg_{mode}_region",
                    help="These are anonymized codes, not real state names.",
                )
                elevation = st.number_input("Elevation (m)", value=source_number(example, "ELEVATION"),
                                            step=1.0, key=f"reg_{mode}_elevation")
                avg_temp = st.number_input("Average annual temperature (°F)",
                                           value=source_number(example, "avg_temp"), step=1.0,
                                           key=f"reg_{mode}_temp")
            with c2:
                cooling = st.number_input("Cooling degree days (°F·days)", min_value=0.0,
                                          value=source_number(example, "cooling_degree_days"), step=1.0,
                                          key=f"reg_{mode}_cooling")
                heating = st.number_input("Heating degree days (°F·days)", min_value=0.0,
                                          value=source_number(example, "heating_degree_days"), step=1.0,
                                          key=f"reg_{mode}_heating")
                precipitation = st.number_input("Annual precipitation (inches)", min_value=0.0,
                                                value=source_number(example, "precipitation_inches"), step=0.1,
                                                key=f"reg_{mode}_precip")
                snowfall = st.number_input("Annual snowfall (inches)", min_value=0.0,
                                           value=source_number(example, "snowfall_inches"), step=0.1,
                                           key=f"reg_{mode}_snow")
        submitted = st.form_submit_button("Estimate Site EUI", type="primary")

    if not submitted:
        return
    if floor_area is None or facility is None:
        st.error("Enter floor area and select a facility use before estimating.")
        return
    values = {
        "floor_area": floor_area, "facility_type": facility,
        "building_class": facility_class[facility], "year_built": year_built,
        "energy_star_rating": rating, "State_Factor": region,
        "ELEVATION": elevation, "avg_temp": avg_temp,
        "cooling_degree_days": cooling, "heating_degree_days": heating,
        "precipitation_inches": precipitation, "snowfall_inches": snowfall,
    }
    row = pd.DataFrame([values], columns=model["features"])
    prediction = float(model["model"].predict(row)[0])
    missing = int(row.isna().sum(axis=1).iloc[0])
    outside = [name for name, value in values.items()
               if name in bounds and value is not None and not bounds[name][0] <= value <= bounds[name][1]]

    st.divider()
    st.metric("Estimated Site EUI", f"{prediction:.1f} kBtu/ft²/year")
    input_note = (f"{missing} of {len(model['features'])} model inputs were missing and imputed."
                  if missing else "All model inputs were supplied.")
    st.caption(f"Building class: {values['building_class']}. {input_note}")
    if missing >= 6:
        st.warning("Many inputs were unknown. Add observed location and weather values if you have them.")
    if outside:
        st.warning("Outside the supplied building data's observed range: " + ", ".join(outside) + ".")
    st.caption("This is a model estimate, not a utility-bill measurement or an ENERGY STAR certification.")
    if tables["regression_test"] is not None:
        draw_regression_charts(tables["regression_test"], prediction)


def parse_water(entered: dict[str, str]) -> tuple[dict, list[str]]:
    values, errors = {}, []
    for field, raw in entered.items():
        if not raw.strip():
            values[field] = np.nan
            continue
        try:
            number = float(raw)
        except ValueError:
            errors.append(WATER_LABELS.get(field, pretty_name(field)))
            continue
        if not np.isfinite(number) or (field != "temperature" and number < 0) or (field == "pH" and number > 14):
            errors.append(WATER_LABELS.get(field, pretty_name(field)))
        else:
            values[field] = number
    return values, errors


def classification_tab(model: dict | None, tables: dict) -> None:
    st.subheader("Water quality · classification")
    st.caption("Predict the supplied dataset's Good, Moderate, or Poor label. "
               "Use measured values in the source dataset's scale; its file does not document measurement units.")
    if model is None:
        st.error("Water-quality model missing. Run the Review 2 classification notebook.")
        return

    source = water_example()
    example_mode = st.toggle("Start from a row in the supplied water dataset", key="water_example")
    example = source if example_mode else {}
    mode = "example" if example_mode else "manual"
    if example_mode:
        st.caption("Source row 1 of `data/classification/water_quality_classification_dataset.csv`. "
                   "Edit any value to create your own input.")

    measurements = [name for name in model["features"] if name not in DATE_FEATURES]
    ranges = water_reference_ranges(tuple(measurements))
    common = [name for name in COMMON_WATER if name in measurements]
    advanced = [name for name in measurements if name not in common]
    with st.form(f"water_{mode}"):
        raw_date = pd.to_datetime(example.get("Sample Date"), errors="coerce") if example_mode else pd.NaT
        sample_date = st.date_input("Sample date · optional", value=raw_date.date() if pd.notna(raw_date) else None,
                                    key=f"water_{mode}_date")
        st.markdown("**Common measurements**")
        entered = {}
        columns = st.columns(2)
        for index, field in enumerate(common):
            with columns[index % 2]:
                entered[field] = st.text_input(
                    WATER_LABELS.get(field, pretty_name(field)), value=source_text(example, field),
                    placeholder="Leave blank if unknown", key=f"water_{mode}_{field}",
                    help=(f"Model field: {field}. Observed 1st–99th percentile in the supplied data: "
                          f"{ranges[field][0]:.3g}–{ranges[field][1]:.3g}. These are not safety limits."),
                )
        with st.expander(f"Additional measurements · {len(advanced)} optional fields"):
            more_columns = st.columns(2)
            for index, field in enumerate(advanced):
                with more_columns[index % 2]:
                    entered[field] = st.text_input(
                        WATER_LABELS.get(field, pretty_name(field)), value=source_text(example, field),
                        placeholder="Leave blank if unknown", key=f"water_{mode}_{field}",
                        help=(f"Model field: {field}. Observed 1st–99th percentile in the supplied data: "
                              f"{ranges[field][0]:.3g}–{ranges[field][1]:.3g}. These are not safety limits."),
                    )
        submitted = st.form_submit_button("Classify water sample", type="primary")

    if not submitted:
        return
    values, errors = parse_water(entered)
    if errors:
        st.error("Check these measurements (numeric, finite, nonnegative except temperature; pH up to 14): "
                 + ", ".join(errors) + ".")
        return
    provided = sum(pd.notna(values[name]) for name in measurements)
    if provided == 0:
        st.error("Enter at least one measured water-quality value before classifying.")
        return
    if sample_date is None:
        values.update({name: np.nan for name in DATE_FEATURES})
    else:
        month = sample_date.month
        values.update({
            "sample_year": sample_date.year, "sample_month": month,
            "sample_dayofyear": sample_date.timetuple().tm_yday,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        })
    row = pd.DataFrame([values], columns=model["features"])
    missing_features = int(row.isna().sum(axis=1).iloc[0])
    probabilities = model["model"].predict_proba(row)[0]
    code = int(model["model"].predict(row)[0])
    predicted = model["class_names"][code]

    st.divider()
    st.metric("Predicted dataset class", predicted)
    input_note = (f"{missing_features} model inputs, including any unknown date fields, were imputed."
                  if missing_features else "All model inputs were supplied.")
    st.caption(f"{provided} of {len(measurements)} measurements supplied. {input_note}")
    if provided < 5:
        st.warning("Most measurements are missing. Treat this classification cautiously.")
    unusual = [field for field in measurements if pd.notna(values[field])
               and not ranges[field][0] <= values[field] <= ranges[field][1]]
    if unusual:
        st.info("Unusual relative to the supplied data's central 98%: "
                + ", ".join(WATER_LABELS.get(field, pretty_name(field)) for field in unusual)
                + ". Check the entered values and units.")
    st.caption("The labels are from the supplied dataset; this is not an official water-safety assessment.")

    probability_table = pd.DataFrame({"Class": model["class_names"], "Model probability": probabilities})
    fig = px.bar(probability_table, x="Class", y="Model probability", color="Class",
                 title="Model probability estimates for this input",
                 color_discrete_sequence=["#7a9e9b", TEAL, "#2b4e59"])
    fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1.08])
    draw(fig)
    st.caption("These percentages are model outputs. They do not measure the true water-quality class.")

    if tables["classification_confusion"] is not None:
        matrix = tables["classification_confusion"].set_index("Actual class")
        fig = px.imshow(matrix.to_numpy(), x=matrix.columns.tolist(), y=matrix.index.tolist(),
                        text_auto=True, aspect="auto", color_continuous_scale="Teal",
                        labels=dict(x="Predicted class", y="Actual class", color="Samples"),
                        title="Actual versus predicted labels on held-out samples")
        draw(fig)
        st.caption(f"Counts from {int(matrix.to_numpy().sum()):,} labeled test samples in the supplied file. "
                   "Your new sample is not included.")


def clustering_tab(model: dict | None, tables: dict) -> None:
    st.subheader("Beijing air readings · clustering")
    st.caption("Find the fitted K-Means cluster for an hourly reading. "
               "Cluster numbers describe similar historical observations, not official air-quality categories.")
    if model is None:
        st.error("Clustering model missing. Run the Review 2 clustering notebook.")
        return

    source, source_label = air_example()
    example_mode = st.toggle("Start from a real Beijing station reading", key="air_example")
    example = source if example_mode else {}
    mode = "example" if example_mode else "manual"
    if example_mode:
        st.caption(f"Source: {source_label}. Edit any value to create your own input.")

    with st.form(f"air_{mode}"):
        st.markdown("**Pollutants**")
        st.caption("Concentrations use µg/m³, as documented by the Beijing dataset.")
        values = {}
        columns = st.columns(3)
        for index, name in enumerate(POLLUTANTS):
            with columns[index % 3]:
                values[name] = st.number_input(
                    AIR_LABELS[name], min_value=0.0, value=source_number(example, name),
                    step=1.0, key=f"air_{mode}_{name}",
                )
        with st.expander("Weather and observation time · optional"):
            weather_columns = st.columns(3)
            for index, name in enumerate(WEATHER):
                with weather_columns[index % 3]:
                    values[name] = st.number_input(
                        AIR_LABELS[name], min_value=0.0 if name in ("RAIN", "WSPM") else None,
                        value=source_number(example, name), step=1.0 if name == "PRES" else 0.1,
                        key=f"air_{mode}_{name}",
                    )
            time_columns = st.columns(3)
            with time_columns[0]:
                month = int(example["month"]) if example_mode else None
                values["month"] = st.selectbox(
                    "Month", list(range(1, 13)), index=month - 1 if month else None,
                    placeholder="Unknown", format_func=lambda number: month_name[number],
                    key=f"air_{mode}_month",
                )
            with time_columns[1]:
                hour = int(example["hour"]) if example_mode else None
                values["hour"] = st.selectbox(
                    "Hour (0–23)", list(range(24)), index=hour if hour is not None else None,
                    placeholder="Unknown", key=f"air_{mode}_hour",
                )
            with time_columns[2]:
                wind_options = list(COMPASS)
                wind = example.get("wd")
                values["wd"] = st.selectbox(
                    "Wind direction", wind_options,
                    index=wind_options.index(wind) if wind in wind_options else None,
                    placeholder="Unknown", key=f"air_{mode}_wind",
                )
        submitted = st.form_submit_button("Find matching cluster", type="primary")

    if not submitted:
        return
    provided_pollutants = [name for name in POLLUTANTS if values[name] is not None]
    if not provided_pollutants:
        st.error("Enter at least one measured pollutant concentration before finding a cluster.")
        return
    reading = pd.DataFrame([values])
    scaled = model["preprocessing"].transform(engineer_features(reading))
    cluster = int(model["kmeans"].predict(scaled)[0])
    profile = model["profile"].loc[cluster]
    provided = sum(value is not None for value in values.values())

    st.divider()
    st.metric("Assigned K-Means cluster", str(cluster))
    input_note = ("Unknown fields were filled by the fitted preprocessing pipeline."
                  if provided < len(values) else "No raw reading fields were missing.")
    st.caption(f"{provided} of {len(values)} raw reading fields supplied. {input_note} "
               "Cluster IDs are arbitrary.")
    if provided < 8:
        st.warning("Many readings are missing. Add measured pollutants, weather, and time where available.")

    if tables["clustering_pca"] is not None and "pca" in model:
        points = tables["clustering_pca"].copy()
        points["Cluster"] = points["K-Means Cluster"].astype(str)
        location = model["pca"].transform(scaled)[0]
        fig = px.scatter(points, x="PC1", y="PC2", color="Cluster", opacity=0.4,
                         title="Your input among historical observations",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        fig.add_scatter(x=[location[0]], y=[location[1]], mode="markers+text",
                        marker=dict(symbol="diamond", size=17, color=CORAL,
                                    line=dict(width=2, color="white")),
                        text=["Your input"], textposition="top center", name="Your input")
        fig.update_layout(xaxis_title="Principal component 1", yaxis_title="Principal component 2")
        draw(fig, height=420)
        st.caption("The points are 3,000 real Beijing observations. The diamond is your input, "
                   "projected with the saved PCA; K-Means itself uses all 17 engineered features.")

    ratios = pd.DataFrame({
        "Pollutant": provided_pollutants,
        "Ratio": [values[name] / profile[name] for name in provided_pollutants],
    })
    fig = px.bar(ratios, x="Pollutant", y="Ratio",
                 title="Entered pollutants compared with this cluster's averages",
                 color_discrete_sequence=[TEAL])
    fig.add_hline(y=1, line_dash="dash", line_color=CORAL)
    fig.update_layout(yaxis_title="Your reading ÷ cluster average")
    draw(fig)
    st.caption("Only pollutants you entered are shown. A ratio of 1 equals the historical cluster average; "
               "it is not a safety threshold.")
    with st.expander("View the cluster's average measurements"):
        st.dataframe(profile.to_frame("Cluster average").T.round(2), width="stretch")


def results_tab(tables: dict) -> None:
    st.subheader("Study results")
    st.caption("Evaluation and plots from the supplied datasets. These describe the fitted models, "
               "not the true outcome of a new input.")
    regression_metrics = tables["regression_metrics"]
    classification_results = tables["classification_results"]
    comparison = tables["clustering_comparison"]
    cols = st.columns(3)
    cols[0].metric("Regression R²", f"{regression_metrics.iloc[0]['R2']:.3f}" if regression_metrics is not None else "—")
    cols[1].metric("Classification weighted F1", f"{classification_results.iloc[0]['Weighted F1']:.3f}" if classification_results is not None else "—")
    cols[2].metric("K-Means silhouette", f"{comparison.iloc[0]['Silhouette Score']:.3f}" if comparison is not None else "—")

    with st.expander("Regression · held-out evaluation"):
        if regression_metrics is not None:
            m = regression_metrics.iloc[0]
            st.write(f"Random Forest on {int(m['Test Rows']):,} held-out buildings: "
                     f"R² {m['R2']:.3f}, RMSE {m['RMSE']:.2f}, MAE {m['MAE']:.2f}.")
        st.caption("This 12-input GUI model is separate from the Review 1 ten-algorithm notebook.")

    with st.expander("Classification · ten-algorithm comparison"):
        if classification_results is not None:
            st.dataframe(classification_results.round(4), width="stretch", hide_index=True)
        st.caption("All algorithms use the same held-out partition; the selected classifier is saved for this GUI.")

    with st.expander("Clustering · algorithms and visualisations"):
        if comparison is not None:
            st.dataframe(comparison.round(3), width="stretch", hide_index=True)
            st.caption("Both final metrics use the same 3,000 scaled observations.")
        if tables["clustering_elbow"] is not None:
            fig = px.line(tables["clustering_elbow"], x="K", y="Inertia", markers=True,
                          title="K-Means elbow curve", color_discrete_sequence=[TEAL])
            fig.update_layout(xaxis_title="Number of clusters", yaxis_title="Inertia")
            draw(fig)
        if tables["clustering_pca"] is not None:
            algorithm = st.radio("Colour by", ["K-Means Cluster", "Agglomerative Cluster"], horizontal=True)
            points = tables["clustering_pca"].copy()
            points["Cluster"] = points[algorithm].astype(str)
            fig = px.scatter(points, x="PC1", y="PC2", color="Cluster", opacity=0.5,
                             title=f"PCA projection · {algorithm}",
                             color_discrete_sequence=px.colors.qualitative.Safe)
            draw(fig, height=420)
            st.caption("PCA displays two components; clustering was fitted on the full engineered feature set.")
        if tables["clustering_linkages"] is not None:
            st.markdown("**Hierarchical linkage comparison**")
            st.dataframe(tables["clustering_linkages"].round(3), width="stretch", hide_index=True)

    with st.expander("Methods and limits"):
        st.write("Regression uses a Random Forest on 12 building and weather fields. "
                 "Classification uses the selected Random Forest after comparing ten algorithms. "
                 "Clustering uses K-Means on pollutant, weather, and cyclic time features.")
        st.write("Missing inputs are filled by preprocessing fitted on the study data. "
                 "The water labels and cluster IDs are dataset-specific. These outputs are exploratory "
                 "and are not certifications, laboratory reports, or health advice.")


def main() -> None:
    st.set_page_config(page_title="Machine Learning Case Study", page_icon="◉", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown("""
    <style>
      .stApp {background: #fbfcfa; color: #18313c;}
      .block-container {max-width: 1080px; padding-top: 2rem; padding-bottom: 4rem;}
      h1, h2, h3 {letter-spacing: -.025em; color: #18313c;}
      h1 {font-size: 2.15rem !important;}
      div[data-testid="stMetric"] {background: white; border: 1px solid #e4ebe7;
        border-radius: 10px; padding: .9rem 1.1rem;}
      div[data-testid="stForm"] {background: white; border: 1px solid #e4ebe7;
        border-radius: 12px; padding: 1rem 1.25rem;}
    </style>
    """, unsafe_allow_html=True)
    st.title("Machine learning case study")
    st.caption("Three prediction tasks · supplied research datasets · local exploratory app")
    st.divider()

    building = load_model("regression_gui_model.joblib")
    water = load_model("classification_model.joblib")
    air = load_model("clustering_model.joblib")
    tables = load_tables()
    view = st.segmented_control(
        "Prediction task", ["Regression", "Classification", "Clustering", "Study results"],
        default="Regression", key="current_view", label_visibility="collapsed", width="stretch",
    )
    if view == "Regression":
        regression_tab(building, tables)
    elif view == "Classification":
        classification_tab(water, tables)
    elif view == "Clustering":
        clustering_tab(air, tables)
    else:
        results_tab(tables)


if __name__ == "__main__":
    main()
