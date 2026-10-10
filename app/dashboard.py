"""Evidence-led interface for the three saved case-study models."""

from __future__ import annotations

from calendar import month_name
from datetime import date
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.clustering import COMPASS, POLLUTANTS, WEATHER, engineer_features
from src.prediction_policy import (
    classification_issues, clustering_issues,
    present, regression_issues, result_record,
)

ROOT = Path(__file__).resolve().parents[1]
TEAL = "#126c70"
INK = "#142b35"
CORAL = "#d76c53"

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
def load_model(filename: str, modified: int):
    path = ROOT / "models" / filename
    return joblib.load(path) if path.exists() else None


def current_model(filename: str):
    path = ROOT / "models" / filename
    return load_model(filename, path.stat().st_mtime_ns if path.exists() else 0)


@st.cache_data
def model_id(filename: str, modified: int) -> str | None:
    path = ROOT / "models" / filename
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def matching_model(filename: str, expected: str) -> bool:
    path = ROOT / "models" / filename
    return bool(path.exists() and model_id(filename, path.stat().st_mtime_ns) == expected)


@st.cache_data
def load_tables(modified: tuple[int, ...]) -> dict[str, pd.DataFrame | None]:
    names = {
        "regression_metrics": "regression_gui_metrics.csv",
        "regression_test": "regression_gui_test_predictions.csv",
        "classification_results": "classification_comparison.csv",
        "classification_confusion": "classification_gui_confusion.csv",
        "classification_reliability": "classification_reliability.csv",
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
def load_evidence(modified: int) -> dict | None:
    path = ROOT / "results/release_validation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def page_intro(number: str, eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-heading"><div class="page-index">{number} / {eyebrow}</div>'
        f'<h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def side_panel(title: str, lines: list[str], *, footer: str = "") -> None:
    points = "".join(f"<li>{line}</li>" for line in lines)
    bottom = f'<div class="side-footer">{footer}</div>' if footer else ""
    st.markdown(
        f'<aside class="side-panel"><div class="side-label">Before you run it</div>'
        f'<h3>{title}</h3><ul>{points}</ul>{bottom}</aside>',
        unsafe_allow_html=True,
    )


def result_intro(title: str, text: str) -> None:
    st.markdown(
        f'<div class="result-heading"><div class="result-label">RESULT / MODEL OUTPUT</div>'
        f'<h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True,
    )


def show_issues(issues: list[str]) -> bool:
    if not issues:
        return False
    st.error("A result was withheld because this input is outside the tested use of the model.")
    for issue in issues:
        st.markdown(f"- {issue}")
    return True


def download_result(record: str, task: str) -> None:
    st.download_button(
        "Download this result", record, file_name=f"{task}-result.json",
        mime="application/json", on_click="ignore", icon=":material/download:",
        key=f"download_{task}",
    )


@st.cache_data
def building_reference(features: tuple[str, ...]):
    """Use actual dataset combinations for the building form."""
    data = pd.read_csv(ROOT / "data" / "regression" / "train.csv", usecols=list(features))
    pairs = data[["facility_type", "building_class"]].dropna().drop_duplicates()
    if pairs.groupby("facility_type")["building_class"].nunique().gt(1).any():
        raise ValueError("One facility type maps to multiple building classes")
    facility_class = pairs.set_index("facility_type")["building_class"].to_dict()
    return facility_class, data.iloc[0].to_dict()


@st.cache_data
def water_example(fields: tuple[str, ...], minimum: int) -> tuple[dict, int]:
    source = ROOT / "data" / "classification" / "water_quality_classification_dataset.csv"
    data = pd.read_csv(source)
    eligible = data[list(fields)].notna().sum(axis=1) >= minimum
    if not eligible.any():
        raise ValueError("No source water row satisfies the release input contract")
    index = int(np.flatnonzero(eligible.to_numpy())[0])
    return data.iloc[index].to_dict(), index + 1


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


def regression_tab(model: dict | None, tables: dict, evidence: dict | None) -> None:
    lead, rail = st.columns([1.8, 1], gap="large")
    with lead:
        page_intro("01", "REGRESSION", "Building energy, estimated.",
                   "Estimate annual Site Energy Use Intensity from a building's size, use, and climate details.")
    with rail:
        side_panel("A measured-input estimate", [
            "Floor area and facility use are required.",
            "At least 8 of 9 numeric fields must be supplied to match the training data's input pattern.",
            "The result is a model estimate in kBtu/ft²/year, not a utility bill.",
        ], footer="DATA / Instructor-supplied building file · 75,757 records")
    if model is None:
        st.error("Building model missing. Run `python -m src.regression`.")
        return
    if evidence is None:
        st.error("Release evidence is missing. Run `python -m src.release_validation` before predicting.")
        return

    facility_class, example_row = building_reference(tuple(model["features"]))
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

        with st.expander("Location and annual weather · usually needed"):
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
    if show_issues(regression_issues(values, evidence, model["numeric"])):
        return
    row = pd.DataFrame([values], columns=model["features"])
    prediction = float(model["model"].predict(row)[0])
    missing = int(row.isna().sum(axis=1).iloc[0])
    radius = evidence["interval"]["radius"]
    lower, upper = max(0, prediction - radius), prediction + radius
    st.divider()
    result_intro("A model estimate, with its measured error range.",
                 "The interval is based on separate internal calibration records. It is a historical 90% error range, not a guarantee for this building.")
    m1, m2, m3 = st.columns([1.5, 1, 1])
    m1.metric("Estimated Site EUI", f"{prediction:.1f} kBtu/ft²/year")
    m2.metric("Lower bound", f"{lower:.1f}")
    m3.metric("Upper bound", f"{upper:.1f}")
    input_note = (f"{missing} of {len(model['features'])} model inputs were missing and imputed."
                  if missing else "All model inputs were supplied.")
    st.caption(f"Building class: {values['building_class']}. {input_note}")
    st.caption(f"Internal evaluation: {evidence['interval']['observed_coverage']:.1%} of "
               f"{evidence['interval']['evaluation_rows']:,} eligible held-out buildings fell within this range. "
               "This has not been validated on a new building population.")
    record = result_record(
        task="building_energy", model_id=evidence["model_id"], values=values,
        imputed_fields=[field for field in model["features"] if not present(values.get(field))],
        result={"site_eui_kbtu_ft2_year": round(prediction, 3),
                "internal_90_percent_lower": round(lower, 3),
                "internal_90_percent_upper": round(upper, 3)},
        limits=["Internal study estimate; not a utility measurement or ENERGY STAR certification.",
                "Interval coverage is an aggregate held-out result, not an individual guarantee."],
    )
    download_result(record, "building-energy")
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


def classification_tab(model: dict | None, calibrated, tables: dict, evidence: dict | None) -> None:
    lead, rail = st.columns([1.8, 1], gap="large")
    with lead:
        page_intro("02", "CLASSIFICATION", "A water dataset label.",
                   "Explore how the fitted model groups a measured sample into the source file's three labels.")
    with rail:
        side_panel("Read the label carefully", [
            "At least 12 measured fields are needed for the evaluated subset.",
            "The original file does not document units or the rules behind Good, Moderate, and Poor.",
            "This label is not a drinking-water or health-safety decision.",
        ], footer="DATA / Instructor-supplied water file · 80,000 raw records")
    if model is None:
        st.error("Water-quality model missing. Run the Review 2 classification notebook.")
        return
    if calibrated is None or evidence is None:
        st.error("Calibrated model or release evidence is missing. Run `python -m src.release_validation`.")
        return

    measurements = [name for name in model["features"] if name not in DATE_FEATURES]
    source, source_row = water_example(tuple(measurements), evidence["measured_fields_required"])
    example_mode = st.toggle("Start from a row in the supplied water dataset", key="water_example")
    example = source if example_mode else {}
    mode = "example" if example_mode else "manual"
    if example_mode:
        st.caption(f"Source row {source_row:,} of `data/classification/water_quality_classification_dataset.csv`. "
                   "Edit any value to create your own input.")

    ranges = {field: (evidence["bounds"][field]["p01"], evidence["bounds"][field]["p99"])
              for field in measurements}
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
    if show_issues(classification_issues(values, evidence, measurements)):
        return
    missing_features = int(row.isna().sum(axis=1).iloc[0])
    probabilities = calibrated.predict_proba(row)[0]
    code = int(np.argmax(probabilities))
    predicted = model["class_names"][code]

    st.divider()
    result_intro("A classification within this dataset.",
                 "Probabilities were calibrated using separate labeled records. They remain estimates about the source labels, not measured water quality.")
    a, b, c = st.columns([1.4, 1, 1])
    a.metric("Predicted dataset class", predicted)
    b.metric("Model probability", f"{probabilities[code]:.1%}")
    c.metric("Measured fields", f"{provided} / {len(measurements)}")
    input_note = (f"{missing_features} model inputs, including any unknown date fields, were imputed."
                  if missing_features else "All model inputs were supplied.")
    st.caption(f"{provided} of {len(measurements)} measurements supplied. {input_note}")
    unusual = [field for field in measurements if pd.notna(values[field])
               and not ranges[field][0] <= values[field] <= ranges[field][1]]
    if unusual:
        st.info("Unusual relative to the supplied data's central 98%: "
                + ", ".join(WATER_LABELS.get(field, pretty_name(field)) for field in unusual)
                + ". Check the entered values and units.")
    poor_recall = evidence["held_out_eligible"]["recall"]["Poor"]
    future_recall = evidence["temporal_stress"]["recall"]["Poor"]
    st.warning(f"On {evidence['held_out_eligible_rows']:,} eligible held-out samples, "
               f"the model recalled {poor_recall:.1%} of the dataset's Poor labels. "
               f"A separate earlier-years-to-later-years check recalled {future_recall:.1%}. "
               "It can miss Poor cases; do not use this output for a safety decision.")
    st.caption("The dataset's label rules and measurement units are undocumented. "
               "The probabilities have no established meaning for a new water source.")
    record = result_record(
        task="water_dataset_class", model_id=evidence["model_id"], values=values,
        imputed_fields=[field for field in model["features"] if not present(values.get(field))],
        result={"dataset_class": predicted,
                "calibrated_model_probabilities": dict(zip(model["class_names"], map(float, probabilities)))},
        limits=["Original measurement units and class-label rules are undocumented.",
                "Not a drinking-water or health-safety assessment.",
                "No external population has been validated."],
    )
    download_result(record, "water-dataset")

    probability_table = pd.DataFrame({"Class": model["class_names"], "Model probability": probabilities})
    fig = px.bar(probability_table, x="Class", y="Model probability", color="Class",
                 title="Calibrated model probabilities for this input",
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


def clustering_tab(model: dict | None, tables: dict, evidence: dict | None) -> None:
    lead, rail = st.columns([1.8, 1], gap="large")
    with lead:
        page_intro("03", "CLUSTERING", "Patterns in Beijing air.",
                   "Place an hourly station-style reading among patterns found in historical pollution and weather data.")
    with rail:
        side_panel("A pattern, not a rating", [
            "Supply at least 4 pollutants, 3 weather values, and 2 time or wind fields.",
            "Cluster IDs are arbitrary; they are not official AQI or health categories.",
            "A reading far outside historical patterns receives no cluster result.",
        ], footer="DATA / UCI Beijing air quality · 12 stations, 2013–2017")
    if model is None:
        st.error("Clustering model missing. Run the Review 2 clustering notebook.")
        return
    if evidence is None:
        st.error("Release evidence is missing. Run `python -m src.release_validation`.")
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
        with st.expander("Weather, observation time, and wind · needed for a reliable match"):
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
    reading = pd.DataFrame([values])
    scaled = model["preprocessing"].transform(engineer_features(reading))
    distance = float(model["kmeans"].transform(scaled).min(axis=1)[0])
    if show_issues(clustering_issues(values, evidence, distance)):
        return
    cluster = int(model["kmeans"].predict(scaled)[0])
    profile = model["profile"].loc[cluster]
    provided = sum(value is not None for value in values.values())

    st.divider()
    result_intro("The closest historical pattern.",
                 "This is an unsupervised assignment among three fitted groups. It does not predict an official air-quality class.")
    a, b, c = st.columns([1.4, 1, 1])
    a.metric("Assigned cluster", f"{cluster + 1} / {model['kmeans'].n_clusters}")
    b.metric("Historical observations", f"{int(profile['Observations']):,}")
    c.metric("Fields supplied", f"{provided} / {len(values)}")
    input_note = ("Unknown fields were filled by the fitted preprocessing pipeline."
                  if provided < len(values) else "No raw reading fields were missing.")
    st.caption(f"{input_note} Display label {cluster + 1} corresponds to saved model cluster ID {cluster}. "
               f"The masked-input agreement check was {evidence['masked_assignment_agreement']:.1%} "
               f"on {evidence['reference_rows']:,} historical rows; this is not ground-truth accuracy.")
    record = result_record(
        task="beijing_air_cluster", model_id=evidence["model_id"], values=values,
        imputed_fields=[field for field in values if not present(values[field])],
        result={"saved_cluster_id": cluster, "display_cluster_number": cluster + 1,
                "distance_to_nearest_centroid": round(distance, 4)},
        limits=["Exploratory pattern assignment, not an air-quality or health category.",
                "Distance threshold comes from historical training-source observations."],
    )
    download_result(record, "beijing-air")

    if tables["clustering_pca"] is not None and "pca" in model:
        points = tables["clustering_pca"].copy()
        points["Cluster"] = "Cluster " + (points["K-Means Cluster"] + 1).astype(str)
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


def results_tab(tables: dict, evidence: dict | None) -> None:
    page_intro("04", "STUDY RESULTS", "The evidence behind each result.",
               "Explore held-out performance, algorithm comparisons, and the limits of these source datasets.")
    st.caption("These aggregate plots describe fitted models. They never reveal the true outcome of a new input.")
    regression_metrics = tables["regression_metrics"]
    classification_results = tables["classification_results"]
    comparison = tables["clustering_comparison"]
    cols = st.columns(3)
    cols[0].metric("Regression R²", f"{regression_metrics.iloc[0]['R2']:.3f}" if regression_metrics is not None else "—")
    cols[1].metric("Classification weighted F1", f"{classification_results.iloc[0]['Weighted F1']:.3f}" if classification_results is not None else "—")
    cols[2].metric("K-Means silhouette", f"{comparison.iloc[0]['Silhouette Score']:.3f}" if comparison is not None else "—")

    with st.expander("Internal reliability checks", expanded=True):
        if evidence is None:
            st.warning("Release evidence has not been generated.")
        else:
            r = evidence["regression"]
            w = evidence["classification"]
            c = evidence["clustering"]
            st.markdown("**Building estimate**")
            st.write(f"The 90% internal error range is ±{r['interval']['radius']:.1f} kBtu/ft²/year. "
                     f"Observed coverage was {r['interval']['observed_coverage']:.1%} on "
                     f"{r['interval']['evaluation_rows']:,} eligible held-out buildings. "
                     f"Commercial MAE: {r['group_mae'][0]['mae']:.1f}; "
                     f"residential MAE: {r['group_mae'][1]['mae']:.1f}.")
            st.markdown("**Water labels**")
            st.write(f"The input gate requires {w['measured_fields_required']} of "
                     f"{w['measured_fields_total']} measured fields. On "
                     f"{w['held_out_eligible_rows']:,} eligible held-out records, "
                     f"accuracy was {w['held_out_eligible']['accuracy']:.1%} and Poor-label recall was "
                     f"{w['held_out_eligible']['recall']['Poor']:.1%}. "
                     f"Multiclass Brier score changed from {w['base_held_out_multiclass_brier']:.3f} "
                     f"to {w['held_out_multiclass_brier']:.3f} after calibration (lower is better).")
            temporal = w["temporal_stress"]
            st.write(f"When a separate model was trained on {temporal['training_rows']:,} records from "
                     f"{temporal['training_years']} and evaluated on {temporal['rows']:,} later records "
                     f"from {temporal['evaluation_years']}, accuracy was {temporal['accuracy']:.1%} "
                     f"and Poor-label recall was {temporal['recall']['Poor']:.1%}. "
                     "This same-file time split is a stress test, not external validation.")
            reliability = tables["classification_reliability"]
            if reliability is not None:
                fig = px.scatter(reliability, x="Mean model confidence", y="Observed accuracy",
                                 size="Samples", hover_name="Confidence band",
                                 title="Confidence versus observed accuracy on held-out labels",
                                 color_discrete_sequence=[TEAL])
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(color=CORAL, dash="dash"))
                fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                                  xaxis_range=[0, 1], yaxis_range=[0, 1])
                draw(fig)
                st.caption("Bubble size is the number of held-out records in each confidence band. "
                           "Calibration on this dataset does not establish reliability on a new water source.")
            st.markdown("**Air clusters**")
            st.write(f"With 4 pollutants, 3 weather fields, and 2 time or wind fields retained, "
                     f"the assignment matched the same record's complete-data cluster "
                     f"{c['masked_assignment_agreement']:.1%} of the time across "
                     f"{c['mask_trials']} masking trials on {c['reference_rows']:,} historical records. "
                     "This is an input-stability check, not a ground-truth score.")
            st.warning("External validation has not been performed. Water units and label rules remain undocumented; "
                       "these results are research outputs, not operational safety assessments.")

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
            points["Cluster"] = "Cluster " + (points[algorithm] + 1).astype(str)
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
    st.set_page_config(page_title="Fieldnote | Applied ML", page_icon="◉", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown("""
    <style>
      :root {--ink:#142b35; --muted:#667b80; --teal:#126c70; --line:#dce6e4; --paper:#f5f8f6;}
      html, body, [class*="css"] {font-family: "Avenir Next", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
      .stApp {background:var(--paper); color:var(--ink);}
      .block-container {max-width:1190px; padding:2.4rem 2.2rem 5rem;}
      header[data-testid="stHeader"] {background:transparent;}
      h1,h2,h3,h4 {letter-spacing:-.035em; color:var(--ink);}
      p,li {line-height:1.62;}
      .masthead {display:flex; justify-content:space-between; align-items:center; padding-bottom:1.05rem;
        border-bottom:1px solid var(--line); gap:1rem;}
      .brand {display:flex; align-items:center; gap:.68rem; color:var(--ink); font-size:.92rem; font-weight:780;
        letter-spacing:.1em; text-transform:uppercase;}
      .brand-mark {display:grid; place-items:center; width:2.05rem; height:2.05rem; background:var(--teal);
        color:white; border-radius:.52rem; font-size:1.12rem; font-weight:600;}
      .mast-meta {font-size:.69rem; letter-spacing:.14em; color:var(--muted); font-weight:750; text-transform:uppercase;}
      .hero {padding:3.3rem 0 2.35rem; max-width:840px;}
      .hero-kicker,.page-index,.result-label,.side-label {font-size:.68rem; letter-spacing:.16em;
        text-transform:uppercase; font-weight:800; color:var(--teal);}
      .hero h1 {font-size:clamp(2.75rem,5vw,4.6rem); line-height:1.06; letter-spacing:-.065em;
        margin:.65rem 0 1.05rem; font-weight:760;}
      .hero p {font-size:1.13rem; color:#586e73; line-height:1.6; max-width:720px; margin:0;}
      .model-strip {display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.55rem;}
      .model-strip span {padding:.42rem .72rem; border:1px solid var(--line); background:#fff; border-radius:999px;
        color:#426269; font-size:.73rem; font-weight:720; letter-spacing:.03em;}
      .page-heading {padding:2.65rem 0 1.2rem;}
      .page-heading h2 {font-size:clamp(2rem,3.1vw,2.9rem); line-height:1.12; margin:.6rem 0 .75rem;}
      .page-heading p {font-size:1rem; color:#61777c; max-width:690px; margin:0;}
      .side-panel {margin-top:2.35rem; padding:1.45rem 1.55rem 1.25rem; border-radius:17px;
        background:#173a43; color:#e4efeb; box-shadow:0 14px 32px rgba(18,48,56,.12);}
      .side-panel h3 {color:white; font-size:1.18rem; margin:.55rem 0 .75rem;}
      .side-panel .side-label {color:#a8dcd2;}
      .side-panel ul {margin:.5rem 0 1rem; padding-left:1.1rem;}
      .side-panel li {font-size:.82rem; line-height:1.5; margin:.45rem 0; color:#d9e9e7;}
      .side-footer {border-top:1px solid #42646a; padding-top:.8rem; color:#b9d5d1; font-size:.64rem;
        letter-spacing:.045em; text-transform:uppercase;}
      .result-heading {padding:.5rem 0 .8rem;}
      .result-heading h3 {font-size:1.75rem; margin:.48rem 0 .38rem;}
      .result-heading p {font-size:.92rem; color:var(--muted); margin:0; max-width:720px;}
      div[data-testid="stMetric"] {background:white; border:1px solid var(--line); border-radius:14px;
        padding:1rem 1.15rem; min-height:6.25rem; box-shadow:0 5px 17px rgba(25,62,68,.035);}
      div[data-testid="stMetricLabel"] {color:#62767a; font-size:.77rem; font-weight:680;}
      div[data-testid="stMetricValue"] {color:var(--ink); letter-spacing:-.045em;}
      div[data-testid="stForm"] {background:#fff; border:1px solid var(--line); border-radius:18px;
        padding:1.45rem 1.6rem 1.55rem; box-shadow:0 10px 30px rgba(25,62,68,.045);}
      div[data-testid="stForm"] p strong {font-size:.92rem; color:#193b44;}
      .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {background:#126c70;
        border-color:#126c70; border-radius:10px; font-weight:700; padding:.5rem 1.1rem;}
      .stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {
        background:#0b5559; border-color:#0b5559;}
      .stDownloadButton button {border-radius:10px; font-weight:680; border-color:#b8d1cd; color:#145e62;}
      div[data-testid="stPlotlyChart"] {background:white; border:1px solid var(--line);
        border-radius:15px; padding:.55rem; overflow:hidden;}
      div[data-testid="stExpander"] {border:1px solid var(--line); border-radius:12px; background:#fff;}
      div[data-testid="stAlert"] {border-radius:11px;}
      div[data-testid="stSegmentedControl"] {padding:.28rem; background:white; border:1px solid var(--line);
        border-radius:13px; box-shadow:0 5px 18px rgba(25,62,68,.035);}
      div[data-testid="stSegmentedControl"] button {border-radius:9px; font-weight:700;}
      .footer-note {border-top:1px solid var(--line); margin-top:3.5rem; padding-top:1.2rem;
        color:#728489; font-size:.74rem; display:flex; justify-content:space-between; gap:1rem;}
      @media(max-width:760px) {.block-container{padding:1.2rem 1rem 3rem;}.hero{padding:2.3rem 0 1.7rem;}
        .mast-meta{display:none;}.side-panel{margin-top:0;}.page-heading{padding:2rem 0 .6rem;}
        div[data-testid="stForm"]{padding:1rem;}.footer-note{display:block;}}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('''<div class="masthead"><div class="brand"><span class="brand-mark">◉</span>Fieldnote</div>
      <div class="mast-meta">Applied machine learning / Review 2</div></div>
      <div class="hero"><div class="hero-kicker">CASE STUDY / INTERACTIVE LAB</div>
      <h1>Models made<br>understandable.</h1>
      <p>Explore three distinct prediction tasks with real source data, visible model evidence,
      and clear boundaries around what each result means.</p>
      <div class="model-strip"><span>01 · BUILDING ENERGY</span><span>02 · WATER LABELS</span>
      <span>03 · AIR PATTERNS</span></div></div>''', unsafe_allow_html=True)

    view = st.segmented_control(
        "Prediction task", ["Regression", "Classification", "Clustering", "Study results"],
        default="Regression", key="current_view", label_visibility="collapsed", width="stretch",
    )
    result_files = sorted((ROOT / "results").glob("*.csv"))
    tables = load_tables(tuple(path.stat().st_mtime_ns for path in result_files))
    evidence_path = ROOT / "results/release_validation.json"
    evidence = load_evidence(evidence_path.stat().st_mtime_ns if evidence_path.exists() else 0)
    if view == "Regression":
        relevant = evidence["regression"] if evidence else None
        if relevant and not matching_model("regression_gui_model.joblib", relevant["model_id"]):
            st.error("The building model changed after validation. Run `python -m src.release_validation` again.")
        else:
            regression_tab(current_model("regression_gui_model.joblib"), tables, relevant)
    elif view == "Classification":
        relevant = evidence["classification"] if evidence else None
        if relevant and (not matching_model("classification_model.joblib", relevant["base_model_id"])
                         or not matching_model("classification_calibrated.joblib", relevant["model_id"])):
            st.error("The water model changed after calibration. Run `python -m src.release_validation` again.")
        else:
            classification_tab(current_model("classification_model.joblib"),
                               current_model("classification_calibrated.joblib"), tables, relevant)
    elif view == "Clustering":
        relevant = evidence["clustering"] if evidence else None
        if relevant and not matching_model("clustering_model.joblib", relevant["model_id"]):
            st.error("The air model changed after validation. Run `python -m src.release_validation` again.")
        else:
            clustering_tab(current_model("clustering_model.joblib"), tables, relevant)
    else:
        results_tab(tables, evidence)
    st.markdown('''<div class="footer-note"><span>Fieldnote / Machine learning case study</span>
      <span>Research interface · Results are not operational safety assessments</span></div>''',
      unsafe_allow_html=True)


if __name__ == "__main__":
    main()
